import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from models import SDARForCausalLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from datasets import Dataset
from itertools import chain

from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


# -------------------------------
# helpers
# -------------------------------

def collapse_k_unique(lst, k: int):
    if k <= 0:
        raise ValueError("k must be > 0")
    uniq = sorted(set(lst))
    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        rep = uniq[end_idx]
        mapping[val] = rep
    return [mapping[x] for x in lst]

def prepare_pretrain_packed_concat(ds, tokenizer, chunk_size, add_eos=True, eos_fallback=1):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = eos_fallback

    def tok_fn(examples):
        out = tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
        )
        ids_list = out["input_ids"]
        if add_eos:
            ids_list = [ids + [eos_id] for ids in ids_list]
        return {"input_ids": ids_list}

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    all_ids = list(chain.from_iterable(tokenized["input_ids"]))
    total_len = (len(all_ids) // chunk_size) * chunk_size
    if total_len == 0:
        raise ValueError("No chunk formed. Decrease chunk_size or increase data")
    all_ids = all_ids[:total_len]
    chunks = [all_ids[i:i + chunk_size] for i in range(0, total_len, chunk_size)]
    return Dataset.from_dict({"input_ids": chunks})


def make_basic_block_attention_additive(N: int, start_pos: int, block_size: int, device=None, dtype=torch.float32):
    """
    返回加性 bias: [1,1,N,N]，允许=0 禁止=-inf
    约定 N = L0 + 2*L1
    """
    B = 1
    L0 = start_pos
    L1 = (N - L0) // 2
    assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

    allow = torch.zeros((B, 1, N, N), dtype=torch.bool, device=device)

    rows_ext   = torch.arange(L0 + L1, L0 + 2 * L1, device=device)  # 扩展段查询行
    rows_token = torch.arange(L0, L0 + L1, device=device)           # 原响应段查询行

    for bi in range((L1 + block_size - 1) // block_size):
        left_end    = L0 + min(bi * block_size, L1)
        right_start = L0 + L1 + (left_end - L0)

        i_start = bi * block_size
        i_end   = min((bi + 1) * block_size, L1)

        # 扩展段查询
        block_rows = rows_ext[i_start:i_end]
        allow[:, :, block_rows.unsqueeze(-1), 0:left_end] = True
        allow[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = True

        # 原响应段查询
        block_rows = rows_token[i_start:i_end]
        left_end2  = L0 + min((bi + 1) * block_size, L1)
        allow[:, :, block_rows.unsqueeze(-1), 0:left_end2] = True

    if L0 > 0:
        for bi in range((L0 + block_size - 1) // block_size):
            row_end   = max(L0 - bi * block_size, 0)
            row_start = max(L0 - (bi + 1) * block_size, 0)
            if row_end > row_start:
                block_rows = torch.arange(row_start, row_end, device=device)
                allow[:, :, block_rows.unsqueeze(-1), 0:row_end] = True

    bias = torch.zeros_like(allow, dtype=dtype)
    bias[~allow] = float("-inf")
    return bias  # [1,1,N,N]


def process_pad_additive(additive_bias, input_ids, pad_id, start_pos):
    """
    additive_bias: [B,1,N,N] 允许=0 禁止=-inf
    对应 key 为 pad 的列置为 -inf
    若某查询整行为 -inf，允许其自环为 0 以避免全 -inf
    """
    B, _, N, _ = additive_bias.shape
    device = input_ids.device
    additive_bias = additive_bias.to(device)

    key_is_pad = (input_ids == pad_id)  # [B,N]
    additive_bias.masked_fill_(key_is_pad[:, None, None, :], float("-inf"))

    A = additive_bias[:, 0]  # [B,N,N]
    all_neg_inf = torch.isneginf(A).all(dim=-1)  # [B,N]
    if all_neg_inf.any():
        b_idx, r_idx = all_neg_inf.nonzero(as_tuple=True)
        A[b_idx, r_idx, r_idx] = 0.0
    return additive_bias


def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
    device = input_ids_b.device
    NB = (L1 + block_size - 1) // block_size

    step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
    step_pad[:L1] = step_map_b
    step_blk = step_pad.view(NB, block_size)

    valid = step_blk.ge(0)
    big = torch.iinfo(step_blk.dtype).max
    tmp = step_blk.masked_fill(~valid, big)
    min_vals, _ = tmp.min(dim=1, keepdim=True)

    pmask_blk = step_blk.eq(min_vals) & valid
    if not pmask_blk.any():
        return None, None, step_map_b, False

    ge_mask_blk = step_blk.ge(min_vals) & valid

    pmask_tail = pmask_blk.view(-1)[:L1]
    ge_mask_tail = ge_mask_blk.view(-1)[:L1]

    pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
    pmask_b[L0:] = pmask_tail

    tail = input_ids_b[L0:L0 + L1].clone()
    tail[ge_mask_tail] = mask_id

    extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
    extended_input_ids_b[:L0 + L1] = input_ids_b
    extended_input_ids_b[L0 + L1:] = tail

    new_step_map_b = step_map_b.clone()
    new_step_map_b[pmask_tail] = -1

    return extended_input_ids_b, pmask_b, new_step_map_b, True


def collect_training_data(input_ids, step_map_list, start_pos, mask_id, pad_id, config):
    B, L = input_ids.shape
    L0 = start_pos
    L1 = L - L0
    block_size = config.training.block_size

    lower = config.training.lower_p
    upper = config.training.upper_p

    if config.training.method == "semi-ar":
        extended_input_ids_list, pmask_list = [], []
        for b in range(B):
            prob_ramp = torch.empty(L1, device=input_ids.device).uniform_(lower, upper)
            rand_tail = torch.rand(L1, device=input_ids.device)
            pmask_tail = rand_tail <= prob_ramp

            pmask_b = torch.cat([
                torch.zeros(L0, dtype=torch.bool, device=input_ids.device),
                pmask_tail
            ], dim=0)

            noise_tail = input_ids[b, L0:].clone()
            noise_tail.masked_fill_(pmask_tail, mask_id)
            extended_b = torch.cat([input_ids[b], noise_tail], dim=0)

            extended_input_ids_list.append(extended_b)
            pmask_list.append(pmask_b)

    elif config.training.method == "trace":
        for b in range(B):
            step_map_i = step_map_list[b]
            for j in range(int((L1 - 1) / block_size) + 1):
                s = j * block_size
                e = min(L1, (j + 1) * block_size)
                step_map_list[b][s:e] = collapse_k_unique(step_map_i[s:e], config.training.shrink)

        step_map = torch.as_tensor(step_map_list, dtype=torch.long, device=input_ids.device)
        assert step_map.shape[1] == L1

        extended_input_ids_list, pmask_list = [], []
        for b in range(B):
            step_b = step_map[b]
            while True:
                out = one_round_vectorized(
                    input_ids_b=input_ids[b],
                    step_map_b=step_b,
                    L0=L0,
                    L1=L1,
                    block_size=block_size,
                    mask_id=mask_id,
                )
                extended_b, pmask_b, step_b, has_any = out
                if not has_any:
                    break
                extended_input_ids_list.append(extended_b)
                pmask_list.append(pmask_b)
    else:
        raise ValueError(f"Unknown training.method: {config.training.method}")

    extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
    p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)

    # 截断 pad 的 post 区
    pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask
    post_num = config.training.post_num
    if post_num is not None:
        cum_pad = torch.cumsum(pad_resp.int(), dim=1)
        p_mask &= ~(pad_resp & (cum_pad > post_num))

    labels = extended_input_ids[:, :L].clone()

    idx = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
    valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
    tok_idx = valid.long().cumsum(dim=-1) - 1
    tok_idx = tok_idx.masked_fill(~valid, 1)
    tok_idx_resp = tok_idx[:, start_pos:]
    tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

    keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
    extended_input_ids = extended_input_ids[keep]
    p_mask            = p_mask[keep]
    tok_idx_ext       = tok_idx_ext[keep]
    labels            = labels[keep]

    return extended_input_ids, p_mask, tok_idx_ext, labels


# -------------------------------
# main
# -------------------------------

def main():
    # config, accelerator, logging
    config = get_config()
    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # model and tokenizer
    logger.info("Loading models and optimizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id  = tokenizer.pad_token_id

    # optimizer
    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if config.optimizer.name == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer.name} not supported")

    # dataset
    import datasets as hfds
    raw_ds = hfds.load_dataset("hendrydong/fineweb-edu-10BT", split="val")
    chunk_size = config.data.chunk_size
    packed_ds  = prepare_pretrain_packed_concat(
        ds=raw_ds,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        add_eos=True
    )

    # constants for this pretrain setting
    start_pos = 0
    L0 = start_pos
    L1 = chunk_size  # packed sample length
    N  = L0 + 2 * L1

    # 构造基础加性 mask（在 device 上，dtype 与模型一致）
    base_bias = make_basic_block_attention_additive(
        N=N,
        start_pos=L0,
        block_size=config.training.block_size,
        device=accelerator.device,
        dtype=next(model.parameters()).dtype
    )

    # collate: 把一个 batch 的 input_ids 变成训练所需四元组
    need_step = (config.training.method == "trace")

    def collate_build_batch(batch):
        # batch: list of dict with "input_ids" list[int] of length L1
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long, device=accelerator.device)  # [B,L1]

        if need_step:
            step_map_list = [list(range(input_ids.size(1))) for _ in range(input_ids.size(0))]
        else:
            step_map_list = [list(range(input_ids.size(1))) for _ in range(input_ids.size(0))]

        with torch.no_grad():
            extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data(
                input_ids=input_ids,
                step_map_list=step_map_list,
                start_pos=start_pos,
                mask_id=mask_id,
                pad_id=pad_id,
                config=config
            )
        # 注意：collect_training_data 可能因为 keep 过滤导致 batch size 变化
        # base_bias 复制到当前 batch
        B_eff = extended_input_ids.size(0)
        attn_bias = base_bias.repeat(B_eff, 1, 1, 1)
        attn_bias = process_pad_additive(attn_bias, extended_input_ids, pad_id, start_pos)

        return {
            "extended_input_ids": extended_input_ids,   # [B_eff, N]
            "p_mask": p_mask,                           # [B_eff, L1]
            "tok_idx_ext": tok_idx_ext,                 # [B_eff, N]
            "labels": labels,                           # [B_eff, L1]
            "attn_bias": attn_bias,                     # [B_eff,1,N,N]
        }

    # dataloader
    train_dataloader_lm = DataLoader(
        packed_ds,
        batch_size=config.training.batch_size_lm,
        shuffle=True,
        collate_fn=collate_build_batch,
        num_workers=0
    )

    # max steps
    # 每个样本都能至少产生一个扩展条目，这里用近似估计步数
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(packed_ds) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    # prepare
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    # training
    logger.info("***** Running training *****")
    logger.info(f"  Num training chunks = {len(packed_ds)}")
    logger.info(f"  Num training steps  = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (parallel with accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F
    from tqdm.auto import tqdm

    def forward_process(batch):
        extended_input_ids = batch["extended_input_ids"]
        p_mask = batch["p_mask"]
        tok_idx_ext = batch["tok_idx_ext"]
        attn_bias = batch["attn_bias"]
        labels = batch["labels"]

        B_eff, L = p_mask.shape
        L0 = start_pos
        L1 = L

        out = model(
            input_ids=extended_input_ids,
            attention_mask=attn_bias,   # 加性 bias
            position_ids=tok_idx_ext
        )
        logits = out.logits  # [B_eff, N, V]
        logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1:, :]], dim=1)  # [B_eff, L1, V]

        log_probs = F.log_softmax(logits, dim=-1)
        logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B_eff, L1]
        loss_lm = - (logp_tok * p_mask).sum(dim=1) / L1
        loss_lm = loss_lm.mean()
        return loss_lm

    step_full = 0
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True
        )

        for step, batch in enumerate(progress_bar, start=1):
            data_time_m.update(time.time() - end)

            # 所有张量已在 collate 放到正确 device
            loss_lm = forward_process(batch)

            accelerator.backward(loss_lm)

            accelerator.log(
                {
                    "train/loss_lm": loss_lm.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step_global": step_full,
                    "train/epoch": epoch + step / len(progress_bar),
                },
                step=step_full,
            )

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            step_full += 1
            end = time.time()

    accelerator.wait_for_everyone()
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)

            for modname in modules:
                try:
                    import importlib, inspect, glob, os
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)
                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f"Skip copying from module {modname}: {e}")

            logger.info(f"Copied {copied} custom module files into {dst_dir}")

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")


if __name__ == "__main__":
    main()
