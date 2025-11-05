import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"


from tqdm import tqdm
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

import os 
os.environ["TOKENIZERS_PARALLELISM"]="false"

from collections import deque
from typing import Dict, Iterable, Iterator, List, Optional

def order_to_attention_mask(order, bias, K):
    bias_shift = bias.clone()
    B, _, N, _ = bias_shift.shape
    index_set = set()
    if K > 0:
        for j in range(N//2, N):
            index_idx = j - N//2
            order_idx = order[index_idx].item()
            for k in range(1, K+1):
                index1 = j - k
                index2 = j + k
                if index1 >= N//2 and bias_shift[:, :, j, index1-N//2] == 1:
                    if index1-N//2 not in index_set:
                        bias_shift[:, :, j, index1] = 1
                        bias_shift[:, :, j, index1-N//2] = 0
                if index2 < N and bias_shift[:, :, j, index2-N//2] == 1:
                    if index2-N//2 not in index_set:
                        bias_shift[:, :, j, index2] = 1
                        bias_shift[:, :, j, index2-N//2] = 0
            index_set.add(order_idx)
    return bias_shift

@torch.no_grad()
def sample_order_anchor_lookahead_fast(
    T: int,
    K: int = 16,        # 只向右看的窗口宽度
    beta: float = 0.7,  # 距离衰减强度：大→更偏好就近
    device="cpu",
):
    assert T > 0 and K > 0
    K = min(K, T)
    chosen = torch.zeros(T, dtype=torch.bool, device=device)

    # 预计算距离对应的对数权重（logits），Gumbel Max 不需要归一化
    dists = torch.arange(K, device=device)
    logits_by_dist = -beta * dists  # shape [K]

    # 结果预分配
    order = torch.empty(T, dtype=torch.long, device=device)
    wptr = 0

    t = 0  # 锚点 = 当前最左未选
    for _ in range(T):
        # 推到当前最左未选（总共最多前进 T 次）
        while t < T and chosen[t]:
            t += 1
        if t >= T:
            break

        # 当前窗口 [t, R]
        R = min(T - 1, t + K - 1)
        wlen = R - t + 1

        # 窗口内未选 mask
        window_mask = ~chosen[t : R + 1]                 # [wlen]
        if not window_mask.any():
            # 窗口被选空，锚点跳到窗口右侧再继续
            t = R + 1
            continue

        # 取对应距离的 logits（长度 wlen），把已选位置置为 -inf
        logits = logits_by_dist[:wlen].clone()           # [wlen]
        logits[~window_mask] = float("-inf")

        # Gumbel-Max 采样：argmax(logits + Gumbel)
        g = -torch.log(-torch.log(torch.rand_like(logits)))
        pick_off = torch.argmax(logits + g)              # 标量张量（仍在 device 上）
        i_star = t + pick_off

        # 记录与标记
        chosen[i_star] = True
        order[wptr] = i_star
        wptr += 1

        # 若恰好选中锚点，则推进锚点到下一个未选
        if i_star == t:
            while t < T and chosen[t]:
                t += 1

    return order[:wptr]


import torch

def _segments_from_labels_1d(labels: torch.Tensor):
    """
    把一维 labels 按相邻相同值切成连续段，返回每段的 [start, end) 边界列表。
    """
    assert labels.dim() == 1
    L = labels.numel()
    if L == 0:
        return []
    # 边界：当前位置与前一位置的 label 不同
    diff = torch.ones(L, dtype=torch.bool, device=labels.device)
    diff[1:] = labels[1:] != labels[:-1]
    starts = torch.nonzero(diff, as_tuple=False).flatten()
    # 末尾补一个 L 作为右边界
    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = L
    return [(int(s.item()), int(e.item())) for s, e in zip(starts, ends)]


@torch.no_grad()
def shuffle_within_segments(
    labels: torch.Tensor,          # (L,) 或 (B, L) 的整型张量，连续相同值构成一个 segment
    pos_tail: torch.Tensor,        # 与 labels 同形
    K: int = 4,
    beta: float = 0.1,
):
    """
    返回：
      indice: 与 labels 同形的 long 张量，每行是 [0..L-1] 的全局置换，且仅在各 segment 内打乱
      labels_shuf  = labels.gather(最后一维, indice)
      pos_tail_shuf= pos_tail.gather(最后一维, indice)
    """
    device = labels.device
    assert labels.shape == pos_tail.shape
    assert labels.dim() in (1, 2)

    def shuffle_1d(lbl_1d: torch.Tensor, pos_1d: torch.Tensor):
        L = lbl_1d.numel()
        if L == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        # 初始化为身份置换
        idx = torch.arange(L, device=device, dtype=torch.long)

        for s, e in _segments_from_labels_1d(lbl_1d):
            T = e - s
            if T <= 1:
                continue
            # 每段独立调用你的采样器，注意 K 不能超过段长
            order_local = sample_order_anchor_lookahead_fast(
                T=T, K=min(K, T), beta=beta, device=device
            )                              # 长度 T，取值在 [0..T-1]
            idx[s:e] = idx[s:e][order_local]

        return idx

    if labels.dim() == 1:
        indice = shuffle_1d(labels, pos_tail)
        labels_shuf = labels[indice]
        pos_tail_shuf = pos_tail[indice]
        return indice, labels_shuf, pos_tail_shuf

    # batched: (B, L)
    B, L = labels.shape
    indice_list = []
    for b in range(B):
        idx_b = shuffle_1d(labels[b], pos_tail[b])
        indice_list.append(idx_b)
    indice = torch.stack(indice_list, dim=0)                    # (B, L)

    # 按最后一维 gather
    gather_dim = -1
    #labels_shuf   = torch.gather(labels,   gather_dim, indice)
    pos_tail_shuf = torch.gather(pos_tail, gather_dim, indice)
    return indice, pos_tail_shuf


def iter_fixed_chunks(
    tokenized: Iterable[Dict[str, List[int]]],
    chunk_size: int,
    drop_last: bool = True,
    with_tqdm: bool = True,
    desc: str = "Processing chunks",
) -> Iterator[Dict[str, List[int]]]:
    """
    按 token 级拼接流式地产生定长 chunk。
    - 仅单次遍历 tokenized
    - 用 deque 避免反复切片拷贝
    - 支持是否丢弃最后不足 chunk_size 的尾块
    """
    assert chunk_size > 0, "chunk_size must be positive"

    buf_ids = deque()
    buf_segs = deque()
    iterator = tokenized
    if with_tqdm:
        iterator = tqdm(iterator, desc=desc, unit="docs")

    for item in iterator:
        ids = item["input_ids"]
        segs = item.get("segment_ids", None)

        # 基本健壮性检查
        if segs is not None and len(ids) != len(segs):
            raise ValueError("input_ids and segment_ids length mismatch in an item")

        buf_ids.extend(ids)
        if segs is not None:
            buf_segs.extend(segs)
        else:
            # 若没有 segment_ids，就用 0 填（或改成你需要的规则）
            buf_segs.extend([0] * len(ids))

        # 够大就吐块：从左侧 popleft，避免拷贝
        while len(buf_ids) >= chunk_size:
            out_ids = [buf_ids.popleft() for _ in range(chunk_size)]
            out_segs = [buf_segs.popleft() for _ in range(chunk_size)]
            yield {"input_ids": out_ids, "segment_ids": out_segs}

    # 处理尾巴
    if not drop_last and len(buf_ids) > 0:
        out_ids = list(buf_ids)
        out_segs = list(buf_segs)
        yield {"input_ids": out_ids, "segment_ids": out_segs}


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
        # 为每条样本分配一个 doc 段号，整条样本同一段号
        seg_list = [ [i] * len(ids) for i, ids in enumerate(ids_list) ]
        return {"input_ids": ids_list, "segment_ids": seg_list}

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names, num_proc=128, batch_size=1000)

    # 使用生成器分批处理，避免一次性加载所有数据到内存
    features = {
        "input_ids": Sequence(Value("int32")),
        "segment_ids": Sequence(Value("int32")),
    }
    ds_chunks = Dataset.from_generator(
        lambda: iter_fixed_chunks(tokenized, chunk_size=4096, drop_last=True, with_tqdm=False),
        features=Features(features),
    )
    return ds_chunks


def make_basic_block_attention_additive(N: int, start_pos: int, block_size: int, device=None, dtype=torch.float32):
    """
    返回加性 bias: [1,1,N,N]，允许=1 禁止=0
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
    bias[allow] = 1
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
    additive_bias.masked_fill_(key_is_pad[:, None, None, :], 0)

    A = additive_bias[:, 0]  # [B,N,N]
    all_neg_inf = torch.isneginf(A).all(dim=-1)  # [B,N]
    if all_neg_inf.any():
        b_idx, r_idx = all_neg_inf.nonzero(as_tuple=True)
        A[b_idx, r_idx, r_idx] = 0.0
    return additive_bias

def make_basic_window_attention_additive(N: int, start_pos: int, block_size: int, device=None, dtype=torch.float32):
    """
    ABC 划分：N = L0 + 2*L1
    - A: [0, L0)
    - B: [L0, L0+L1)
    - C: [L0+L1, L0+2*L1)

    规则：
    - A/B 的查询行：保持与原代码一致（分块推进）
    - C 的查询行（索引 i ∈ [0, L1)）：
        * 能看到全部 A
        * 在 B 中能看到：< i 的所有，以及 > i 的接下来的 k 个（k = block_size），但看不到第 i 个本位
        * 在 C 中只能看到自己
    """
    BATCH = 1
    L0 = start_pos
    L1 = (N - L0) // 2
    assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

    allow = torch.zeros((BATCH, 1, N, N), dtype=torch.bool, device=device)

    # --- 索引区间 ---
    A_start, A_end = 0, L0
    B_start, B_end = L0, L0 + L1
    C_start, C_end = L0 + L1, L0 + 2 * L1

    rows_A = torch.arange(A_start, A_end, device=device)
    rows_B = torch.arange(B_start, B_end, device=device)
    rows_C = torch.arange(C_start, C_end, device=device)

    # =========================================================
    # 1) A、B 查询行：沿用原始源码行为（分块推进）
    # =========================================================

    # C 的原始分块行为被新的 C 规则替换，所以这里只处理 A/B 和 A 的三角前缀

    # 对 B 查询行：原逻辑
    k = 0#block_size - 1
    block_size = 1
    if L1 > 0 and block_size > 0:
        for bi in range((L1 + block_size - 1) // block_size):
            i_start = bi * block_size
            i_end   = min((bi + 1) * block_size, L1)

            block_rows = rows_B[i_start:i_end]
            # 该块内 B 行能看到从开头到 left_end2 的前缀（含 A + B 的一部分）
            left_end2  = L0 + min((bi + 1) * block_size, L1)
            # 注意：这里 left_end2 是绝对列索引的右开界
            allow[:, :, block_rows.unsqueeze(-1), 0:left_end2] = True

    # 对 A 查询行：原逻辑（A 内部前缀可见，形成上三角）
    if L0 > 0 and block_size > 0:
        for bi in range((L0 + block_size - 1) // block_size):
            row_end   = max(L0 - bi * block_size, 0)
            row_start = max(L0 - (bi + 1) * block_size, 0)
            if row_end > row_start:
                block_rows = torch.arange(row_start, row_end, device=device)
                # A 行能看到从开头到自身行号的前缀
                allow[:, :, block_rows.unsqueeze(-1), 0:row_end] = True

    # =========================================================
    # 2) C 查询行：按你的新规则
    #    k 使用 block_size（可为 0）
    # =========================================================

    for i in range(L1):
        r = C_start + i  # C 段第 i 个位置的行

        # 2.1 看到全部 A
        if L0 > 0:
            allow[:, :, r:r+1, A_start:A_end] = True

        # 2.2 B 中看到：< i 的所有，以及 > i 的接下来的 k 个，但看不到第 i 个
        #    把 B 的“第 i 个位置”换成绝对列 L0 + i
        b_i_abs = B_start + i

        #    之前的所有（左开右开边界注意：PyTorch 切片右边是开区间）
        if i > 0:
            allow[:, :, r:r+1, B_start:b_i_abs] = True

        #    之后的 k 个（不含 b_i_abs 自身）
        if k > 0:
            after_start = b_i_abs + 1
            after_end   = min(B_start + L1, b_i_abs + 1 + k)
            if after_start < after_end:
                allow[:, :, r:r+1, after_start+L1:after_end+L1] = True

        # 2.3 C 中只能看到自己
        allow[:, :, r, r] = True

    # 转成加性 bias（允许=1，否则=0）
    bias = torch.zeros_like(allow, dtype=dtype)
    bias[allow] = 1
    return bias


def _softmax_exp_neg(offsets, beta):
    logits = -beta * offsets.to(torch.float32)
    return torch.softmax(logits - logits.max(), dim=0)



def perm_to_mask(perm: torch.Tensor):
    T = perm.numel()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(T, device=perm.device)
    rank = inv
    return rank.unsqueeze(0) > rank.unsqueeze(1)  # [T,T] bool




def segment_positions_from_segids(seg_ids: torch.LongTensor) -> torch.LongTensor:
    """
    seg_ids: [B, L]，同一个原文档段号相同且连续（拼接时的做法）
    返回 pos: [B, L]，每个段内从 0 递增，到下一个段重置为 0
    """
    B, L = seg_ids.shape
    pos = torch.zeros_like(seg_ids)
    # 简洁做法：找段边界 + 累积计数，但要在边界处重置
    # 这里用一个简洁的 for 循环，B 通常不大，且 L 是 chunk_size，性能足够
    for b in range(B):
        s = seg_ids[b]
        p = pos[b]
        p[0] = 0
        for i in range(1, L):
            if s[i] == s[i-1]:
                p[i] = p[i-1] + 1
            else:
                p[i] = 0
    return pos  # [B, L]


def collect_training_data(
    input_ids,              # [B, L]
    step_map_list,          # list(len B) of list[int] (len L)
    start_pos,              # 预训练=0
    mask_id,
    pad_id,
    config,
    segment_ids=None,       # [B, L] LongTensor，必传以便分段位置
    K = config.training.block_size,
):
    assert segment_ids is not None, "segment_ids 不能为空；packing 时需要用于位置重置"
    B, L = input_ids.shape
    L0 = start_pos
    L1 = L - L0
    block_size = config.training.block_size

    lower = config.training.lower_p
    upper = config.training.upper_p

    extended_input_ids_list = []
    pmask_list = []
    seg_kept_list = []   # 每条扩展样本左半段对应的 segment_ids（用于生成扩展段 seg）
    order_list = []
    # -------- semi-ar --------
    if config.training.method == "semi-ar":
        for b in range(B):
            pmask_tail = torch.ones(L1, dtype=torch.bool, device=input_ids.device)
            pmask_b = torch.cat([
                torch.zeros(L0, dtype=torch.bool, device=input_ids.device),
                pmask_tail
            ], dim=0)  # [L0+L1]==[L]

            noise_tail = input_ids[b, L0:].clone()
            input_labels = input_ids[b, L0:]
            noise_tail.masked_fill_(pmask_tail, mask_id)
            
            pos_left = segment_positions_from_segids(segment_ids[b].reshape(1,-1))       # [1, L]
    
            pos_tail = pos_left[:,L0:]        # [1, L1]
            
            #sample_order_anchor_lookahead_fast,
            shuffled_indices, shuffled_pos_tail = shuffle_within_segments(segment_ids[b][L0:].reshape(1,-1), pos_tail, K = K, beta = 0.1)
            shuffled_input_labels = torch.gather(input_labels, 0, shuffled_indices.squeeze(0))
            extended_b = torch.cat([input_ids[b][:L0], shuffled_input_labels, noise_tail], dim=0)  # [2L]

            extended_pos = torch.cat([pos_left[:, :L0], shuffled_pos_tail, shuffled_pos_tail], dim=1)  # [1, 2L]


            extended_input_ids_list.append(extended_b)
            pmask_list.append(pmask_b)
            seg_kept_list.append(segment_ids[b])  # 记录下这条样本的段号
            order_list.append(extended_pos.squeeze(0))
    else:
        raise ValueError(f"Unknown training.method: {config.training.method}")

    # 堆叠
    extended_input_ids = torch.stack(extended_input_ids_list, dim=0)  # [B_eff, 2L]
    p_mask             = torch.stack(pmask_list, dim=0).to(torch.bool)  # [B_eff, L]
    seg_left           = torch.stack(seg_kept_list, dim=0)  # [B_eff, L]（左半段的段号）
    tok_idx_ext      = torch.stack(order_list, dim=0)        # [B_eff, L1]

    labels = extended_input_ids[:, :L].clone()

    #tok_idx_ext = torch.cat([pos_left, pos_tail], dim=1)              # [B_eff, 2L]

    # （可选）如需对 pad 做处理：如果你的打包没有 pad 可忽略
    if pad_id is not None:
        # 若左半段某 token 是 pad，则其位置可设为 0（或 -1，视模型实现）
        pad_mask_left = (extended_input_ids[:, :L] == pad_id)
        tok_idx_ext[:, :L][pad_mask_left] = 0
        # 右半段对应 tail 部分如果有 pad（通常没有），同样置 0
        pad_mask_tail = (extended_input_ids[:, L:] == pad_id)
        tok_idx_ext[:, L:][pad_mask_tail] = 0

    # —— 过滤掉没有监督位置的样本（与你原逻辑一致）——
    keep = p_mask.view(p_mask.size(0), -1).any(dim=1)                 # [B_eff]
    extended_input_ids = extended_input_ids[keep]
    p_mask            = p_mask[keep]
    tok_idx_ext       = tok_idx_ext[keep]
    labels            = labels[keep]
    seg_left          = seg_left[keep]                                 # 方便外面若还需要

    # （可选）把扩展段的 seg 也返回，便于在外面构造跨段屏蔽 same_seg
    extended_segment_ids = torch.cat([seg_left, seg_left], dim=1)      # [B_eff, 2L]

    return extended_input_ids, p_mask, tok_idx_ext, labels, keep, extended_segment_ids

# 方案 B：HF Dataset.from_generator（注意潜在物化）
from datasets import Dataset, Features, Sequence, Value




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
            name=config.experiment.name,
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
    raw_ds = hfds.load_dataset(config.dataset.optimization_data, split="train")
    # shuffle the dataset
    raw_ds = raw_ds.shuffle(seed=config.training.seed)
    # select 10% for checking
    raw_ds = raw_ds.select([i for i in list(range(int(len(raw_ds)*0.1)))])
    raw_ds_val = hfds.load_dataset(config.dataset.optimization_data, split="val")
    chunk_size = config.training.chunk_size
    packed_ds  = prepare_pretrain_packed_concat(
        ds=raw_ds,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        add_eos=True
    )
    packed_ds_val  = prepare_pretrain_packed_concat(
        ds=raw_ds_val,
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
    base_bias = make_basic_window_attention_additive(
        N=N,
        start_pos=L0,
        block_size=config.training.block_size,
        device=accelerator.device,
        dtype=next(model.parameters()).dtype
    )

    base_bias_eval = make_basic_window_attention_additive(
        N=N,
        start_pos=L0,
        block_size=config.training.block_size,
        device=accelerator.device,
        dtype=next(model.parameters()).dtype
    )

    # collate: 把一个 batch 的 input_ids 变成训练所需四元组
    need_step = (config.training.method == "trace")

    def collate_build_batch(batch, base_bias=base_bias, K=config.training.block_size):
        # batch: list of {"input_ids": [L], "segment_ids": [L]}
        input_ids = torch.tensor([ex["input_ids"] for ex in batch],
                                dtype=torch.long, device=accelerator.device)    # [B, L]
        seg_ids   = torch.tensor([ex["segment_ids"] for ex in batch],
                                dtype=torch.long, device=accelerator.device)    # [B, L]

        # step_map（trace 需要，semi-ar 也可用线性占位）
        if config.training.method == "trace":
            step_map_list = [list(range(input_ids.size(1))) for _ in range(input_ids.size(0))]
        else:
            step_map_list = [list(range(input_ids.size(1))) for _ in range(input_ids.size(0))]

        with torch.no_grad():
            extended_input_ids, p_mask, tok_idx_ext, labels, keep, extended_segment_ids = collect_training_data(
                input_ids=input_ids,
                step_map_list=step_map_list,
                start_pos=0,
                mask_id=mask_id,
                pad_id=pad_id,
                config=config,
                segment_ids=seg_ids,
                K=K,
            )

        # 因为 collect_training_data 可能过滤（keep），所以要同步筛选 seg_ids
        # keep 的逻辑是按 p_mask 的任意 True 行保留。我们重建一个 keep 掩码：
        # 这里等价：每条样本至少产生了一条 extended；如果你担心不一致，可返回 keep 从函数里带出。
        B_eff = extended_input_ids.size(0)
        # seg_ids 扩展到 [L | L]，右半段与左半段同段号
        # seg_ext = torch.cat([seg_ids[:B_eff], seg_ids[:B_eff]], dim=1)        # [B_eff, N]

        # 先复制你的 block 规则得到 base add-bias
        

        
        attn_bias = base_bias.repeat(B_eff, 1, 1, 1).to(base_bias.dtype)

        
        
        # 与分段可见性交集
        same_seg = extended_segment_ids.unsqueeze(-1).eq(extended_segment_ids.unsqueeze(-2)).unsqueeze(1)
        attn_bias = attn_bias.masked_fill(~same_seg, 0)
        # pad 列屏蔽 & 自环修复
        attn_bias = process_pad_additive(attn_bias, extended_input_ids, pad_id, start_pos=0)
        
        # attn_bias = base_bias.repeat(B_eff, 1, 1, 1).to(extended_input_ids.dtype)

        # # 叠加 segment 隔离：同段可见，不同段禁用
        # # same_seg: [B, N, N]
        # same_seg = seg_ext.unsqueeze(-1).eq(seg_ext.unsqueeze(-2))
        # same_seg = same_seg.unsqueeze(1)                                         # [B,1,N,N]
        # attn_bias = attn_bias.masked_fill(~same_seg, float("-inf"))

        # # 处理 pad 列
        # attn_bias = process_pad_additive(attn_bias, extended_input_ids, pad_id, start_pos=0)

        return {
            "extended_input_ids": extended_input_ids,   # [B_eff, N]
            "p_mask": p_mask,                           # [B_eff, L]
            "tok_idx_ext": tok_idx_ext,                 # [B_eff, N]
            "labels": labels,                           # [B_eff, L]
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
    
    eval_dataloader_lm = DataLoader(
        packed_ds_val,
        batch_size=config.training.batch_size_lm,
        shuffle=False,
        collate_fn=lambda x: collate_build_batch(x, base_bias=base_bias_eval, K=1),
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
        loss_lm = - (logp_tok * p_mask).sum(dim=1) / p_mask.sum(dim=1).clamp_min(1)
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
            model.train()
            # 所有张量已在 collate 放到正确 device
            loss_lm = forward_process(batch)

            accelerator.backward(loss_lm)

            accelerator.log(
                {
                    "train/mask_num": batch["p_mask"].sum().item(),
                    "train/loss_lm": loss_lm.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step_global": step_full,
                    "train/epoch": epoch + step / len(progress_bar),
                },
                step=step_full,
            )

            if (step_full + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            step_full += 1
            end = time.time()
            
            if step_full % config.training.save_steps == 0:
                print("eval and save checkpoint at step", step_full)
                model.eval()
                total_loss = 0.0
                total_tokens = 0
                with torch.no_grad():
                    for eval_batch in eval_dataloader_lm:
                        extended_input_ids = eval_batch["extended_input_ids"]
                        p_mask = eval_batch["p_mask"]
                        tok_idx_ext = eval_batch["tok_idx_ext"]
                        attn_bias = eval_batch["attn_bias"]
                        labels = eval_batch["labels"]

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
                        loss_lm = - (logp_tok * p_mask).sum(dim=1)  # [B_eff]
                        
                        total_loss += loss_lm.sum().item()
                        total_tokens += p_mask.sum().item()
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                accelerator.log(
                    {
                        "eval/loss_lm": avg_loss,
                        "eval/perplexity": perplexity,
                        "eval/step_global": step_full,
                    },
                    step=step_full,
                )
                
                accelerator.wait_for_everyone()
                save_checkpoint(model, tokenizer, config, accelerator, f"step-{step_full}")
                

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
