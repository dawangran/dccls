# -*- coding: utf-8 -*-
import json, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .utils import open_text, stable_hash_u32

BWAV_RE = re.compile(r"<\|bwav:(\d+)\|>")


def parse_token_ids(text: str, vocab_size: Optional[int]) -> List[int]:
    ids = [int(m.group(1)) for m in BWAV_RE.finditer(text)]
    if vocab_size is not None:
        ids = [t for t in ids if 0 <= t < vocab_size]
    return ids


def encode_text(
    text: str,
    vocab_size: Optional[int],
    tokenizer=None,
    add_special_tokens: bool = False,
) -> List[int]:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=add_special_tokens).input_ids
    else:
        ids = parse_token_ids(text, vocab_size)
    if vocab_size is not None:
        ids = [t for t in ids if 0 <= t < vocab_size]
    return ids


def build_gene2id_with_support(paths: List[str], num_classes: int, reads_per_gene: int):
    """
    Count gene frequency, keep genes with >= reads_per_gene, pick top num_classes among them.
    Return:
      gene2id: {gene -> class_id}
      selected_counts: {gene -> count}
    """
    counts = {}
    for p in paths:
        with open_text(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                g = obj.get("gene_id", None)
                if g is None:
                    continue
                counts[g] = counts.get(g, 0) + 1

    eligible = [(g, c) for g, c in counts.items() if c >= reads_per_gene]
    eligible.sort(key=lambda x: x[1], reverse=True)
    eligible = eligible[:num_classes]

    gene2id = {g: i for i, (g, _) in enumerate(eligible)}
    selected_counts = {g: int(c) for g, c in eligible}
    return gene2id, selected_counts


def build_split_map_gene_7_2_1(
    paths: List[str],
    gene2id: Dict[str, int],
    reads_per_gene: int,
    split_salt: str,
    pct_train: int = 70,
    pct_val: int = 20,
    pct_test: int = 10,
):
    """
    Deterministic per-gene subsampling (exact reads_per_gene) + deterministic 7:2:1 split.

    Return:
      split_map: {rid: "train"/"val"/"test"}
      sampled_rids_by_gene: {gene: [rid,...]}  (length == reads_per_gene if possible)
    """
    gene_to_rids: Dict[str, List[str]] = {g: [] for g in gene2id.keys()}

    for p in paths:
        with open_text(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = obj.get("id", None)
                gene = obj.get("gene_id", None)
                if rid is None or gene is None:
                    continue
                if gene not in gene_to_rids:
                    continue
                gene_to_rids[gene].append(rid)

    split_map: Dict[str, str] = {}
    sampled: Dict[str, List[str]] = {}

    for gene, rids in gene_to_rids.items():
        if len(rids) == 0:
            continue

        keyed = [(stable_hash_u32(f"{rid}|{split_salt}"), rid) for rid in rids]
        keyed.sort(key=lambda x: x[0])

        # subsample exactly reads_per_gene if possible
        take = min(reads_per_gene, len(keyed))
        sel = [rid for _, rid in keyed[:take]]
        sampled[gene] = sel

        n = len(sel)
        n_tr = int(round(n * (pct_train / 100.0)))
        n_va = int(round(n * (pct_val / 100.0)))
        n_tr = max(1, n_tr) if n >= 3 else max(0, min(n_tr, n))
        n_va = max(1, n_va) if n >= 3 else max(0, min(n_va, n))
        if n_tr + n_va > n:
            n_va = max(0, n - n_tr)
        n_te = max(0, n - n_tr - n_va)

        tr = sel[:n_tr]
        va = sel[n_tr:n_tr + n_va]
        te = sel[n_tr + n_va:n_tr + n_va + n_te]

        for rid in tr:
            split_map[rid] = "train"
        for rid in va:
            split_map[rid] = "val"
        for rid in te:
            split_map[rid] = "test"

    return split_map, sampled


class JsonlIterable(IterableDataset):
    """
    Stream jsonl(.gz). Split is determined by split_map[read_id] in {"train","val","test"}.

    Each yielded item: (read_id, token_ids(list[int]), label_int)
    """
    def __init__(
        self,
        paths: List[str],
        gene2id: Dict[str, int],
        split: str,
        split_map: Dict[str, str],
        vocab_size: int,
        text_field: str = "text",
        tokenizer=None,
        add_special_tokens: bool = False,
        max_token_len: int = 200000,
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        self.paths = paths
        self.gene2id = gene2id
        self.split = split
        self.split_map = split_map
        self.vocab_size = vocab_size
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_token_len = max_token_len

    def _iter_one_file(self, path: str):
        with open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                rid = obj.get("id", None)
                gene = obj.get("gene_id", None)
                text = obj.get(self.text_field, "")

                if rid is None or gene is None:
                    continue
                if gene not in self.gene2id:
                    continue

                sp = self.split_map.get(rid, None)
                if sp != self.split:
                    continue

                token_ids = encode_text(
                    text,
                    self.vocab_size,
                    tokenizer=self.tokenizer,
                    add_special_tokens=self.add_special_tokens,
                )
                if len(token_ids) == 0:
                    continue
                if len(token_ids) > self.max_token_len:
                    token_ids = token_ids[: self.max_token_len]

                yield rid, token_ids, self.gene2id[gene]

    def __iter__(self):
        for p in self.paths:
            yield from self._iter_one_file(p)


def make_chunks_deterministic(
    token_ids: List[int],
    chunk_len: int,
    stride: int,
    K: int,
    pad_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      chunks: (K, L) int64
      mask  : (K,) bool
    Deterministic selection:
      - generate all start positions
      - if N >= K: select K indices uniformly (linspace)
      - else: pad chunks with PAD and set mask False
    """
    L = chunk_len
    T = len(token_ids)

    if T <= L:
        starts = [0]
    else:
        starts = list(range(0, T - L + 1, stride))
        if len(starts) == 0:
            starts = [0]

    n = len(starts)
    chunks = np.full((K, L), pad_id, dtype=np.int64)
    mask = np.zeros((K,), dtype=np.bool_)

    if n >= K:
        idxs = np.linspace(0, n - 1, K).astype(np.int64)
        sel_starts = [starts[j] for j in idxs]
        for i, s in enumerate(sel_starts):
            seg = token_ids[s: s + L]
            if len(seg) < L:
                seg = seg + [pad_id] * (L - len(seg))
            chunks[i] = np.asarray(seg, dtype=np.int64)
            mask[i] = True
    else:
        for i, s in enumerate(starts):
            seg = token_ids[s: s + L]
            if len(seg) < L:
                seg = seg + [pad_id] * (L - len(seg))
            chunks[i] = np.asarray(seg, dtype=np.int64)
            mask[i] = True

    return chunks, mask


def collate_fn_factory(chunk_len: int, stride: int, K: int, pad_id: int):
    def _collate(batch):
        B = len(batch)
        chunks = np.zeros((B, K, chunk_len), dtype=np.int64)
        masks = np.zeros((B, K), dtype=np.bool_)
        ys = np.zeros((B,), dtype=np.int64)
        rids = []
        for i, (rid, token_ids, y) in enumerate(batch):
            c, m = make_chunks_deterministic(token_ids, chunk_len, stride, K, pad_id)
            chunks[i] = c
            masks[i] = m
            ys[i] = y
            rids.append(rid)
        return (
            torch.from_numpy(chunks),        # (B,K,L) int64
            torch.from_numpy(masks),         # (B,K)   bool
            torch.from_numpy(ys),            # (B,)    int64
            rids,
        )
    return _collate


# ============================
# No-split evaluation dataset
# ============================

import gzip
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Try reuse existing token parser from your project
try:
    from .utils import parse_token_ids  # type: ignore
except Exception:
    BWAV_RE = re.compile(r"<\|bwav:(\d+)\|>")

    def parse_token_ids(text: str, vocab_size: int) -> List[int]:
        ids = [int(m.group(1)) for m in BWAV_RE.finditer(text)]
        if vocab_size is not None:
            ids = [t for t in ids if 0 <= t < vocab_size]
        return ids


def make_chunks_deterministic(
    token_ids: List[int],
    chunk_len: int,
    stride: int,
    K: int,
    pad_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same behavior as your training script:
      - generate all chunk start positions
      - if enough chunks: select K uniformly by linspace
      - else: use all chunks then pad
    Returns:
      chunks: (K, L) int64
      mask  : (K,) bool
    """
    L = int(chunk_len)
    T = len(token_ids)

    if T <= L:
        starts = [0]
    else:
        starts = list(range(0, T - L + 1, int(stride)))
        if len(starts) == 0:
            starts = [0]

    n = len(starts)
    chunks = np.full((K, L), pad_id, dtype=np.int64)
    mask = np.zeros((K,), dtype=np.bool_)

    if n >= K:
        idxs = np.linspace(0, n - 1, K).astype(np.int64)
        sel_starts = [starts[j] for j in idxs]
        for i, s in enumerate(sel_starts):
            seg = token_ids[s : s + L]
            if len(seg) < L:
                seg = seg + [pad_id] * (L - len(seg))
            chunks[i] = np.asarray(seg, dtype=np.int64)
            mask[i] = True
    else:
        for i, s in enumerate(starts):
            seg = token_ids[s : s + L]
            if len(seg) < L:
                seg = seg + [pad_id] * (L - len(seg))
            chunks[i] = np.asarray(seg, dtype=np.int64)
            mask[i] = True

    return chunks, mask
