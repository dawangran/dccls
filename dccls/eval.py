#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate frozen HF base encoder + MIL head on a NEW labeled dataset.

Input (plain jsonl or jsonl.gz):
  {"id": "...", "txt": "ACGT...", "gene_id": "ENSG..."}

Artifacts in ckpt_dir:
  - best.pt              (contains head_state_dict)
  - gene2id.json
  - id2gene.json

Behavior:
- Closed-set evaluation: only keep samples whose gene_id exists in gene2id.json.
- Output CSV includes:
    id, true_gene_id, pred_gene_id_top1, pred_prob_top1, correct_top1,
    optional topk lists, optional attn_top1_pos.
- Optional JSONL output (more details).

Performance knobs:
- --amp
- TF32 enabled on CUDA
- DataLoader: num_workers, persistent_workers, prefetch_factor, pin_memory
"""

import os
import re
import json
import csv
import time
import gzip
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from multimolecule import RnaErnieModel, RnaTokenizer
from transformers import AutoModel, AutoTokenizer


# ----------------------------
# Utils
# ----------------------------

BWAV_RE = re.compile(r"<\|bwav:(\d+)\|>")

def rank0_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# ----------------------------
# Chunking (deterministic)
# ----------------------------

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
    Deterministic:
      - all starts = range(0, T-L+1, stride)
      - if n>=K, take K uniformly by linspace
      - else take all then pad
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
        sel = [starts[j] for j in idxs]
        for i, s in enumerate(sel):
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

def collate_fn_factory_no_split(chunk_len: int, stride: int, K: int, pad_id: int):
    """
    batch item: (rid, token_ids, y_int, true_gene_id)
    returns:
      chunks: (B,K,L) int64
      masks : (B,K) bool
      ys    : (B,) int64
      rids  : list[str]
      true_gene_ids: list[str]
    """
    def _collate(batch):
        B = len(batch)
        chunks = np.zeros((B, K, chunk_len), dtype=np.int64)
        masks = np.zeros((B, K), dtype=np.bool_)
        ys = np.zeros((B,), dtype=np.int64)
        rids = []
        true_genes = []
        for i, (rid, token_ids, y, true_gene) in enumerate(batch):
            c, m = make_chunks_deterministic(token_ids, chunk_len, stride, K, pad_id)
            chunks[i] = c
            masks[i] = m
            ys[i] = y
            rids.append(rid)
            true_genes.append(true_gene)
        return (
            torch.from_numpy(chunks),
            torch.from_numpy(masks),
            torch.from_numpy(ys),
            rids,
            true_genes,
        )
    return _collate


# ----------------------------
# Dataset: NO split_map, closed-set filter
# ----------------------------

class JsonlIterableNoSplit(IterableDataset):
    """
    Stream jsonl(.gz). Only keep gene_id in gene2id.
    yields: (rid, token_ids, y_int, true_gene_id)
    """
    def __init__(
        self,
        paths: List[str],
        gene2id: Dict[str, int],
        vocab_size: Optional[int],
        text_field: str,
        tokenizer=None,
        add_special_tokens: bool = False,
        max_token_len: int = 200000,
        debug_every: int = 0,
        max_reads: int = 0,
    ):
        super().__init__()
        self.paths = paths
        self.gene2id = gene2id
        self.vocab_size = vocab_size
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_token_len = max_token_len
        self.debug_every = int(debug_every)
        self.max_reads = int(max_reads)

    def __iter__(self):
        seen = 0
        kept = 0
        for p in self.paths:
            with _open_text(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    seen += 1
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
                        if self.debug_every and (seen % self.debug_every == 0):
                            rank0_print(f"[DEBUG][dataset] seen={seen} kept={kept} keep_rate={kept/max(seen,1):.4f}")
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

                    kept += 1
                    if self.debug_every and (seen % self.debug_every == 0):
                        rank0_print(f"[DEBUG][dataset] seen={seen} kept={kept} keep_rate={kept/max(seen,1):.4f}")

                    yield rid, token_ids, int(self.gene2id[gene]), gene

                    if self.max_reads > 0 and kept >= self.max_reads:
                        return


# ----------------------------
# Models
# ----------------------------

class HFChunkEncoder(nn.Module):
    """
    AutoModel as encoder:
      input_ids (N,L) -> hidden_states[layer] (N,L,H) -> masked mean pool -> (N,H)
      -> proj -> (N,out_dim)
    """
    def __init__(
        self,
        model_path: str,
        vocab_size: int,
        pad_id: int,
        out_dim: int = 256,
        hidden_layer: int = -1,
        model_type: str = "auto",
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.hidden_layer = int(hidden_layer)

        if model_type == "rnaernie":
            self.model = RnaErnieModel.from_pretrained(model_path)
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        emb = self.model.get_input_embeddings()
        if emb is not None and emb.num_embeddings < vocab_size:
            self.model.resize_token_embeddings(vocab_size)

        cfg = self.model.config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "d_model", None)
            or out_dim
        )
        hidden_size = int(hidden_size)
        self.proj = nn.Linear(hidden_size, out_dim, bias=False) if hidden_size != out_dim else nn.Identity()
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attn = (input_ids != self.pad_id).to(torch.long)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states
        try:
            hs = hidden_states[self.hidden_layer]
        except IndexError:
            raise ValueError(
                f"hidden_layer={self.hidden_layer} out of range; "
                f"num_hidden_states={len(hidden_states)}"
            )
        mask = attn.unsqueeze(-1).to(hs.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hs * mask).sum(dim=1) / denom
        z = self.ln(self.proj(pooled))
        return z

class ReadClassifierAttn(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.gate = nn.Linear(dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, ems: torch.Tensor, chunk_mask: torch.Tensor, return_attn: bool = False):
        chunk_mask = chunk_mask.bool()
        x = self.pre(ems)
        scores = self.gate(x).squeeze(-1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~chunk_mask, neg_inf)
        attn = F.softmax(scores, dim=1)
        read_emb = torch.einsum("bk,bkd->bd", attn, x)
        logits = self.head(read_emb)
        return (logits, attn) if return_attn else logits

class ReadClassifierGatedAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_attn: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = float(temperature)

        self.pre = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.V = nn.Linear(dim, hidden_attn, bias=True)
        self.U = nn.Linear(dim, hidden_attn, bias=True)
        self.w = nn.Linear(hidden_attn, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, ems: torch.Tensor, chunk_mask: torch.Tensor, return_attn: bool = False):
        chunk_mask = chunk_mask.bool()
        x = self.pre(ems)

        Vh = torch.tanh(self.V(x))
        Uh = torch.sigmoid(self.U(x))
        gh = Vh * Uh
        scores = self.w(gh).squeeze(-1)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~chunk_mask, neg_inf)

        tau = max(self.temperature, 1e-4)
        attn = F.softmax(scores / tau, dim=1)

        # attention dropout + renorm
        attn = self.attn_drop(attn)
        attn = attn * chunk_mask.to(attn.dtype)
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom

        read_emb = torch.einsum("bk,bkd->bd", attn, x)
        logits = self.head(read_emb)
        return (logits, attn) if return_attn else logits


# ----------------------------
# Eval Loop
# ----------------------------

@torch.no_grad()
def run_eval(
    base: HFChunkEncoder,
    head: nn.Module,
    loader: DataLoader,
    device: str,
    amp: bool,
    topk: int,
    save_attn: bool,
    out_jsonl: Optional[str],
    out_csv: Optional[str],
    id2gene: Dict[str, str],
    log_every_steps: int = 10,
):
    base.eval()
    head.eval()

    jsonl_f = None
    csv_f = None
    csv_writer = None

    if out_jsonl:
        ensure_dir(os.path.dirname(out_jsonl) or ".")
        jsonl_f = open(out_jsonl, "w", encoding="utf-8")

    if out_csv:
        ensure_dir(os.path.dirname(out_csv) or ".")
        csv_f = open(out_csv, "w", encoding="utf-8", newline="")
        fields = ["id", "true_gene_id", "pred_gene_id_top1", "pred_prob_top1", "correct_top1"]
        if topk > 1:
            fields += ["pred_gene_id_topk", "pred_prob_topk"]
        if save_attn:
            fields += ["attn_top1_pos"]
        csv_writer = csv.DictWriter(csv_f, fieldnames=fields)
        csv_writer.writeheader()

    n_total, n_correct = 0, 0

    t_start = time.time()
    t_last = time.time()
    seen = 0

    rank0_print("[DEBUG] enter dataloader loop...")

    for step, (chunks, chunk_mask, y, rids, true_gene_ids) in enumerate(loader, start=1):
        if step == 1:
            rank0_print("[DEBUG] got first batch from dataloader")
            rank0_print(f"[DEBUG] chunks.device(before to)={chunks.device}")

        chunks = chunks.to(device, non_blocking=True)
        chunk_mask = chunk_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if step == 1:
            rank0_print(f"[DEBUG] chunks.is_cuda(after to)={chunks.is_cuda}")
            rank0_print(f"[DEBUG] base_device={next(base.parameters()).device} head_device={next(head.parameters()).device}")

        B, K, L = chunks.shape
        flat = chunks.view(B * K, L)

        if amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                em = base(flat)
                ems = em.view(B, K, -1)
                if save_attn:
                    logits, attn = head(ems, chunk_mask, return_attn=True)
                else:
                    logits = head(ems, chunk_mask)
        else:
            em = base(flat)
            ems = em.view(B, K, -1)
            if save_attn:
                logits, attn = head(ems, chunk_mask, return_attn=True)
            else:
                logits = head(ems, chunk_mask)

        probs = F.softmax(logits, dim=1)
        tk = min(int(topk), probs.size(1))
        pvals, pidx = torch.topk(probs, k=tk, dim=1)

        pvals_np = pvals.detach().cpu().numpy()
        pidx_np = pidx.detach().cpu().numpy()

        if save_attn:
            attn_np = attn.detach().cpu().numpy()
            attn_top1_pos = attn_np.argmax(axis=1).tolist()
        else:
            attn_np = None
            attn_top1_pos = None

        for i in range(B):
            rid = rids[i]
            true_gene = true_gene_ids[i]

            topk_class = pidx_np[i].tolist()
            topk_prob = [float(x) for x in pvals_np[i]]
            topk_gene = [id2gene.get(str(cid), "NA") for cid in topk_class]

            pred_gene_top1 = topk_gene[0]
            pred_prob_top1 = topk_prob[0]
            correct = int(pred_gene_top1 == true_gene)

            n_total += 1
            n_correct += correct
            seen += 1

            if jsonl_f is not None:
                rec = {
                    "id": rid,
                    "true_gene_id": true_gene,
                    "topk_class_id": topk_class,
                    "topk_gene_id": topk_gene,
                    "topk_prob": topk_prob,
                    "correct_top1": correct,
                }
                if save_attn:
                    rec["attn_top1_pos"] = int(attn_top1_pos[i])
                    rec["attn"] = [float(x) for x in attn_np[i]]
                jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if csv_writer is not None:
                row = {
                    "id": rid,
                    "true_gene_id": true_gene,
                    "pred_gene_id_top1": pred_gene_top1,
                    "pred_prob_top1": pred_prob_top1,
                    "correct_top1": correct,
                }
                if tk > 1:
                    row["pred_gene_id_topk"] = "|".join(topk_gene)
                    row["pred_prob_topk"] = "|".join([f"{p:.6g}" for p in topk_prob])
                if save_attn:
                    row["attn_top1_pos"] = int(attn_top1_pos[i])
                csv_writer.writerow(row)

        if log_every_steps > 0 and step % log_every_steps == 0:
            now = time.time()
            per_step = (now - t_last) / log_every_steps
            thr = seen / max(now - t_start, 1e-9)
            rank0_print(f"[DEBUG] step={step} seen={seen} step_time={per_step:.3f}s avg_throughput={thr:.2f} reads/s")
            t_last = now

    if jsonl_f is not None:
        jsonl_f.close()
    if csv_f is not None:
        csv_f.close()

    acc = (n_correct / n_total) if n_total > 0 else 0.0
    rank0_print(f"[EVAL] n_known={n_total} top1_acc={acc:.6f}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser("Gene classification eval (no split_map)")

    ap.add_argument("--data", type=str, nargs="+", required=True, help="with_gene_all.jsonl or .jsonl.gz path(s)")
    ap.add_argument("--model_path", type=str, required=True, help="HF base model path (same as training)")

    ap.add_argument("--ckpt_dir", type=str, required=True, help="contains best.pt/gene2id.json/id2gene.json")
    ap.add_argument("--ckpt_name", type=str, default="best.pt")
    ap.add_argument("--gene2id_name", type=str, default="gene2id.json")
    ap.add_argument("--id2gene_name", type=str, default="id2gene.json")

    ap.add_argument("--outdir", type=str, required=True, help="output dir ONLY for predictions")

    ap.add_argument("--head_type", type=str, default="gated", choices=["single", "gated"])
    ap.add_argument("--gated_hidden", type=int, default=128)
    ap.add_argument("--gated_attn_dropout", type=float, default=0.1)
    ap.add_argument("--gated_temperature", type=float, default=1.0)

    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--pad_id", type=int, default=None)
    ap.add_argument("--hidden_layer", type=int, default=-1)
    ap.add_argument("--text_field", type=str, default="txt")
    ap.add_argument("--model_type", type=str, default="auto", choices=["auto", "rnaernie"])
    ap.add_argument("--add_special_tokens", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--chunk_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--K_chunks", type=int, default=64)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--save_attn", action="store_true")

    ap.add_argument("--export_csv", action="store_true")
    ap.add_argument("--csv_name", type=str, default="eval.csv")
    ap.add_argument("--jsonl_name", type=str, default="eval.jsonl")

    ap.add_argument("--max_token_len", type=int, default=200000)
    ap.add_argument("--dataset_debug_every", type=int, default=0)
    ap.add_argument("--max_reads", type=int, default=0, help="quick test: limit number of kept reads (0 = no limit)")

    ap.add_argument("--log_every_steps", type=int, default=10)

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    # CUDA perf knobs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    gene2id_path = os.path.join(args.ckpt_dir, args.gene2id_name)
    id2gene_path = os.path.join(args.ckpt_dir, args.id2gene_name)

    for p, name in [(ckpt_path, "best.pt"), (gene2id_path, "gene2id.json"), (id2gene_path, "id2gene.json")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    with open(gene2id_path, "r", encoding="utf-8") as f:
        gene2id = json.load(f)
    with open(id2gene_path, "r", encoding="utf-8") as f:
        id2gene = json.load(f)

    num_classes = len(gene2id)
    rank0_print(f"[INFO] device={device} num_classes={num_classes} head_type={args.head_type}")
    rank0_print(f"[INFO] ckpt_dir={args.ckpt_dir} outdir={args.outdir}")
    rank0_print(f"[INFO] data={args.data}")
    rank0_print(f"[INFO] K={args.K_chunks} L={args.chunk_len} stride={args.stride} batch_size={args.batch_size} workers={args.num_workers}")
    rank0_print(f"[INFO] amp={args.amp} pin_memory={args.pin_memory} prefetch_factor={args.prefetch_factor} persistent_workers={args.persistent_workers}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    head_state = ckpt.get("head_state_dict", ckpt)

    if args.model_type == "rnaernie":
        tokenizer = RnaTokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    tokenizer.model_max_length = int(1e9)

    add_special_tokens = args.add_special_tokens
    if add_special_tokens is None:
        add_special_tokens = args.model_type == "rnaernie"

    vocab_size = args.vocab_size if args.vocab_size is not None else tokenizer.vocab_size
    pad_id = args.pad_id if args.pad_id is not None else tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("pad_id is required; pass --pad_id or use a tokenizer with pad_token_id.")

    base = HFChunkEncoder(
        model_path=args.model_path,
        vocab_size=vocab_size,
        pad_id=pad_id,
        out_dim=256,
        hidden_layer=args.hidden_layer,
        model_type=args.model_type,
    ).to(device)
    for p in base.parameters():
        p.requires_grad = False
    base.eval()

    if args.head_type == "single":
        head = ReadClassifierAttn(dim=256, num_classes=num_classes, dropout=0.0).to(device)
    else:
        head = ReadClassifierGatedAttn(
            dim=256,
            num_classes=num_classes,
            hidden_attn=args.gated_hidden,
            dropout=0.0,
            attn_dropout=args.gated_attn_dropout,
            temperature=args.gated_temperature,
        ).to(device)

    head.load_state_dict(head_state, strict=True)
    head.eval()

    ds = JsonlIterableNoSplit(
        paths=args.data,
        gene2id=gene2id,
        vocab_size=vocab_size,
        text_field=args.text_field,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
        max_token_len=args.max_token_len,
        debug_every=args.dataset_debug_every,
        max_reads=args.max_reads,
    )
    collate_fn = collate_fn_factory_no_split(args.chunk_len, args.stride, args.K_chunks, pad_id)

    # DataLoader kwargs (prefetch/persistent only valid when num_workers>0)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    loader = DataLoader(ds, **loader_kwargs)

    out_csv = os.path.join(args.outdir, args.csv_name) if args.export_csv else None
    out_jsonl = os.path.join(args.outdir, args.jsonl_name)

    run_eval(
        base=base,
        head=head,
        loader=loader,
        device=device,
        amp=args.amp,
        topk=args.topk,
        save_attn=args.save_attn,
        out_jsonl=out_jsonl,
        out_csv=out_csv,
        id2gene=id2gene,
        log_every_steps=args.log_every_steps,
    )

    rank0_print(f"[DONE] wrote jsonl: {out_jsonl}")
    if out_csv:
        rank0_print(f"[DONE] wrote csv : {out_csv}")


if __name__ == "__main__":
    main()
