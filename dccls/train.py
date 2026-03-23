# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from .metrics import topk_acc, confusion_matrix_np
from .utils import ensure_dir

# ---------- attention diagnostics ----------

def _default_topm_list(K: int) -> List[int]:
    base = [1, 2, 4, 8, 16, 32]
    out = [m for m in base if m <= K]
    if K not in out:
        out.append(K)
    return out


def update_attn_stats(
    stats: dict,
    attn: torch.Tensor,         # (B,K) float
    chunk_mask: torch.Tensor,   # (B,K) bool
    rids: List[str],
    topm_list: List[int],
    max_samples: int = 64,
):
    if stats.get("K", None) is None:
        stats["K"] = attn.size(1)
        K = stats["K"]
        stats["topm_list"] = list(topm_list)
        M = len(topm_list)

        stats["sum_attn_by_pos"] = torch.zeros((K,), device=attn.device, dtype=torch.float32)
        stats["cnt_by_pos"] = torch.zeros((K,), device=attn.device, dtype=torch.float32)
        stats["top1_pos_counts"] = torch.zeros((K,), device=attn.device, dtype=torch.float32)

        stats["sum_entropy"] = torch.zeros((), device=attn.device, dtype=torch.float32)
        stats["n_reads"] = torch.zeros((), device=attn.device, dtype=torch.float32)

        stats["sum_topm_mass"] = torch.zeros((M,), device=attn.device, dtype=torch.float32)

        stats["samples_rid"] = []
        stats["samples_attn"] = []
        stats["samples_top1"] = []

    K = stats["K"]
    m = chunk_mask.to(attn.dtype)  # (B,K)

    # sum attn per position (valid only)
    stats["sum_attn_by_pos"] += (attn * m).sum(dim=0).float()
    stats["cnt_by_pos"] += m.sum(dim=0).float()

    # top-1 position counts (valid only)
    attn_valid = attn.masked_fill(~chunk_mask, 0.0)
    top1 = attn_valid.argmax(dim=1)  # (B,)
    stats["top1_pos_counts"] += torch.bincount(top1, minlength=K).float()

    # entropy per read: -sum a log a
    eps = 1e-12
    a = attn_valid / attn_valid.sum(dim=1, keepdim=True).clamp(min=eps)
    ent = -(a * (a.clamp(min=eps).log())).sum(dim=1)  # (B,)
    stats["sum_entropy"] += ent.float().sum()
    stats["n_reads"] += float(attn.size(0))

    # top-m cumulative mass curve
    topk_vals, _ = torch.topk(attn_valid, k=K, dim=1, largest=True, sorted=True)  # (B,K)
    cum = torch.cumsum(topk_vals, dim=1)  # (B,K)
    for i_m, m_val in enumerate(stats["topm_list"]):
        stats["sum_topm_mass"][i_m] += cum[:, m_val - 1].float().sum()

    # store a few sample vectors
    if len(stats["samples_rid"]) < max_samples:
        room = max_samples - len(stats["samples_rid"])
        take = min(room, len(rids))
        if take > 0:
            attn_cpu = attn_valid[:take].detach().cpu().numpy()
            top1_cpu = top1[:take].detach().cpu().numpy().tolist()
            stats["samples_rid"].extend(rids[:take])
            stats["samples_attn"].extend([attn_cpu[i] for i in range(take)])
            stats["samples_top1"].extend(top1_cpu)


def save_attention_artifacts(outdir: str, attn_stats: dict, prefix: str):
    if not attn_stats or "sum_attn_by_pos" not in attn_stats:
        return
    ensure_dir(outdir)

    sum_attn = attn_stats["sum_attn_by_pos"].detach().cpu().numpy()
    cnt = attn_stats["cnt_by_pos"].detach().cpu().numpy()
    top1 = attn_stats["top1_pos_counts"].detach().cpu().numpy()

    avg_attn = sum_attn / np.maximum(cnt, 1e-9)

    sum_entropy = float(attn_stats["sum_entropy"].item())
    n_reads = float(attn_stats["n_reads"].item())
    mean_entropy = sum_entropy / max(n_reads, 1.0)

    topm_list = attn_stats.get("topm_list", [])
    sum_topm_mass = attn_stats["sum_topm_mass"].detach().cpu().numpy()
    mean_topm_mass = sum_topm_mass / max(n_reads, 1.0)

    npz_path = os.path.join(outdir, f"{prefix}_attn_stats.npz")
    np.savez_compressed(
        npz_path,
        avg_attn_by_pos=avg_attn.astype(np.float32),
        top1_pos_counts=top1.astype(np.float32),
        cnt_by_pos=cnt.astype(np.float32),
        mean_entropy=np.array([mean_entropy], dtype=np.float32),
        n_reads=np.array([int(n_reads)], dtype=np.int64),
        topm_list=np.array(topm_list, dtype=np.int64),
        mean_topm_mass=mean_topm_mass.astype(np.float32),
    )

    samp_path = os.path.join(outdir, f"{prefix}_attn_samples.jsonl")
    with open(samp_path, "w", encoding="utf-8") as f:
        for rid, a, t1 in zip(attn_stats["samples_rid"], attn_stats["samples_attn"], attn_stats["samples_top1"]):
            f.write(json.dumps({"id": rid, "top1_pos": int(t1), "attn": [float(x) for x in a]}, ensure_ascii=False) + "\n")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(topm_list, mean_topm_mass, marker="o")
        plt.xlabel("top-m")
        plt.ylabel("mean sum(top-m attn)")
        plt.title(f"Attention cumulative mass ({prefix})")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_attn_topm_mass.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(avg_attn)), avg_attn, marker="o")
        plt.xlabel("chunk position (0..K-1)")
        plt.ylabel("avg attention weight")
        plt.title(f"Avg attention by chunk position ({prefix})")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_attn_avg_by_pos.png"), dpi=200)
        plt.close()
    except Exception:
        pass

# ---------- confusion artifacts ----------

def save_confusion_artifacts(outdir: str, cm: np.ndarray, id2gene: Dict[str, str], prefix: str = "val"):
    ensure_dir(outdir)
    npy_path = os.path.join(outdir, f"{prefix}_confusion_matrix.npy")
    np.save(npy_path, cm)

    denom = cm.sum(axis=1).clip(min=1)
    acc = cm.diagonal() / denom
    csv_path = os.path.join(outdir, f"{prefix}_per_class_acc.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class_id,gene_id,acc,n_val\n")
        for cid in range(cm.shape[0]):
            gene = id2gene.get(str(cid), "NA")
            f.write(f"{cid},{gene},{acc[cid]:.6f},{int(denom[cid])}\n")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({prefix})")
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{prefix}_confusion_matrix.png")
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception:
        pass

# ---------- train / eval ----------

def train_one_epoch_frozen_base(
    base: nn.Module,
    head: nn.Module,
    loader,
    opt: torch.optim.Optimizer,
    device: str,
    amp: bool,
    label_smoothing: float,
    class_weight: Optional[torch.Tensor],
    scheduler = None,
):
    base.eval()
    head.train()

    scaler = torch.amp.GradScaler("cuda") if (amp and device.startswith("cuda")) else None

    total_loss = total_top1 = total_top5 = 0.0
    n = 0

    for chunks, chunk_mask, y, _ in loader:
        chunks = chunks.to(device, non_blocking=True)
        chunk_mask = chunk_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        B, K, L = chunks.shape
        flat = chunks.view(B * K, L)

        opt.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    em = base(flat)
                ems = em.view(B, K, -1)
                logits = head(ems, chunk_mask)
                loss = F.cross_entropy(
                    logits, y,
                    weight=class_weight,
                    label_smoothing=label_smoothing
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                em = base(flat)
            ems = em.view(B, K, -1)
            logits = head(ems, chunk_mask)
            loss = F.cross_entropy(
                logits, y,
                weight=class_weight,
                label_smoothing=label_smoothing
            )
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

        top1, top5 = topk_acc(logits, y, topk=(1, 5))
        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_top1 += top1 * bsz
        total_top5 += top5 * bsz
        n += bsz

    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_top1 / n, total_top5 / n


@torch.no_grad()
def eval_one_epoch(
    base: nn.Module,
    head: nn.Module,
    loader,
    device: str,
    amp: bool,
    num_classes: int,
    collect_attn: bool,
    attn_max_samples: int,
):
    base.eval()
    head.eval()

    total_loss = total_top1 = total_top5 = 0.0
    n = 0
    all_pred, all_true = [], []

    attn_stats = {}

    for chunks, chunk_mask, y, rids in loader:
        chunks = chunks.to(device, non_blocking=True)
        chunk_mask = chunk_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        B, K, L = chunks.shape
        flat = chunks.view(B * K, L)

        topm_list_eff = _default_topm_list(K)

        if amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                em = base(flat)
                ems = em.view(B, K, -1)
                if collect_attn:
                    logits, attn = head(ems, chunk_mask, return_attn=True)
                else:
                    logits = head(ems, chunk_mask)
                loss = F.cross_entropy(logits, y)
        else:
            em = base(flat)
            ems = em.view(B, K, -1)
            if collect_attn:
                logits, attn = head(ems, chunk_mask, return_attn=True)
            else:
                logits = head(ems, chunk_mask)
            loss = F.cross_entropy(logits, y)

        if collect_attn:
            update_attn_stats(attn_stats, attn, chunk_mask, rids, topm_list=topm_list_eff, max_samples=attn_max_samples)

        top1, top5 = topk_acc(logits, y, topk=(1, 5))
        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_top1 += top1 * bsz
        total_top5 += top5 * bsz
        n += bsz

        pred = logits.argmax(dim=1)
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    if n == 0:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        return 0.0, 0.0, 0.0, cm, attn_stats

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    cm = confusion_matrix_np(y_true, y_pred, num_classes)

    return total_loss / n, total_top1 / n, total_top5 / n, cm, attn_stats
