# -*- coding: utf-8 -*-
import os, time, json
from glob import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .utils import ensure_dir, save_json, rank0_print, set_seed, now_str, open_text
from .data import (
    build_class2id_with_support,
    build_split_map_class_7_2_1,
    JsonlIterable,
    collate_fn_factory,
)
from .model import HFChunkEncoder, ReadClassifierAttn, ReadClassifierGatedAttn
from .train import (
    train_one_epoch_frozen_base,
    eval_one_epoch,
    save_confusion_artifacts,
    save_attention_artifacts,
)


def attn_stats_to_wandb(attn_stats: dict, prefix: str = "attn/") -> dict:
    if not attn_stats:
        return {}
    if ("n_reads" not in attn_stats) or ("sum_topm_mass" not in attn_stats):
        return {}

    n_reads = float(attn_stats["n_reads"].item())
    if n_reads <= 0:
        return {}

    out = {}
    if "sum_entropy" in attn_stats:
        out[f"{prefix}mean_entropy"] = float(attn_stats["sum_entropy"].item()) / max(n_reads, 1.0)

    topm_list = attn_stats.get("topm_list", [])
    sum_topm_mass = attn_stats["sum_topm_mass"].detach().float().cpu().numpy()
    mean_topm_mass = (sum_topm_mass / max(n_reads, 1.0)).tolist()
    for m, v in zip(topm_list, mean_topm_mass):
        out[f"{prefix}top{int(m)}_mass_mean"] = float(v)
    return out


def compute_class_weight(data_paths, class2id, split_map, num_classes, device, path_class_map):
    freq = np.zeros((num_classes,), dtype=np.int64)
    for p in data_paths:
        cls = path_class_map.get(os.path.abspath(p))
        if cls is None:
            continue
        with open_text(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = obj.get("id", None)
                if cls in class2id and rid in split_map and split_map[rid] == "train":
                    freq[class2id[cls]] += 1
    w = 1.0 / np.sqrt(np.maximum(freq, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def discover_data_from_class_dirs(data_root: str):
    """Discover jsonl/jsonl.gz files from subfolders and use subfolder names as labels."""
    data_paths = []
    path_class_map = {}
    for entry in sorted(os.scandir(data_root), key=lambda x: x.name):
        if not entry.is_dir():
            continue
        class_name = entry.name
        files = []
        for pat in (os.path.join(entry.path, "*.jsonl"), os.path.join(entry.path, "*.jsonl.gz")):
            files.extend(glob(pat))
        for fp in sorted(set(files)):
            abs_fp = os.path.abspath(fp)
            data_paths.append(abs_fp)
            path_class_map[abs_fp] = class_name
    return data_paths, path_class_map


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True, help="Root dir whose subfolders are class labels")
    ap.add_argument("--model_path", type=str, required=True, help="HF base model path")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--reads_per_class", type=int, default=100, help="How many reads to sample per class")
    ap.add_argument("--split_salt", type=str, default="0")

    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--pad_id", type=int, default=None)
    ap.add_argument("--hidden_layer", type=int, default=-1,
                    help="Which backbone hidden layer to use for pooling (-1 last, -2 second last, ...).")
    ap.add_argument("--text_field", type=str, default="text")
    ap.add_argument("--min_text_length", type=int, default=1000, help="Minimum tokenized text length required to keep a sample")
    ap.add_argument("--add_special_tokens", action=argparse.BooleanOptionalAction, default=None)

    ap.add_argument("--chunk_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--K_chunks", type=int, default=64)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup ratio over total training steps")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_class_weight", action="store_true")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--head_type", type=str, default="single", choices=["single", "gated"])
    ap.add_argument("--gated_hidden", type=int, default=128)
    ap.add_argument("--gated_attn_dropout", type=float, default=0.1)
    ap.add_argument("--gated_temperature", type=float, default=1.0)

    ap.add_argument("--save_attn", action="store_true")
    ap.add_argument("--attn_max_samples", type=int, default=64)

    ap.add_argument("--write_split_map_only", action="store_true")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="nanopore-gene-class")
    ap.add_argument("--wandb_name", type=str, default="")
    ap.add_argument("--wandb_tags", type=str, default="")
    ap.add_argument("--wandb_offline", action="store_true")

    args = ap.parse_args()
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")

    data_paths, path_class_map = discover_data_from_class_dirs(args.data_root)
    if len(data_paths) == 0:
        raise RuntimeError(f"No jsonl/jsonl.gz files found under data_root={args.data_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    ensure_dir(args.outdir)
    rank0_print(f"[INFO] device={device} outdir={args.outdir}")
    rank0_print(f"[INFO] reads_per_class={args.reads_per_class} split_salt={args.split_salt} head_type={args.head_type}")
    rank0_print(f"[INFO] K={args.K_chunks} L={args.chunk_len} stride={args.stride} batch_size(reads)={args.batch_size}")

    class2id, sel_counts = build_class2id_with_support(
        data_paths,
        args.reads_per_class,
        path_class_map=path_class_map,
        text_field=args.text_field,
        min_text_length=args.min_text_length,
    )
    if len(class2id) == 0:
        raise RuntimeError("No valid classes found in data_root subfolders.")
    args.num_classes = len(class2id)

    id2class = {str(v): k for k, v in class2id.items()}
    save_json(os.path.join(args.outdir, "class2id.json"), class2id)
    save_json(os.path.join(args.outdir, "id2class.json"), id2class)
    save_json(os.path.join(args.outdir, "selected_class_counts.json"), sel_counts)
    rank0_print(f"[INFO] discovered classes={args.num_classes}")

    split_map, sampled = build_split_map_class_7_2_1(
        data_paths,
        class2id,
        args.reads_per_class,
        args.split_salt,
        path_class_map=path_class_map,
        pct_train=70,
        pct_val=20,
        pct_test=10,
        text_field=args.text_field,
        min_text_length=args.min_text_length,
    )
    save_json(os.path.join(args.outdir, f"split_map_7_2_1_reads{args.reads_per_class}_salt{args.split_salt}.json"), split_map)
    save_json(os.path.join(args.outdir, f"sampled_rids_by_class_reads{args.reads_per_class}_salt{args.split_salt}.json"), sampled)

    if args.write_split_map_only:
        rank0_print("[INFO] --write_split_map_only set; exiting.")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    tokenizer.model_max_length = int(1e9)

    add_special_tokens = args.add_special_tokens
    if add_special_tokens is None:
        add_special_tokens = False

    vocab_size = args.vocab_size if args.vocab_size is not None else tokenizer.vocab_size
    pad_id = args.pad_id if args.pad_id is not None else tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("pad_id is required; pass --pad_id or use a tokenizer with pad_token_id.")

    train_ds = JsonlIterable(
        data_paths,
        class2id,
        split="train",
        split_map=split_map,
        vocab_size=vocab_size,
        path_class_map=path_class_map,
        text_field=args.text_field,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
        min_text_length=args.min_text_length,
    )
    val_ds = JsonlIterable(
        data_paths,
        class2id,
        split="val",
        split_map=split_map,
        vocab_size=vocab_size,
        path_class_map=path_class_map,
        text_field=args.text_field,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
        min_text_length=args.min_text_length,
    )
    test_ds = JsonlIterable(
        data_paths,
        class2id,
        split="test",
        split_map=split_map,
        vocab_size=vocab_size,
        path_class_map=path_class_map,
        text_field=args.text_field,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
        min_text_length=args.min_text_length,
    )

    collate_fn = collate_fn_factory(args.chunk_len, args.stride, args.K_chunks, pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=collate_fn, drop_last=False)

    base = HFChunkEncoder(
        model_path=args.model_path,
        vocab_size=vocab_size,
        pad_id=pad_id,
        out_dim=256,
        hidden_layer=args.hidden_layer,
    ).to(device)

    for p in base.parameters():
        p.requires_grad = False
    base.eval()

    if args.head_type == "single":
        head = ReadClassifierAttn(dim=256, num_classes=args.num_classes, dropout=0.1).to(device)
    else:
        head = ReadClassifierGatedAttn(
            dim=256,
            num_classes=args.num_classes,
            hidden_attn=args.gated_hidden,
            dropout=0.1,
            attn_dropout=args.gated_attn_dropout,
            temperature=args.gated_temperature,
        ).to(device)

    class_weight = None
    if args.use_class_weight:
        class_weight = compute_class_weight(
            data_paths, class2id, split_map, args.num_classes, device, path_class_map=path_class_map
        )

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_train_samples = sum(1 for sp in split_map.values() if sp == "train")
    train_steps_per_epoch = max(1, (num_train_samples + max(args.batch_size, 1) - 1) // max(args.batch_size, 1))
    total_train_steps = max(1, args.epochs * train_steps_per_epoch)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    wb = None
    if args.wandb:
        import wandb
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        run_name = args.wandb_name or f"class_{args.head_type}_{os.path.basename(args.model_path)}_{now_str()}_salt{args.split_salt}"
        tags = [t for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else []
        wb = wandb.init(project=args.wandb_project, name=run_name, tags=tags, config=vars(args))

    best_top1 = -1.0
    best_path = os.path.join(args.outdir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_top1, tr_top5 = train_one_epoch_frozen_base(
            base=base,
            head=head,
            loader=train_loader,
            opt=opt,
            device=device,
            amp=args.amp,
            label_smoothing=args.label_smoothing,
            class_weight=class_weight,
            scheduler=scheduler,
        )

        val_loss, val_top1, val_top5, cm, attn_stats = eval_one_epoch(
            base=base,
            head=head,
            loader=val_loader,
            device=device,
            amp=args.amp,
            num_classes=args.num_classes,
            collect_attn=args.save_attn,
            attn_max_samples=args.attn_max_samples,
        )

        dt = time.time() - t0
        rank0_print(
            f"[E{epoch:03d}][freeze_base_{args.head_type}] "
            f"train loss={tr_loss:.4f} top1={tr_top1:.4f} top5={tr_top5:.4f} | "
            f"val loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f} | time={dt:.1f}s"
        )

        if wb is not None:
            log_obj = {
                "epoch": epoch,
                "train/loss": tr_loss, "train/top1": tr_top1, "train/top5": tr_top5,
                "val/loss": val_loss, "val/top1": val_top1, "val/top5": val_top5,
                "time/epoch_sec": dt,
            }
            if args.save_attn:
                log_obj.update(attn_stats_to_wandb(attn_stats, prefix="attn/"))
            wb.log(log_obj)

        save_confusion_artifacts(args.outdir, cm, id2class, prefix=f"val_e{epoch:03d}")
        if args.save_attn:
            save_attention_artifacts(args.outdir, attn_stats, prefix=f"val_e{epoch:03d}")

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(
                {"epoch": epoch, "best_val_top1": best_top1, "args": vars(args), "head_state_dict": head.state_dict()},
                best_path,
            )
            rank0_print(f"[SAVE] {best_path}  val_top1={best_top1:.4f}")

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        head.load_state_dict(ckpt["head_state_dict"], strict=True)

    test_loss, test_top1, test_top5, cm_test, attn_test = eval_one_epoch(
        base, head, test_loader, device, args.amp,
        num_classes=args.num_classes,
        collect_attn=args.save_attn,
        attn_max_samples=args.attn_max_samples,
    )
    rank0_print(f"[TEST] loss={test_loss:.4f} top1={test_top1:.4f} top5={test_top5:.4f}")

    save_confusion_artifacts(args.outdir, cm_test, id2class, prefix="test_best")
    if args.save_attn:
        save_attention_artifacts(args.outdir, attn_test, prefix="test_best")

    if wb is not None:
        test_log = {"test/loss": test_loss, "test/top1": test_top1, "test/top5": test_top5}
        if args.save_attn:
            test_log.update(attn_stats_to_wandb(attn_test, prefix="test_attn/"))
        wb.log(test_log)
        wb.finish()


if __name__ == "__main__":
    main()
