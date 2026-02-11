# -*- coding: utf-8 -*-
import os, json, time, gzip, hashlib, random
from typing import Any

import numpy as np
import torch


def now_str():
    return time.strftime("%Y%m%d_%H%M%S")


def rank0_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stable_hash_u32(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
