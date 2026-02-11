# -*- coding: utf-8 -*-
import numpy as np
import torch


def topk_acc(logits: torch.Tensor, y: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        num_classes = logits.size(1)
        maxk = min(max(topk), num_classes)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B,maxk)
        correct = pred.eq(y.view(-1, 1))  # (B,maxk)
        out = []
        for k in topk:
            k = min(k, num_classes)
            acc = correct[:, :k].any(dim=1).float().mean().item()
            out.append(acc)
    return tuple(out)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    y_true/y_pred: 1D int arrays
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm
