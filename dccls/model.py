# -*- coding: utf-8 -*-
"""
Model components:
- HFChunkEncoder: frozen HF causal LM used as chunk encoder
- ReadClassifierAttn: single-gate attention MIL pooling head
- ReadClassifierGatedAttn: gated-attention MIL pooling head (Ilse et al., 2018)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel



# -*- coding: utf-8 -*-
class HFChunkEncoder(nn.Module):
    """
    Wrap AutoModel as an encoder:
      input_ids (N,L) -> hidden_states[layer] (N,L,H) -> masked mean pool -> (N,H)
      -> proj -> (N,out_dim)

    Typical usage: keep frozen (requires_grad=False) and call under no_grad().
    """

    def __init__(
        self,
        model_path: str,
        vocab_size: int,
        pad_id: int,
        out_dim: int = 256,
        hidden_layer: int = -1,  # ✅ 新增：选择用哪一层（-1/-2/-3...）
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.hidden_layer = int(hidden_layer)

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        emb = self.model.get_input_embeddings()
        if emb is not None and vocab_size is not None and emb.num_embeddings < int(vocab_size):
            self.model.resize_token_embeddings(int(vocab_size))

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
        """
        input_ids: (N,L) int64
        returns:   (N,out_dim) float32/float16
        """
        attn = (input_ids != self.pad_id).to(torch.long)  # (N,L)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,  # 需要 hidden_states 才能选层
            return_dict=True,
        )

        hidden_states = out.hidden_states  # tuple(len = n_layers + 1), [0] 是 embedding
        try:
            hs = hidden_states[self.hidden_layer]  # (N,L,H)
        except IndexError:
            raise ValueError(
                f"hidden_layer={self.hidden_layer} out of range; "
                f"num_hidden_states={len(hidden_states)}"
            )

        mask = attn.unsqueeze(-1).to(hs.dtype)            # (N,L,1)
        denom = mask.sum(dim=1).clamp(min=1.0)           # (N,1)
        pooled = (hs * mask).sum(dim=1) / denom          # (N,H)

        z = self.ln(self.proj(pooled))                   # (N,out_dim)
        return z


# class HFChunkEncoder(nn.Module):
#     """
#     Wrap AutoModelForCausalLM as an encoder:
#       input_ids (N,L) -> hidden_states[-1] (N,L,H) -> masked mean pool -> (N,H)
#       -> proj -> (N,out_dim)
#     Typical usage: keep frozen (requires_grad=False) and call under no_grad().
#     """
#     def __init__(self, model_path: str, vocab_size: int, pad_id: int, out_dim: int = 256):
#         super().__init__()
#         self.pad_id = int(pad_id)

#         self.causal_lm = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             trust_remote_code=True,
#         )

#         emb = self.causal_lm.get_input_embeddings()
#         if emb is not None and vocab_size is not None and emb.num_embeddings < int(vocab_size):
#             self.causal_lm.resize_token_embeddings(int(vocab_size))

#         hidden_size = int(getattr(self.causal_lm.config, "hidden_size", out_dim))
#         self.proj = nn.Linear(hidden_size, out_dim, bias=False) if hidden_size != out_dim else nn.Identity()
#         self.ln = nn.LayerNorm(out_dim)

#     def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
#         """
#         input_ids: (N,L) int64
#         returns:   (N,out_dim) float32/float16
#         """
#         attn = (input_ids != self.pad_id).to(torch.long)  # (N,L)
#         out = self.causal_lm(
#             input_ids=input_ids,
#             attention_mask=attn,
#             output_hidden_states=True,
#             use_cache=False,
#         )
#         hs = out.hidden_states[-1]  # (N,L,H)
#         mask = attn.unsqueeze(-1).to(hs.dtype)  # (N,L,1)
#         denom = mask.sum(dim=1).clamp(min=1.0)  # (N,1)
#         pooled = (hs * mask).sum(dim=1) / denom  # (N,H)
#         z = self.ln(self.proj(pooled))  # (N,out_dim)
#         return z


class ReadClassifierAttn(nn.Module):
    """
    Single-gate Attention MIL pooling:
      ems: (B,K,D), chunk_mask: (B,K) bool
      -> logits (B,C)
    """
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
        x = self.pre(ems)                   # (B,K,D)
        scores = self.gate(x).squeeze(-1)   # (B,K)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~chunk_mask, neg_inf)
        attn = F.softmax(scores, dim=1)     # (B,K)

        read_emb = torch.einsum("bk,bkd->bd", attn, x)  # (B,D)
        logits = self.head(read_emb)                   # (B,C)
        if return_attn:
            return logits, attn
        return logits


class ReadClassifierGatedAttn(nn.Module):
    """
    Gated Attention MIL pooling (Ilse et al., 2018):
      h_k = f(x_k)
      a_k = softmax( w^T (tanh(V h_k) ⊙ sigmoid(U h_k)) / tau )
      H   = Σ a_k h_k
      logits = head(H)

    Extras:
      - temperature tau
      - attention dropout (with renorm over valid chunks)
    """
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
        x = self.pre(ems)  # (B,K,D)

        Vh = torch.tanh(self.V(x))      # (B,K,H)
        Uh = torch.sigmoid(self.U(x))   # (B,K,H)
        gh = Vh * Uh                    # (B,K,H)
        scores = self.w(gh).squeeze(-1) # (B,K)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~chunk_mask, neg_inf)

        tau = max(self.temperature, 1e-4)
        attn = F.softmax(scores / tau, dim=1)  # (B,K)

        # attn dropout + renorm over valid chunks
        attn = self.attn_drop(attn)
        attn = attn * chunk_mask.to(attn.dtype)
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom

        read_emb = torch.einsum("bk,bkd->bd", attn, x)  # (B,D)
        logits = self.head(read_emb)                    # (B,C)
        if return_attn:
            return logits, attn
        return logits
