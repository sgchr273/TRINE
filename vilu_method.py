"""
vilu_method.py

Inference-only implementation of ViLU – Vision-Language Uncertainty quantification
for failure prediction (Lafon et al., 2025).
  https://arxiv.org/abs/2507.07620

Architecture (Section 4.2 + Appendix A.2)
------------------------------------------
Given a query image embedding z_v  (d,)  and K L2-normalised text embeddings Z_t (K, d):

1. Cross-attention  (frozen weights W_Q, W_K, W_V  ∈ R^{d×d}):
       α   = softmax( (W_Q z_v)^T (W_K Z_t) / √d )      # (K,)
       z_t_alpha = Σ_j α_j (W_V z_{t_j})                 # (d,)

2. ViLU embedding:
       z_ViLU = concat(z_v, z_t_hat, z_t_alpha)          # (3d,)
   where z_t_hat is the text embedding of the CLIP-predicted class.

3. MLP  (4 layers: 3d → 512 → 256 → 128 → 1, ReLU activations, sigmoid output):
       failure_score = σ( MLP(z_ViLU) )                  # ∈ (0, 1)

Higher failure_score → more likely to be a CLIP misclassification.

Pre-trained weights
-------------------
Weights are loaded from a .pt or .ckpt file.  The file must contain a state_dict
with keys matching the ViLUModel nn.Module defined below.  Weights are expected
per-dataset and the path is resolved as:

    {weights_dir}/{dataset_name}.ckpt     (tried first)
    {weights_dir}/{dataset_name}.pt       (fallback)

or supplied directly via  --vilu_weights_path.

The expected state_dict keys are:
    cross_attn.W_Q.weight / .bias
    cross_attn.W_K.weight / .bias
    cross_attn.W_V.weight / .bias
    mlp.layers.{0,2,4,6}.weight / .bias    (Linear layers at even indices)

Public API (drop-in for other methods):

    out = compute_vilu_scores(
        query_embeddings      = query_pack["embeddings"].to(device),
        query_labels          = query_pack["labels"].to(device),
        query_clip_preds      = query_pack["clip_preds"].to(device),
        query_clip_pred_probs = query_pack["clip_pred_probs"].to(device),
        text_features         = text_bank.to(device),          # (K, d)
        class_names           = class_names,
        weights_path          = "/path/to/dataset.pt",
        chunk_size            = 2048,
        device                = device,
    )

Returns
-------
dict with keys:
    "summary"    – per-group statistics (correct / wrong CLIP predictions)
    "metrics"    – AUROC and FPR@95TPR (wrong prediction as positive class)
    "per_sample" – list of per-query dicts
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, tpr_thresh: float = 0.95) -> float:
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr_arr, tpr_thresh)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "median": float(np.median(arr)),
    }


# ---------------------------------------------------------------------------
# ViLU architecture
# ---------------------------------------------------------------------------

class CrossAttentionModule(nn.Module):
    """
    Single-head cross-attention with two operating modes:

    Mode A – separate projections (standard, bias=True, shared_kv=False):
        W_Q projects z_v   (image query)
        W_K projects Z_t   (text keys)
        W_V projects Z_t   (text values, separate weights)
        z_t_alpha = softmax(W_Q(z_v) @ W_K(Z_t)^T / √d) @ W_V(Z_t)

    Mode B – shared text projection (in_proj_v/in_proj_t convention,
              bias=False, shared_kv=True):
        W_Q  ≡  in_proj_v  – projects z_v
        W_K  ≡  in_proj_t  – projects Z_t for BOTH keys and values
        z_t_alpha = softmax(W_Q(z_v) @ W_K(Z_t)^T / √d) @ W_K(Z_t)

    The ViLU embedding is:
        z_ViLU = concat(W_Q(z_v),  W_K(z_t_hat),  z_t_alpha)
                         ^^^^^^^^   ^^^^^^^^^^^^^   ^^^^^^^^^
                         projected  projected pred  cross-attn
                         visual     text embed      output

    Critically, W_Q(z_v) and W_K(z_t_hat) are the PROJECTED versions,
    not the raw CLIP embeddings.  This matches the authors' checkpoint
    where the MLP receives already-projected features.
    """

    def __init__(self, embed_dim: int, bias: bool = True, shared_kv: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.shared_kv = shared_kv
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=bias)
        if not shared_kv:
            self.W_V = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        z_v: torch.Tensor,              # (B, d)  raw visual embedding
        Z_t: torch.Tensor,              # (K, d)  raw text embeddings
        z_t_hat_raw: torch.Tensor,      # (B, d)  raw predicted-class text embedding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        q_proj     : (B, d)  W_Q(z_v)            – projected visual
        t_hat_proj : (B, d)  W_K(z_t_hat_raw)    – projected predicted text
        z_t_alpha  : (B, d)  cross-attention output
        """
        q_proj = self.W_Q(z_v)                     # (B, d)
        K      = self.W_K(Z_t)                     # (K, d)
        V      = self.W_K(Z_t) if self.shared_kv else self.W_V(Z_t)   # (K, d)

        # Attention weights using projected query vs projected keys
        attn_scores = (q_proj @ K.T) * self.scale  # (B, K)
        alpha       = torch.softmax(attn_scores, dim=-1)  # (B, K)
        z_t_alpha   = alpha @ V                    # (B, d)

        # Project the predicted text embedding with the same text projection
        t_hat_proj = self.W_K(z_t_hat_raw)         # (B, d)

        return q_proj, t_hat_proj, z_t_alpha


class ViLUModel(nn.Module):
    """
    Full ViLU uncertainty predictor.

    Forward pass (Eq. 5-6 of the paper + authors' checkpoint convention):

        q_proj, t_hat_proj, z_t_alpha = CrossAttention(z_v, Z_t, z_t_hat)
        z_ViLU = concat(q_proj, t_hat_proj, z_t_alpha)   # (3d,)
        failure_score = σ( MLP(z_ViLU) )                 # → 1 if likely wrong

    The MLP input dim is 3d (= 1536 for CLIP ViT-B/32 with d=512).
    """

    MLP_HIDDEN = [512, 256, 128]

    def __init__(self, embed_dim: int, bias: bool = True, shared_kv: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = CrossAttentionModule(embed_dim, bias=bias, shared_kv=shared_kv)

        in_dim = 3 * embed_dim
        layers: list[nn.Module] = []
        for h in self.MLP_HIDDEN:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_v: torch.Tensor,               # (B, d)
        Z_t: torch.Tensor,               # (K, d)
        clip_pred_indices: torch.Tensor, # (B,) long
    ) -> torch.Tensor:                   # (B,) failure scores ∈ (0,1)
        z_t_hat_raw = Z_t[clip_pred_indices]   # (B, d)  raw predicted text embed

        q_proj, t_hat_proj, z_t_alpha = self.cross_attn(z_v, Z_t, z_t_hat_raw)

        # z_ViLU = [projected visual, projected pred-text, cross-attn output]
        z_vilu = torch.cat([q_proj, t_hat_proj, z_t_alpha], dim=-1)  # (B, 3d)

        logit = self.mlp(z_vilu).squeeze(-1)    # (B,)
        return torch.sigmoid(logit)             # (B,) → 1 means likely wrong


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

# Known alternative naming conventions used by different ViLU checkpoints.
# Each entry is (ckpt_key_fragment, vilu_key_fragment).
# Applied after prefix stripping; tried in order until the model loads cleanly.
_CROSS_ATTN_ALIASES = [
    # Our naming (default)
    ("cross_attn.W_Q", "cross_attn.W_Q"),
    ("cross_attn.W_K", "cross_attn.W_K"),
    ("cross_attn.W_V", "cross_attn.W_V"),
]

_CROSS_ATTN_REMAPS = [
    # Common alternative: q_proj / k_proj / v_proj style
    {
        "cross_attn.q_proj": "cross_attn.W_Q",
        "cross_attn.k_proj": "cross_attn.W_K",
        "cross_attn.v_proj": "cross_attn.W_V",
    },
    # attention.query / attention.key / attention.value style
    {
        "attention.query": "cross_attn.W_Q",
        "attention.key":   "cross_attn.W_K",
        "attention.value": "cross_attn.W_V",
    },
    # attn.W_Q / attn.W_K / attn.W_V (short prefix)
    {
        "attn.W_Q": "cross_attn.W_Q",
        "attn.W_K": "cross_attn.W_K",
        "attn.W_V": "cross_attn.W_V",
    },
    # cross_attention.query_proj etc.
    {
        "cross_attention.query_proj": "cross_attn.W_Q",
        "cross_attention.key_proj":   "cross_attn.W_K",
        "cross_attention.value_proj": "cross_attn.W_V",
    },
]

# Special case: in_proj_v / in_proj_t convention (ViLU authors' released checkpoints).
# in_proj_v  →  W_Q  (visual projection, query)
# in_proj_t  →  W_K  (text projection; K and V share the same weight matrix)
# W_V is absent – the model must be built with shared_kv=True for this checkpoint.
_IN_PROJ_CONVENTION = "in_proj_v" in ["in_proj_v"]  # sentinel – detected at load time


def _try_remap(state_dict: dict, remap: dict) -> dict:
    """Apply a {old_fragment -> new_fragment} remap to state_dict keys."""
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for old_frag, new_frag in remap.items():
            if old_frag in k:
                new_k = k.replace(old_frag, new_frag)
                break
        remapped[new_k] = v
    return remapped


def _strip_prefix(state_dict: dict) -> dict:
    """Strip common trainer prefixes (model., module.) from all keys."""
    for prefix in ("model.", "module."):
        if all(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_vilu_model(
    weights_path: str,
    embed_dim: int,
    device: torch.device,
    key_prefix: str = "",
) -> ViLUModel:
    """
    Load a ViLUModel from a .pt or .ckpt checkpoint file.

    Supported checkpoint formats
    ----------------------------
    1. Raw state_dict                    – dict of tensors
    2. {"state_dict": ..., ...}          – PyTorch / Lightning wrapper
    3. {"model": ..., ...}               – alternative wrapper key
    4. Lightning .ckpt with "model." prefix on all keys (stripped automatically)
    5. DataParallel .ckpt with "module." prefix (stripped automatically)

    Cross-attention key remapping
    ------------------------------
    If the checkpoint uses a different naming convention for cross-attention
    weights (e.g. q_proj / k_proj / v_proj), several alternatives are tried
    automatically before raising an error.

    Parameters
    ----------
    key_prefix : str
        Extra prefix to strip from all checkpoint keys before loading.
        Useful when the checkpoint has a unique top-level wrapper not covered
        by the automatic stripping, e.g. key_prefix="encoder.uncertainty_head.".
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"[ViLU] Weights file not found: {weights_path}\n"
            "Please supply --vilu_weights_path or --vilu_weights_dir pointing to "
            "your pre-trained .pt / .ckpt file."
        )

    raw = torch.load(weights_path, map_location="cpu", weights_only=False)

    # ---- unwrap checkpoint wrappers ----
    if isinstance(raw, dict):
        if "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif "model" in raw:
            state_dict = raw["model"]
        else:
            state_dict = raw
    else:
        raise ValueError(f"[ViLU] Unexpected checkpoint type: {type(raw)}")

    # ---- strip common trainer prefixes ----
    state_dict = _strip_prefix(state_dict)

    # ---- strip user-supplied prefix ----
    if key_prefix:
        state_dict = {
            (k[len(key_prefix):] if k.startswith(key_prefix) else k): v
            for k, v in state_dict.items()
        }

    # ---- infer / validate embed_dim from checkpoint ----
    for dim_key in (
        "cross_attn.W_Q.weight", "cross_attn.q_proj.weight",
        "attn.W_Q.weight", "attention.query.weight",
    ):
        if dim_key in state_dict:
            ckpt_dim = state_dict[dim_key].shape[0]
            if ckpt_dim != embed_dim:
                print(
                    f"[ViLU] embed_dim from checkpoint ({ckpt_dim}) != provided "
                    f"({embed_dim}). Using checkpoint value."
                )
                embed_dim = ckpt_dim
            break

    # ---- detect in_proj_v / in_proj_t convention (ViLU authors' checkpoints) ----
    uses_in_proj = "in_proj_v.weight" in state_dict

    if uses_in_proj:
        # Build a remapped state_dict for the shared-KV model variant:
        #   in_proj_v  →  cross_attn.W_Q   (visual / query projection)
        #   in_proj_t  →  cross_attn.W_K   (text projection; K and V share weights)
        remapped_sd = {}
        for k, v in state_dict.items():
            if k == "in_proj_v.weight":
                remapped_sd["cross_attn.W_Q.weight"] = v
            elif k == "in_proj_t.weight":
                remapped_sd["cross_attn.W_K.weight"] = v
            elif k == "iteration_count":
                pass  # training counter – not a model parameter, skip
            else:
                remapped_sd[k] = v
        state_dict = remapped_sd
        # Build model with shared_kv=True and no projection bias
        model = ViLUModel(embed_dim=embed_dim, bias=False, shared_kv=True)
        print("[ViLU] Detected in_proj_v/in_proj_t convention (shared K/V, no bias).")
    else:
        model = ViLUModel(embed_dim=embed_dim)

    # ---- attempt loading; try generic key remaps if needed ----
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing and not uses_in_proj:
        # Try each known cross-attention naming remap
        for remap in _CROSS_ATTN_REMAPS:
            remapped = _try_remap(state_dict, remap)
            missing2, unexpected2 = model.load_state_dict(remapped, strict=False)
            if not missing2:
                print(f"[ViLU] Loaded with key remap: {remap}")
                missing, unexpected = missing2, unexpected2
                break

    if missing:
        # Nothing worked – print full key listings to help the user debug
        print("\n[ViLU] === CHECKPOINT KEYS ===")
        for k in sorted(state_dict.keys()):
            print(f"  {k}  shape={tuple(state_dict[k].shape)}")
        print("\n[ViLU] === EXPECTED MODEL KEYS ===")
        for k, v in sorted(model.state_dict().items()):
            print(f"  {k}  shape={tuple(v.shape)}")
        raise RuntimeError(
            f"[ViLU] Could not load weights. Missing keys after all remap attempts:\n"
            f"  {missing}\n\n"
            "Compare the CHECKPOINT KEYS and EXPECTED MODEL KEYS printed above.\n"
            "Then either:\n"
            "  • Pass --vilu_key_prefix <prefix> to strip an extra prefix, or\n"
            "  • Open an issue with the key listing so a remap can be added."
        )

    if unexpected:
        print(f"[ViLU] Ignoring unexpected checkpoint keys: {unexpected}")

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_vilu_scores(
    # ---- required ----
    query_embeddings: torch.Tensor,         # (N_query, d)
    query_labels: torch.Tensor,             # (N_query,)
    query_clip_preds: torch.Tensor,         # (N_query,)
    text_features: torch.Tensor,            # (K, d)  L2-normalised text embeddings
    class_names: List[str],
    weights_path: str,                      # path to pre-trained .pt / .ckpt weights
    # ---- unused, kept for API compatibility ----
    bank_embeddings: Optional[torch.Tensor] = None,
    bank_labels: Optional[torch.Tensor] = None,
    query_clip_pred_probs: Optional[torch.Tensor] = None,
    # ---- misc ----
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
    key_prefix: str = "",
) -> dict:
    """
    Run ViLU inference and return failure scores.

    Parameters
    ----------
    weights_path : str
        Path to the .pt or .ckpt file containing the pre-trained ViLUModel state_dict.
    chunk_size : int
        Number of query samples processed per forward pass (avoids OOM).
    key_prefix : str
        Optional prefix to strip from all checkpoint keys before loading.
        Only needed if automatic prefix stripping does not cover your checkpoint format.
    """
    if device is None:
        device = query_embeddings.device

    # L2-normalise embeddings (matches CLIP's convention)
    query_embeddings = F.normalize(query_embeddings.to(device).float(), dim=-1)
    text_features    = F.normalize(text_features.to(device).float(), dim=-1)
    query_labels     = query_labels.to(device)
    query_clip_preds = query_clip_preds.to(device)

    embed_dim = query_embeddings.shape[1]
    N_query   = query_embeddings.shape[0]

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print(f"[ViLU] Loading weights from: {weights_path}")
    model = load_vilu_model(
        weights_path, embed_dim=embed_dim, device=device, key_prefix=key_prefix
    )

    # ------------------------------------------------------------------ #
    # Chunked inference
    # ------------------------------------------------------------------ #
    all_failure_scores = torch.empty(N_query, device=device, dtype=torch.float32)

    for start in range(0, N_query, chunk_size):
        end = min(start + chunk_size, N_query)
        z_v_chunk   = query_embeddings[start:end]        # (B, d)
        pred_chunk  = query_clip_preds[start:end].long() # (B,)

        scores_chunk = model(z_v_chunk, text_features, pred_chunk)  # (B,)
        all_failure_scores[start:end] = scores_chunk

    # ------------------------------------------------------------------ #
    # Correctness mask
    # ------------------------------------------------------------------ #
    is_correct    = (query_clip_preds == query_labels)    # (N_query,) bool
    correct_mask  = is_correct.cpu().numpy()
    wrong_mask    = ~correct_mask
    fail_np       = all_failure_scores.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    summary = {
        "correct_clip_predictions": {
            "count":         int(correct_mask.sum()),
            "failure_score": _stats(fail_np[correct_mask]) if correct_mask.any() else {},
        },
        "wrong_clip_predictions": {
            "count":         int(wrong_mask.sum()),
            "failure_score": _stats(fail_np[wrong_mask]) if wrong_mask.any() else {},
        },
    }

    # ------------------------------------------------------------------ #
    # AUROC and FPR@95TPR  (wrong prediction = positive class)
    # ------------------------------------------------------------------ #
    binary_labels = wrong_mask.astype(int)

    if binary_labels.sum() > 0 and (1 - binary_labels).sum() > 0:
        auroc = float(roc_auc_score(binary_labels, fail_np))
        fpr95 = _fpr_at_tpr(binary_labels, fail_np, tpr_thresh=0.95)
    else:
        auroc = float("nan")
        fpr95 = float("nan")

    metrics = {
        "auroc_wrong_positive":         auroc,
        "fpr_at_95_tpr_wrong_positive": fpr95,
    }

    # ------------------------------------------------------------------ #
    # Per-sample records
    # ------------------------------------------------------------------ #
    query_labels_np = query_labels.cpu().numpy()
    clip_preds_np   = query_clip_preds.cpu().numpy()

    per_sample = [
        {
            "true_label":      int(query_labels_np[i]),
            "clip_pred":       int(clip_preds_np[i]),
            "is_clip_correct": bool(is_correct[i].item()),
            "failure_score":   float(fail_np[i]),
        }
        for i in range(N_query)
    ]

    return {
        "summary":    summary,
        "metrics":    metrics,
        "per_sample": per_sample,
    }