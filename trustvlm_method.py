"""
trustvlm_method.py

Implementation of TrustVLM (Dong et al., 2025):
  "To Trust Or Not To Trust Your Vision-Language Model's Prediction"
  https://arxiv.org/abs/2505.23745

TrustVLM augments the standard CLIP image-to-text confidence score (MSP / S_i-t)
with an image-to-image similarity score (S_i-i) computed against per-class visual
prototypes built from the training bank.

Confidence:  κ(x) = S_i-t + S_i-i
Failure score: 1 - κ(x)   (higher → more likely to be a CLIP failure)

Three variants are implemented (controlled by `variant` argument):
  - "base"   : TrustVLM   – uses fixed prototypes, CLIP image encoder (same as bank_embeddings)
  - "star"   : TrustVLM*  – also re-scores classification via image-to-image probabilities
                             (improves accuracy *and* MisD)
  - "aux"    : same as "base" but intended for use with an external auxiliary encoder
               (DINOv2, MoCo-v2, …); pass pre-extracted aux_bank_embeddings /
               aux_query_embeddings instead of the CLIP ones.

Usage (drop-in for compute_knn_scores_train_bank_test_query):

    out = compute_trustvlm_scores(
        bank_embeddings=bank_pack["embeddings"].to(device),
        bank_labels=bank_pack["labels"].to(device),
        query_embeddings=query_pack["embeddings"].to(device),
        query_labels=query_pack["labels"].to(device),
        query_clip_preds=query_pack["clip_preds"].to(device),
        query_clip_pred_probs=query_pack["clip_pred_probs"].to(device),
        class_names=class_names,
        n_shot=16,          # prototypes averaged over up to n_shot bank samples per class
        tau=0.01,           # temperature for softmax re-scoring (TrustVLM*)
        variant="base",     # "base" | "star"
        device=device,
    )
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Prototype builder
# ---------------------------------------------------------------------------

def build_visual_prototypes(
    bank_embeddings: torch.Tensor,   # (N, D) – L2-normalised image embeddings
    bank_labels: torch.Tensor,       # (N,)   – integer class indices
    num_classes: int,
    n_shot: int = -1,                # ≤0 → use all available samples per class
    random_seed: int = 0,
) -> torch.Tensor:
    """
    Compute per-class prototype embeddings by averaging (then re-normalising)
    up to `n_shot` bank embeddings per class.

    Returns
    -------
    prototypes : (C, D) float32 tensor on the same device as bank_embeddings
    """
    device = bank_embeddings.device
    D = bank_embeddings.shape[1]
    prototypes = torch.zeros(num_classes, D, device=device, dtype=bank_embeddings.dtype)

    rng = np.random.default_rng(random_seed)

    for c in range(num_classes):
        idx = (bank_labels == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        if 0 < n_shot < len(idx):
            chosen = rng.choice(len(idx), size=n_shot, replace=False)
            idx = idx[chosen]
        proto = bank_embeddings[idx].mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes[c] = proto

    return prototypes   # (C, D)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_trustvlm_scores(
    bank_embeddings: torch.Tensor,        # (N_bank, D)
    bank_labels: torch.Tensor,            # (N_bank,)
    query_embeddings: torch.Tensor,       # (N_query, D) – CLIP image embeddings
    query_labels: torch.Tensor,           # (N_query,)
    query_clip_preds: torch.Tensor,       # (N_query,) – CLIP zero-shot predictions
    query_clip_pred_probs: torch.Tensor,  # (N_query,) – CLIP softmax prob for pred class (S_i-t)
    class_names: List[str],
    # ---- optional auxiliary encoder embeddings (for TrustVLM-D / -M / etc.) ----
    aux_bank_embeddings: Optional[torch.Tensor] = None,   # (N_bank, D_aux)
    aux_query_embeddings: Optional[torch.Tensor] = None,  # (N_query, D_aux)
    # ---- hyper-parameters -------------------------------------------------------
    n_shot: int = 16,        # samples per class for prototype computation
    tau: float = 0.01,       # temperature for TrustVLM* re-scoring
    variant: str = "base",   # "base" (TrustVLM) | "star" (TrustVLM*)
    random_seed: int = 0,
    device: Optional[torch.device] = None,
    chunk_size: int = 2048,
) -> dict:
    """
    Compute TrustVLM confidence / failure scores.

    Returns a dict with the same structure as compute_knn_scores_train_bank_test_query:
        {
          "summary":    { "correct_clip_predictions": {...}, "wrong_clip_predictions": {...} },
          "metrics":    { "auroc_wrong_positive": float, "fpr_at_95_tpr_wrong_positive": float },
          "per_sample": [ { "true_label", "clip_pred", "is_clip_correct",
                            "s_i_t", "s_i_i", "confidence", "failure_score" }, ... ]
        }
    """
    if device is None:
        device = bank_embeddings.device

    # Move everything to device
    bank_embeddings      = bank_embeddings.to(device)
    bank_labels          = bank_labels.to(device)
    query_embeddings     = query_embeddings.to(device)
    query_labels         = query_labels.to(device)
    query_clip_preds     = query_clip_preds.to(device)
    query_clip_pred_probs = query_clip_pred_probs.to(device)

    num_classes = len(class_names)

    # ------------------------------------------------------------------
    # Decide which embeddings to use for prototypes / image-to-image sim
    # ------------------------------------------------------------------
    if aux_bank_embeddings is not None and aux_query_embeddings is not None:
        proto_bank_emb  = F.normalize(aux_bank_embeddings.to(device).float(), dim=-1)
        proto_query_emb = F.normalize(aux_query_embeddings.to(device).float(), dim=-1)
    else:
        proto_bank_emb  = F.normalize(bank_embeddings.float(), dim=-1)
        proto_query_emb = F.normalize(query_embeddings.float(), dim=-1)

    # ------------------------------------------------------------------
    # Step 1 – build visual prototypes {P_c}
    # ------------------------------------------------------------------
    prototypes = build_visual_prototypes(
        bank_embeddings=proto_bank_emb,
        bank_labels=bank_labels,
        num_classes=num_classes,
        n_shot=n_shot,
        random_seed=random_seed,
    )  # (C, D)

    # ------------------------------------------------------------------
    # Step 2 – image-to-text confidence  S_i-t  (already in query_clip_pred_probs)
    # ------------------------------------------------------------------
    s_i_t = query_clip_pred_probs.float()   # (N_query,)

    # ------------------------------------------------------------------
    # Step 3 – image-to-image confidence  S_i-i
    #   For each query, look up the prototype of the CLIP-predicted class
    #   and compute cosine similarity with the query embedding.
    # ------------------------------------------------------------------
    # proto_query_emb: (N_query, D), prototypes: (C, D)
    # We index prototypes by the CLIP prediction.
    predicted_prototypes = prototypes[query_clip_preds.long()]  # (N_query, D)
    s_i_i = (proto_query_emb * predicted_prototypes).sum(dim=-1)  # (N_query,) cosine sim

    # ------------------------------------------------------------------
    # TrustVLM*: re-rank predictions using combined image-to-image probs
    # ------------------------------------------------------------------
    if variant == "star":
        # Compute softmax over image-to-image similarities for all classes
        # shape: (N_query, C)
        N_query = proto_query_emb.shape[0]
        ii_logits = torch.zeros(N_query, num_classes, device=device)

        # chunked to avoid OOM on large datasets
        for start in range(0, N_query, chunk_size):
            end = min(start + chunk_size, N_query)
            chunk = proto_query_emb[start:end]           # (chunk, D)
            sims  = chunk @ prototypes.T                 # (chunk, C)
            ii_logits[start:end] = sims

        ii_probs = torch.softmax(ii_logits / tau, dim=-1)   # (N_query, C)

        # Re-derive CLIP text-based softmax probabilities are NOT stored per-class
        # in the standard pipeline, so for TrustVLM* we approximate S_i-t as the
        # existing clip_pred_probs (max-class) and combine with the i-i softmax argmax.
        # Full TrustVLM* would need all-class text logits; here we use the i-i re-scoring
        # to update the final prediction.
        star_preds     = ii_probs.argmax(dim=-1)                 # (N_query,)
        star_max_prob  = ii_probs.max(dim=-1).values             # (N_query,)

        # Overwrite clip_preds and update s_i_i with the new prediction's prototype sim
        query_clip_preds_for_scoring = star_preds
        predicted_prototypes_star    = prototypes[star_preds.long()]
        s_i_i = (proto_query_emb * predicted_prototypes_star).sum(dim=-1)
    else:
        query_clip_preds_for_scoring = query_clip_preds

    # ------------------------------------------------------------------
    # Combined confidence  κ(x) = S_i-t + S_i-i
    # Failure score = 1 - κ(x)  (higher = more likely a CLIP error)
    # ------------------------------------------------------------------
    confidence    = s_i_t + s_i_i                   # (N_query,)
    failure_score = 1.0 - confidence                # (N_query,)

    # ------------------------------------------------------------------
    # Determine correctness
    # ------------------------------------------------------------------
    is_clip_correct = (query_clip_preds_for_scoring == query_labels)  # (N_query,) bool

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    def _stats(scores: torch.Tensor) -> dict:
        arr = scores.cpu().numpy()
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
        }

    correct_mask = is_clip_correct.cpu()
    wrong_mask   = ~correct_mask

    summary = {
        "correct_clip_predictions": {
            "count":         int(correct_mask.sum()),
            "failure_score": _stats(failure_score[correct_mask]),
            "confidence":    _stats(confidence[correct_mask]),
            "s_i_t":         _stats(s_i_t[correct_mask]),
            "s_i_i":         _stats(s_i_i[correct_mask]),
        },
        "wrong_clip_predictions": {
            "count":         int(wrong_mask.sum()),
            "failure_score": _stats(failure_score[wrong_mask]),
            "confidence":    _stats(confidence[wrong_mask]),
            "s_i_t":         _stats(s_i_t[wrong_mask]),
            "s_i_i":         _stats(s_i_i[wrong_mask]),
        },
    }

    # ------------------------------------------------------------------
    # AUROC / FPR@95TPR  (wrong predictions as positive class)
    # ------------------------------------------------------------------
    labels_np        = (~correct_mask).numpy().astype(int)   # 1 = wrong (positive)
    fail_scores_np   = failure_score.cpu().numpy()

    auroc = float(roc_auc_score(labels_np, fail_scores_np)) if labels_np.sum() > 0 else float("nan")
    fpr95 = _compute_fpr_at_tpr(labels_np, fail_scores_np, tpr_threshold=0.95)

    metrics = {
        "auroc_wrong_positive":      auroc,
        "fpr_at_95_tpr_wrong_positive": fpr95,
    }

    # ------------------------------------------------------------------
    # Per-sample records
    # ------------------------------------------------------------------
    per_sample = []
    for i in range(len(query_labels)):
        per_sample.append({
            "true_label":     int(query_labels[i].item()),
            "clip_pred":      int(query_clip_preds_for_scoring[i].item()),
            "is_clip_correct": bool(is_clip_correct[i].item()),
            "s_i_t":          float(s_i_t[i].item()),
            "s_i_i":          float(s_i_i[i].item()),
            "confidence":     float(confidence[i].item()),
            "failure_score":  float(failure_score[i].item()),
        })

    return {
        "summary":    summary,
        "metrics":    metrics,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Utility: FPR at given TPR level
# ---------------------------------------------------------------------------

def _compute_fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    tpr_threshold: float = 0.95,
) -> float:
    """Compute FPR when TPR >= tpr_threshold."""
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Find the smallest FPR where TPR >= threshold
    idx = np.searchsorted(tpr_arr, tpr_threshold)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])
