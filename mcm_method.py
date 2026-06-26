"""
mcm_method.py

Implementation of MCM – Maximum Concept Matching for OOD / Misclassification Detection
(Ming et al., NeurIPS 2022).
  https://arxiv.org/abs/2211.13445

Method overview
---------------
Standard CLIP zero-shot classification computes cosine similarities between a query
image embedding and K class text embeddings, then takes the argmax.  A key observation
is that OOD (or misclassified) inputs produce *nearly uniform* cosine similarity scores
across all ID classes – there is no dominant concept match.  Directly thresholding on the
maximum cosine similarity alone gives poor ID/OOD separation because the raw gap is small.

MCM addresses this by applying softmax scaling to the cosine similarities before taking the
maximum.  The softmax sharpens the distribution for ID samples (one concept clearly wins)
while keeping the OOD distribution nearly flat, dramatically improving separability.

MCM confidence score (Eq. 1 of the paper):

    S_MCM(x) = max_{i in 1..K}   exp(s_i(x) / τ)
                                 ─────────────────────────────
                                 Σ_{j=1}^{K} exp(s_j(x) / τ)

where s_i(x) = cosine_similarity(image_embedding, text_embedding_i)
and τ is the temperature (default 1.0, paper shows τ ∈ [0.5, 100] all work similarly).

This is exactly the maximum softmax probability (MSP) computed over *cosine similarities*
rather than over learned logits.

Failure / OOD score = 1 − S_MCM(x)   (higher → more likely OOD / misclassified).

Key properties (from the paper):
  - Training-free, zero-shot, OOD-agnostic.
  - Uses ONLY the pre-computed text bank and query image embeddings – no training data needed.
  - Temperature τ=1 is the default; the method is robust to τ in [0.5, 100].
  - Also supports the "without softmax" variant S_MCM^wo = max_i s_i(x)
    (set tau=0 to activate this mode).

Public API (drop-in replacement for compute_knn_scores_train_bank_test_query):

    out = compute_mcm_scores(
        query_embeddings      = query_pack["embeddings"].to(device),
        query_labels          = query_pack["labels"].to(device),
        query_clip_preds      = query_pack["clip_preds"].to(device),
        query_clip_pred_probs = query_pack["clip_pred_probs"].to(device),
        text_features         = text_bank.to(device),   # (C, d) L2-normalised
        class_names           = class_names,
        tau                   = 1.0,
        device                = device,
    )

Note: bank_embeddings / bank_labels are accepted for API compatibility but are NOT used
by MCM – the method requires only the text features and query image embeddings.

Returns
-------
dict with keys:
    "summary"    – per-group statistics (correct / wrong CLIP predictions)
    "metrics"    – AUROC and FPR@95TPR (wrong prediction as positive class)
    "per_sample" – list of per-query dicts
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, tpr_thresh: float = 0.95) -> float:
    """Return FPR when TPR first reaches tpr_thresh."""
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
# Core scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mcm_scores(
    # ---- required ----
    query_embeddings: torch.Tensor,        # (N_query, d)  L2-normalised image embeddings
    query_labels: torch.Tensor,            # (N_query,)    ground-truth class indices
    query_clip_preds: torch.Tensor,        # (N_query,)    CLIP zero-shot predictions
    text_features: torch.Tensor,           # (C, d)        L2-normalised text embeddings
    class_names: List[str],
    # ---- unused, kept for API compatibility with other methods ----
    bank_embeddings: Optional[torch.Tensor] = None,
    bank_labels: Optional[torch.Tensor] = None,
    query_clip_pred_probs: Optional[torch.Tensor] = None,
    # ---- hyper-parameters ----
    tau: float = 1.0,
    # ---- misc ----
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compute MCM confidence and failure scores for every query image.

    Parameters
    ----------
    tau : float
        Softmax temperature.  τ=1 is the paper default.
        Set tau=0 to use the "without softmax" variant S_MCM^wo = max_i cos(v, t_i).
    """
    if device is None:
        device = query_embeddings.device

    query_embeddings = F.normalize(query_embeddings.to(device).float(), dim=-1)  # (N, d)
    text_features    = F.normalize(text_features.to(device).float(), dim=-1)     # (C, d)
    query_labels     = query_labels.to(device)
    query_clip_preds = query_clip_preds.to(device)

    N_query, C = query_embeddings.shape[0], text_features.shape[0]

    # ------------------------------------------------------------------
    # Cosine similarities: (N_query, C)
    # Computed in chunks to avoid OOM on large datasets.
    # ------------------------------------------------------------------
    all_cos_sims = torch.empty(N_query, C, device=device, dtype=torch.float32)

    for start in range(0, N_query, chunk_size):
        end = min(start + chunk_size, N_query)
        chunk = query_embeddings[start:end]          # (chunk, d)
        all_cos_sims[start:end] = chunk @ text_features.T  # (chunk, C)

    # ------------------------------------------------------------------
    # MCM score = max softmax probability over cosine similarities
    # ------------------------------------------------------------------
    if tau == 0:
        # S_MCM^wo variant: no softmax, just max cosine similarity
        mcm_score = all_cos_sims.max(dim=-1).values          # (N_query,)
    else:
        # Standard MCM (Eq. 1): softmax-scaled then max
        # softmax( s / τ )  →  max over classes
        scaled = all_cos_sims / tau                          # (N_query, C)
        softmax_probs = torch.softmax(scaled, dim=-1)        # (N_query, C)
        mcm_score = softmax_probs.max(dim=-1).values         # (N_query,)

    failure_score = 1.0 - mcm_score                         # (N_query,)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    is_correct = query_clip_preds == query_labels            # (N_query,) bool

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    correct_mask = is_correct.cpu().numpy()
    wrong_mask   = ~correct_mask
    mcm_np       = mcm_score.cpu().numpy()
    fail_np      = failure_score.cpu().numpy()
    cos_np       = all_cos_sims.cpu().numpy()
    max_cos_np   = cos_np.max(axis=-1)

    summary = {
        "correct_clip_predictions": {
            "count":           int(correct_mask.sum()),
            "failure_score":   _stats(fail_np[correct_mask])    if correct_mask.any() else {},
            "mcm_score":       _stats(mcm_np[correct_mask])     if correct_mask.any() else {},
            "max_cos_sim":     _stats(max_cos_np[correct_mask]) if correct_mask.any() else {},
        },
        "wrong_clip_predictions": {
            "count":           int(wrong_mask.sum()),
            "failure_score":   _stats(fail_np[wrong_mask])    if wrong_mask.any() else {},
            "mcm_score":       _stats(mcm_np[wrong_mask])     if wrong_mask.any() else {},
            "max_cos_sim":     _stats(max_cos_np[wrong_mask]) if wrong_mask.any() else {},
        },
    }

    # ------------------------------------------------------------------
    # AUROC and FPR@95TPR  (wrong prediction = positive class)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Per-sample records
    # ------------------------------------------------------------------
    query_labels_np  = query_labels.cpu().numpy()
    clip_preds_np    = query_clip_preds.cpu().numpy()

    per_sample = [
        {
            "true_label":      int(query_labels_np[i]),
            "clip_pred":       int(clip_preds_np[i]),
            "is_clip_correct": bool(is_correct[i].item()),
            "mcm_score":       float(mcm_np[i]),
            "max_cos_sim":     float(max_cos_np[i]),
            "failure_score":   float(fail_np[i]),
        }
        for i in range(N_query)
    ]

    return {
        "summary":    summary,
        "metrics":    metrics,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Convenience alias keeping the naming convention of other method files
# ---------------------------------------------------------------------------
compute_mcm_scores_train_bank_test_query = compute_mcm_scores
