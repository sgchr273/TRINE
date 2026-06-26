# bayesvlm_method.py

import math
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Metric utilities
# ------------------------------------------------------------

def _rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    AUROC with wrong samples as positive.
    Pure NumPy implementation with average ranks for ties.
    """
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1

        # ranks are 1-indexed
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Average precision with wrong samples as positive.
    """
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    precision = tp / np.maximum(tp + fp, 1)
    ap = precision[y_sorted == 1].sum() / n_pos

    return float(ap)


def _fpr_at_tpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    FPR at target TPR with wrong samples as positive.
    """
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    tpr = tp / n_pos
    fpr = fp / n_neg

    valid = fpr[tpr >= target_tpr]
    if len(valid) == 0:
        return 1.0

    return float(valid.min())


def _summary(scores: np.ndarray) -> Dict[str, float]:
    if len(scores) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


# ------------------------------------------------------------
# Variance proxy utilities
# ------------------------------------------------------------

@torch.no_grad()
def _streaming_topk_mean_similarity(
    query_embeddings: torch.Tensor,
    bank_embeddings: torch.Tensor,
    k: int,
    query_chunk_size: int = 256,
    bank_chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Computes mean top-k cosine similarity without materializing the full
    query-bank similarity matrix.

    Used only when posterior_proxy == "local".
    """
    device = query_embeddings.device
    num_query = query_embeddings.shape[0]
    num_bank = bank_embeddings.shape[0]

    k_eff = min(int(k), int(num_bank))
    all_mean_topk = []

    for qs in range(0, num_query, query_chunk_size):
        qe = min(qs + query_chunk_size, num_query)
        q = query_embeddings[qs:qe]

        running_topk = None

        for bs in range(0, num_bank, bank_chunk_size):
            be = min(bs + bank_chunk_size, num_bank)
            b = bank_embeddings[bs:be]

            sims = q @ b.T
            local_k = min(k_eff, sims.shape[1])
            vals = torch.topk(sims, k=local_k, dim=1).values

            if running_topk is None:
                running_topk = vals
            else:
                running_topk = torch.cat([running_topk, vals], dim=1)
                if running_topk.shape[1] > k_eff:
                    running_topk = torch.topk(
                        running_topk,
                        k=k_eff,
                        dim=1,
                    ).values

        all_mean_topk.append(running_topk.mean(dim=1))

    return torch.cat(all_mean_topk, dim=0).to(device)


@torch.no_grad()
def _compute_class_centroids_and_vars(
    bank_embeddings: torch.Tensor,
    bank_labels: torch.Tensor,
    num_classes: int,
    global_var: torch.Tensor,
    var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes class-wise centroids and diagonal variances from the bank.
    """
    device = bank_embeddings.device
    dim = bank_embeddings.shape[1]

    centroids = torch.zeros(num_classes, dim, device=device)
    class_vars = torch.zeros(num_classes, dim, device=device)

    for c in range(num_classes):
        idx = bank_labels == c

        if idx.sum() < 2:
            centroids[c] = torch.zeros(dim, device=device)
            class_vars[c] = global_var
            continue

        x = bank_embeddings[idx]
        mu = x.mean(dim=0)
        centroids[c] = F.normalize(mu, dim=0)
        class_vars[c] = torch.var(x, dim=0, unbiased=False).clamp_min(var_floor)

    return centroids, class_vars


@torch.no_grad()
def _estimate_query_embedding_vars(
    bank_embeddings: torch.Tensor,
    bank_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_clip_preds: torch.Tensor,
    num_classes: int,
    image_var_scale: float,
    var_floor: float,
    posterior_proxy: str,
    local_k: int,
    local_beta: float,
    chunk_size: int,
) -> torch.Tensor:
    """
    Proxy for BayesVLM diagonal image-embedding variance.

    Full BayesVLM would compute:
        Sigma_g = (phi(x)^T A_img^{-1} phi(x)) B_img^{-1}

    Since your current pipeline only has final CLIP embeddings, this estimates
    a diagonal proxy variance from the bank embeddings.
    """
    global_var = torch.var(
        bank_embeddings,
        dim=0,
        unbiased=False,
    ).clamp_min(var_floor)

    base_var = image_var_scale * global_var
    num_query = query_embeddings.shape[0]

    if posterior_proxy == "global":
        return base_var.unsqueeze(0).repeat(num_query, 1)

    if posterior_proxy == "local":
        mean_topk_sim = _streaming_topk_mean_similarity(
            query_embeddings=query_embeddings,
            bank_embeddings=bank_embeddings,
            k=local_k,
            query_chunk_size=min(256, max(1, chunk_size)),
            bank_chunk_size=8192,
        )

        local_uncertainty = (1.0 - mean_topk_sim).clamp_min(0.0)
        scale = 1.0 + local_beta * local_uncertainty

        return base_var.unsqueeze(0) * scale.unsqueeze(1)

    if posterior_proxy == "class":
        centroids, class_vars = _compute_class_centroids_and_vars(
            bank_embeddings=bank_embeddings,
            bank_labels=bank_labels,
            num_classes=num_classes,
            global_var=global_var,
            var_floor=var_floor,
        )

        pred = query_clip_preds.clamp(min=0, max=num_classes - 1)
        pred_centroids = centroids[pred]
        pred_class_vars = class_vars[pred]

        sim_to_pred_class = (query_embeddings * pred_centroids).sum(dim=1)
        class_uncertainty = (1.0 - sim_to_pred_class).clamp_min(0.0)
        scale = 1.0 + local_beta * class_uncertainty

        return image_var_scale * pred_class_vars * scale.unsqueeze(1)

    raise ValueError(
        f"Unknown posterior_proxy={posterior_proxy}. "
        "Use one of: global, local, class."
    )


@torch.no_grad()
def _estimate_text_embedding_vars(
    text_features: torch.Tensor,
    text_var_scale: float,
    var_floor: float,
) -> torch.Tensor:
    """
    Proxy for BayesVLM diagonal text-embedding variance.

    Full BayesVLM would compute:
        Sigma_h = (psi(x)^T A_txt^{-1} psi(x)) B_txt^{-1}

    With only one text embedding per class, this uses variance across class
    text embeddings as a weak diagonal proxy.
    """
    if text_var_scale <= 0:
        return torch.full_like(text_features, fill_value=var_floor)

    text_dim_var = torch.var(
        text_features,
        dim=0,
        unbiased=False,
    ).clamp_min(var_floor)

    return text_var_scale * text_dim_var.unsqueeze(0).repeat(
        text_features.shape[0],
        1,
    )


# ------------------------------------------------------------
# BayesVLM scoring
# ------------------------------------------------------------

@torch.no_grad()
def compute_bayesvlm_scores(
    bank_embeddings: torch.Tensor,
    bank_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    query_clip_preds: torch.Tensor,
    query_clip_pred_probs: torch.Tensor,
    text_features: torch.Tensor,
    class_names: List[str],
    clip_logit_scale: float,
    clip_logit_bias: float = 0.0,
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
    score_type: str = "entropy",
    normalize_entropy: bool = True,
    image_var_scale: float = 0.05,
    text_var_scale: float = 0.01,
    var_floor: float = 1e-6,
    posterior_proxy: str = "global",
    local_k: int = 20,
    local_beta: float = 5.0,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """
    BayesVLM-style failure scoring for your current CLIP pipeline.

    Main idea:
        1. Treat image/text embeddings as Gaussian random variables.
        2. Approximate expected cosine similarity using ProbCosine.
        3. Approximate cosine variance.
        4. Apply the BayesVLM probit approximation to obtain calibrated logits.
        5. Convert logits into class probabilities.
        6. Use predictive entropy, 1 - max probability, or 1 - BayesVLM
           probability of CLIP's predicted class as the failure score.

    Notes:
        - This is a drop-in version for your current script.
        - The full paper-faithful version requires projection-layer posterior
          covariance from KFAC/Laplace or the official BayesVLM model outputs.
    """
    if device is None:
        device = query_embeddings.device

    bank_embeddings = F.normalize(
        bank_embeddings.to(device).float(),
        dim=-1,
    )
    query_embeddings = F.normalize(
        query_embeddings.to(device).float(),
        dim=-1,
    )
    text_features = F.normalize(
        text_features.to(device).float(),
        dim=-1,
    )

    bank_labels = bank_labels.to(device).long()
    query_labels = query_labels.to(device).long()
    query_clip_preds = query_clip_preds.to(device).long()
    query_clip_pred_probs = query_clip_pred_probs.to(device).float()

    num_query = query_embeddings.shape[0]
    num_classes = text_features.shape[0]
    eps = 1e-12

    if score_type not in ["entropy", "msp", "clip_pred_prob"]:
        raise ValueError(
            f"Unknown score_type={score_type}. "
            "Use one of: entropy, msp, clip_pred_prob."
        )

    query_vars = _estimate_query_embedding_vars(
        bank_embeddings=bank_embeddings,
        bank_labels=bank_labels,
        query_embeddings=query_embeddings,
        query_clip_preds=query_clip_preds,
        num_classes=num_classes,
        image_var_scale=image_var_scale,
        var_floor=var_floor,
        posterior_proxy=posterior_proxy,
        local_k=local_k,
        local_beta=local_beta,
        chunk_size=chunk_size,
    )

    text_vars = _estimate_text_embedding_vars(
        text_features=text_features,
        text_var_scale=text_var_scale,
        var_floor=var_floor,
    )

    text_mu = text_features
    text_var = text_vars.clamp_min(var_floor)

    text_second_moment = text_mu.square() + text_var
    text_norm2 = text_second_moment.sum(dim=1).clamp_min(eps)

    failure_scores_all = []
    entropy_all = []
    max_prob_all = []
    bayes_pred_all = []
    clip_pred_bayes_prob_all = []
    clip_pred_expected_cos_all = []
    clip_pred_cos_var_all = []

    for start in range(0, num_query, chunk_size):
        end = min(start + chunk_size, num_query)

        q_mu = query_embeddings[start:end]
        q_var = query_vars[start:end].clamp_min(var_floor)

        q_second_moment = q_mu.square() + q_var
        q_norm2 = q_second_moment.sum(dim=1).clamp_min(eps)

        # Eq. (8)-style expected cosine.
        numerator_mean = q_mu @ text_mu.T
        denominator_mean = torch.sqrt(
            q_norm2.unsqueeze(1) * text_norm2.unsqueeze(0)
        ).clamp_min(eps)

        expected_cos = numerator_mean / denominator_mean

        # Eq. (9)-style cosine variance using diagonal covariances.
        numerator_var = (
            q_var @ (text_var + text_mu.square()).T
            + q_mu.square() @ text_var.T
        )

        denominator_var = (
            q_norm2.unsqueeze(1) * text_norm2.unsqueeze(0)
        ).clamp_min(eps)

        cos_var = (numerator_var / denominator_var).clamp_min(0.0)

        # BayesVLM probit approximation:
        # logits = t * E[cos] / sqrt(1 + pi/8 * t^2 * Var[cos])
        # logit_mean = clip_logit_scale * expected_cos
        # logit_var = (clip_logit_scale ** 2) * cos_var

        # bayesvlm_logits = logit_mean / torch.sqrt(
        #     1.0 + (math.pi / 8.0) * logit_var
        # )

        # BayesVLM probit approximation:
        # logits = t * E[cos] + bias
        # Var[logits] = t^2 * Var[cos]
        logit_mean = clip_logit_scale * expected_cos + clip_logit_bias
        logit_var = (clip_logit_scale ** 2) * cos_var

        bayesvlm_logits = logit_mean / torch.sqrt(
            1.0 + (math.pi / 8.0) * logit_var
        )

        bayesvlm_probs = torch.softmax(bayesvlm_logits, dim=1)

        entropy = -torch.sum(
            bayesvlm_probs * torch.log(bayesvlm_probs.clamp_min(eps)),
            dim=1,
        )

        if normalize_entropy:
            entropy = entropy / math.log(num_classes)

        max_prob, bayes_pred = torch.max(bayesvlm_probs, dim=1)

        clip_pred_chunk = query_clip_preds[start:end]
        clip_pred_bayes_prob = bayesvlm_probs.gather(
            dim=1,
            index=clip_pred_chunk.unsqueeze(1),
        ).squeeze(1)

        clip_pred_expected_cos = expected_cos.gather(
            dim=1,
            index=clip_pred_chunk.unsqueeze(1),
        ).squeeze(1)

        clip_pred_cos_var = cos_var.gather(
            dim=1,
            index=clip_pred_chunk.unsqueeze(1),
        ).squeeze(1)

        if score_type == "entropy":
            failure_score = entropy
        elif score_type == "msp":
            failure_score = 1.0 - max_prob
        else:
            failure_score = 1.0 - clip_pred_bayes_prob

        failure_scores_all.append(failure_score.detach().cpu())
        entropy_all.append(entropy.detach().cpu())
        max_prob_all.append(max_prob.detach().cpu())
        bayes_pred_all.append(bayes_pred.detach().cpu())
        clip_pred_bayes_prob_all.append(clip_pred_bayes_prob.detach().cpu())
        clip_pred_expected_cos_all.append(clip_pred_expected_cos.detach().cpu())
        clip_pred_cos_var_all.append(clip_pred_cos_var.detach().cpu())

    failure_scores = torch.cat(failure_scores_all).numpy()
    entropy_scores = torch.cat(entropy_all).numpy()
    max_probs = torch.cat(max_prob_all).numpy()
    bayes_preds = torch.cat(bayes_pred_all).numpy()
    clip_pred_bayes_probs = torch.cat(clip_pred_bayes_prob_all).numpy()
    clip_pred_expected_cos = torch.cat(clip_pred_expected_cos_all).numpy()
    clip_pred_cos_var = torch.cat(clip_pred_cos_var_all).numpy()

    query_labels_np = query_labels.detach().cpu().numpy()
    query_clip_preds_np = query_clip_preds.detach().cpu().numpy()
    query_clip_pred_probs_np = query_clip_pred_probs.detach().cpu().numpy()

    clip_correct = query_clip_preds_np == query_labels_np
    wrong_as_positive = (~clip_correct).astype(np.int64)

    auroc = _rank_auc(wrong_as_positive, failure_scores)
    aupr = _average_precision(wrong_as_positive, failure_scores)
    fpr95 = _fpr_at_tpr(
        wrong_as_positive,
        failure_scores,
        target_tpr=target_tpr,
    )

    correct_scores = failure_scores[clip_correct]
    wrong_scores = failure_scores[~clip_correct]

    per_sample = []

    for i in range(num_query):
        true_label = int(query_labels_np[i])
        clip_pred = int(query_clip_preds_np[i])
        bayes_pred = int(bayes_preds[i])

        per_sample.append(
            {
                "index": int(i),
                "true_label": true_label,
                "true_class": class_names[true_label],
                "clip_pred": clip_pred,
                "clip_pred_class": class_names[clip_pred],
                "clip_correct": bool(clip_correct[i]),
                "clip_pred_prob": float(query_clip_pred_probs_np[i]),
                "bayesvlm_pred": bayes_pred,
                "bayesvlm_pred_class": class_names[bayes_pred],
                "failure_score": float(failure_scores[i]),
                "bayesvlm_entropy": float(entropy_scores[i]),
                "bayesvlm_max_prob": float(max_probs[i]),
                "bayesvlm_clip_pred_prob": float(clip_pred_bayes_probs[i]),
                "bayesvlm_expected_cos_clip_pred": float(
                    clip_pred_expected_cos[i]
                ),
                "bayesvlm_cos_var_clip_pred": float(clip_pred_cos_var[i]),
            }
        )

    out = {
        "summary": {
            "correct_clip_predictions": _summary(correct_scores),
            "wrong_clip_predictions": _summary(wrong_scores),
        },
        "metrics": {
            "num_samples": int(num_query),
            "num_correct": int(clip_correct.sum()),
            "num_wrong": int((~clip_correct).sum()),
            "wrong_rate": float((~clip_correct).mean()),
            "auroc_wrong_positive": auroc,
            "aupr_wrong_positive": aupr,
            "fpr_at_95_tpr_wrong_positive": fpr95,
            "target_tpr": float(target_tpr),
            "bayesvlm_score_type": score_type,
            "bayesvlm_posterior_proxy": posterior_proxy,
            "bayesvlm_image_var_scale": float(image_var_scale),
            "bayesvlm_text_var_scale": float(text_var_scale),
            "bayesvlm_var_floor": float(var_floor),
            "bayesvlm_local_k": int(local_k),
            "bayesvlm_local_beta": float(local_beta),
        },
        "per_sample": per_sample,
    }

    return out