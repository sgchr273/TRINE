import os
import json
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def _fpr_at_tpr(y_true, scores, target_tpr=0.95):
    """
    Compute FPR at target TPR.

    y_true:
        1 = wrong CLIP prediction
        0 = correct CLIP prediction

    scores:
        Higher = more suspicious / more likely wrong.
    """

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    valid = np.where(tpr >= target_tpr)[0]

    if len(valid) == 0:
        return 1.0

    return float(fpr[valid[0]])


def _fpr_at_tpr_debug(y_true, scores, target_tpr=0.95):
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    valid = np.where(tpr >= target_tpr)[0]

    if len(valid) == 0:
        return 1.0, None, None

    idx = valid[0]

    return float(fpr[idx]), float(thresholds[idx]), float(tpr[idx])


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def fit_icpe_gaussians(
    bank_embeddings: torch.Tensor,
    bank_labels: torch.Tensor,
    num_classes: int,
    projection_dim: int = 128,
    covariance_reg: float = 1e-4,
    min_samples_per_class: int = 2,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Fit class-specific Gaussian distributions on CLIP image embeddings.

    ICPE-style procedure:
      1. Normalize image embeddings.
      2. Compute a PCA projection from the bank features.
      3. Project bank embeddings into the lower-dimensional subspace.
      4. Fit one multivariate Gaussian per class.
    """

    if device is None:
        device = bank_embeddings.device

    # Use float64 here to reduce numerical saturation in Gaussian densities.
    x = _normalize(bank_embeddings.to(device).double())
    y = bank_labels.to(device).long()

    n, d = x.shape
    projection_dim = min(projection_dim, d, n - 1)

    global_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - global_mean

    # Keep the original SVD-based projection.
    # This matches your previous implementation more closely than changing to covariance PCA.
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    projection = vh[:projection_dim].T.contiguous()

    z = x_centered @ projection

    means = torch.zeros(num_classes, projection_dim, device=device, dtype=torch.float64)
    inv_covs = torch.zeros(
        num_classes,
        projection_dim,
        projection_dim,
        device=device,
        dtype=torch.float64,
    )
    logdets = torch.zeros(num_classes, device=device, dtype=torch.float64)
    valid_classes = torch.zeros(num_classes, dtype=torch.bool, device=device)

    eye = torch.eye(projection_dim, device=device, dtype=torch.float64)

    for c in range(num_classes):
        idx = y == c
        zc = z[idx]

        if zc.shape[0] < min_samples_per_class:
            continue

        mu = zc.mean(dim=0)
        centered = zc - mu

        cov = centered.T @ centered / max(zc.shape[0] - 1, 1)
        cov = cov + covariance_reg * eye

        sign, logdet = torch.linalg.slogdet(cov)

        if sign <= 0:
            cov = cov + (10.0 * covariance_reg) * eye
            sign, logdet = torch.linalg.slogdet(cov)

        if sign <= 0:
            continue

        inv_cov = torch.linalg.inv(cov)

        means[c] = mu
        inv_covs[c] = inv_cov
        logdets[c] = logdet
        valid_classes[c] = True

    return {
        "global_mean": global_mean,
        "projection": projection,
        "means": means,
        "inv_covs": inv_covs,
        "logdets": logdets,
        "valid_classes": valid_classes,
        "projection_dim": projection_dim,
        "covariance_reg": covariance_reg,
    }


@torch.no_grad()
def compute_icpe_scores_train_bank_test_query(
    bank_embeddings: torch.Tensor,
    bank_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    query_clip_preds: torch.Tensor,
    query_clip_pred_probs: torch.Tensor,
    class_names: List[str],
    projection_dim: int = 128,
    covariance_reg: float = 1e-4,
    density_temperature: float = 1.0,
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
    print_debug: bool = True,
):
    """
    ICPE-style failure scoring.

    This version keeps the original ICPE idea:

        low density/posterior support for the CLIP-predicted class
        means higher uncertainty.

    However, instead of using the saturated bounded score

        1 - softmax(log_probs)[clip_pred]

    for metrics, it uses the equivalent logit-margin form:

        logsumexp(log_probs of non-predicted classes)
        - log_prob(CLIP-predicted class)

    This avoids the 0/1 score collapse that can make FPR@95TPR equal to 1.
    Larger score means more suspicious.
    """

    if device is None:
        device = query_embeddings.device

    num_classes = len(class_names)

    model = fit_icpe_gaussians(
        bank_embeddings=bank_embeddings,
        bank_labels=bank_labels,
        num_classes=num_classes,
        projection_dim=projection_dim,
        covariance_reg=covariance_reg,
        device=device,
    )

    q = _normalize(query_embeddings.to(device).double())
    q_labels = query_labels.to(device).long()
    q_preds = query_clip_preds.to(device).long()
    q_clip_probs = query_clip_pred_probs.to(device).float()

    global_mean = model["global_mean"]
    projection = model["projection"]
    means = model["means"]
    inv_covs = model["inv_covs"]
    logdets = model["logdets"]
    valid_classes = model["valid_classes"]

    qz = (q - global_mean) @ projection

    all_failure_scores = []
    all_density_confidences = []
    all_pred_log_probs = []
    all_density_preds = []
    all_density_pred_probs = []
    all_pmax_scores = []

    d = model["projection_dim"]
    log_2pi = np.log(2.0 * np.pi)

    for start in range(0, qz.shape[0], chunk_size):
        end = min(start + chunk_size, qz.shape[0])
        z = qz[start:end]

        class_log_probs = []

        for c in range(num_classes):
            if not valid_classes[c]:
                lp = torch.full(
                    (z.shape[0],),
                    -1e12,
                    device=device,
                    dtype=torch.float64,
                )
            else:
                diff = z - means[c]
                mahal = torch.sum((diff @ inv_covs[c]) * diff, dim=1)
                lp = -0.5 * (mahal + logdets[c] + d * log_2pi)

            class_log_probs.append(lp)

        log_probs = torch.stack(class_log_probs, dim=1)  # [B, C]

        preds_chunk = q_preds[start:end]

        pred_log_prob = log_probs.gather(
            1,
            preds_chunk.view(-1, 1),
        ).squeeze(1)

        # ------------------------------------------------------------
        # Main fix:
        # use logit-margin failure score instead of saturated 1 - softmax.
        #
        # failure_score = logsumexp(other classes) - pred_log_prob
        #
        # This is the logit of:
        #     1 - softmax(log_probs)[predicted class]
        #
        # but does not collapse to exactly 0 or 1.
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Paper-exact ICPE scoring
        #
        # 1. Softmax-normalize Gaussian log-likelihoods across classes.
        # 2. Extract s_d for the CLIP-predicted class.
        # 3. Combine s_d with CLIP p_max:
        #
        #       s_unc = 1 - (p_max + s_d) / 2
        #
        # Higher s_unc = more uncertain / more likely wrong.
        # ------------------------------------------------------------

        density_probs = F.softmax(log_probs / density_temperature, dim=1)

        # s_d: normalized intra-class density score for CLIP-predicted class
        s_d = density_probs.gather(
            1,
            preds_chunk.view(-1, 1),
        ).squeeze(1)

        # p_max: CLIP image-text maximum softmax probability
        p_max = q_clip_probs[start:end].to(s_d.dtype)

        # Final paper uncertainty score
        failure_score = 1.0 - 0.5 * (p_max + s_d)

        density_pred_prob, density_pred = density_probs.max(dim=1)

        all_failure_scores.append(failure_score.detach().cpu())
        all_density_confidences.append(s_d.detach().cpu())
        all_pred_log_probs.append(pred_log_prob.detach().cpu())
        all_density_preds.append(density_pred.detach().cpu())
        all_density_pred_probs.append(density_pred_prob.detach().cpu())
        all_pmax_scores.append(p_max.detach().cpu())

    failure_scores = torch.cat(all_failure_scores).numpy()
    density_confidences = torch.cat(all_density_confidences).numpy()
    pred_log_probs = torch.cat(all_pred_log_probs).numpy()
    density_preds = torch.cat(all_density_preds).numpy()
    density_pred_probs = torch.cat(all_density_pred_probs).numpy()
    pmax_scores = torch.cat(all_pmax_scores).numpy()

    labels_np = q_labels.detach().cpu().numpy()
    preds_np = q_preds.detach().cpu().numpy()
    clip_probs_np = q_clip_probs.detach().cpu().numpy()

    correct = preds_np == labels_np
    wrong = ~correct
    y_wrong = wrong.astype(int)

    metrics = {
        "auroc_wrong_positive": float(roc_auc_score(y_wrong, failure_scores)),
        "aupr_wrong_positive": float(average_precision_score(y_wrong, failure_scores)),
        "fpr_at_95_tpr_wrong_positive": _fpr_at_tpr(
            y_true=y_wrong,
            scores=failure_scores,
            target_tpr=0.95,
        ),
    }

    fpr95, threshold95, actual_tpr = _fpr_at_tpr_debug(
        y_true=y_wrong,
        scores=failure_scores,
        target_tpr=0.95,
    )

    metrics["fpr_at_95_tpr_threshold"] = threshold95
    metrics["actual_tpr_at_threshold"] = actual_tpr

    if print_debug:
        quantiles = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1]

        print("\nPaper-exact ICPE uncertainty score quantiles")
        print("Correct:", np.quantile(failure_scores[correct], quantiles))
        print("Wrong:  ", np.quantile(failure_scores[wrong], quantiles))

        print("\nIntra-class density score s_d quantiles")
        print("Correct:", np.quantile(density_confidences[correct], quantiles))
        print("Wrong:  ", np.quantile(density_confidences[wrong], quantiles))

        print("\nCLIP p_max quantiles")
        print("Correct:", np.quantile(pmax_scores[correct], quantiles))
        print("Wrong:  ", np.quantile(pmax_scores[wrong], quantiles))

        print("\nFPR@95TPR debug")
        print("Threshold:", threshold95)
        print("Actual TPR:", actual_tpr)
        print("FPR:", fpr95)

    summary = {
        "correct_clip_predictions": {
            "count": int(correct.sum()),
            "mean": float(np.mean(failure_scores[correct])) if correct.any() else None,
            "std": float(np.std(failure_scores[correct])) if correct.any() else None,
            "min": float(np.min(failure_scores[correct])) if correct.any() else None,
            "max": float(np.max(failure_scores[correct])) if correct.any() else None,
        },
        "wrong_clip_predictions": {
            "count": int(wrong.sum()),
            "mean": float(np.mean(failure_scores[wrong])) if wrong.any() else None,
            "std": float(np.std(failure_scores[wrong])) if wrong.any() else None,
            "min": float(np.min(failure_scores[wrong])) if wrong.any() else None,
            "max": float(np.max(failure_scores[wrong])) if wrong.any() else None,
        },
    }

    per_sample = []

    for i in range(len(labels_np)):
        true_label = int(labels_np[i])
        clip_pred = int(preds_np[i])
        density_pred_i = int(density_preds[i])

        per_sample.append(
            {
                "index": int(i),

                "true_label": true_label,
                "true_class": class_names[true_label],

                "clip_pred": clip_pred,
                "clip_pred_class": class_names[clip_pred],
                "clip_pred_prob": float(clip_probs_np[i]),
                "clip_correct": bool(correct[i]),

                # Main metric score.
                "icpe_failure_score": float(failure_scores[i]),
                "icpe_pmax": float(pmax_scores[i]),
                "icpe_sd": float(density_confidences[i]),
                "icpe_pred_log_prob": float(pred_log_probs[i]),
                "icpe_density_confidence_for_clip_pred": float(density_confidences[i]),
                "icpe_density_pred": density_pred_i,
                "icpe_density_pred_class": class_names[density_pred_i],
                "icpe_density_pred_prob": float(density_pred_probs[i]),
            }
        )

    return {
        "summary": summary,
        "metrics": metrics,
        "per_sample": per_sample,
    }


def save_icpe_results(results, output_json):
    output_dir = os.path.dirname(output_json)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)