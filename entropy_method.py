# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# def _fpr_at_tpr(y_true, scores, target_tpr=0.95):
#     """
#     Compute FPR at target TPR.

#     y_true:
#         1 = wrong CLIP prediction
#         0 = correct CLIP prediction

#     scores:
#         Higher = more uncertain / more likely wrong.
#     """

#     fpr, tpr, _ = roc_curve(y_true, scores)
#     valid = np.where(tpr >= target_tpr)[0]

#     if len(valid) == 0:
#         return 1.0

#     return float(fpr[valid[0]])


# def _safe_metrics(y_true, scores, target_tpr=0.95):
#     y_true = np.asarray(y_true)
#     scores = np.asarray(scores)

#     if len(np.unique(y_true)) < 2:
#         return {
#             "auroc_wrong_positive": None,
#             "aupr_wrong_positive": None,
#             "fpr_at_95_tpr_wrong_positive": None,
#         }

#     return {
#         "auroc_wrong_positive": float(roc_auc_score(y_true, scores)),
#         "aupr_wrong_positive": float(average_precision_score(y_true, scores)),
#         "fpr_at_95_tpr_wrong_positive": _fpr_at_tpr(
#             y_true=y_true,
#             scores=scores,
#             target_tpr=target_tpr,
#         ),
#     }


# def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


# @torch.no_grad()
# def compute_entropy_scores(
#     query_embeddings: torch.Tensor,
#     query_labels: torch.Tensor,
#     query_clip_preds: torch.Tensor,
#     query_clip_pred_probs: torch.Tensor,
#     text_features: torch.Tensor,
#     class_names,
#     clip_logit_scale: float = 100.0,
#     chunk_size: int = 2048,
#     device=None,
#     normalize_entropy: bool = True,
#     print_debug: bool = True,
# ):
#     """
#     Entropy-based uncertainty score.

#     This follows the predictive-entropy uncertainty metric used in
#     Sensoy et al., "Evidential Deep Learning to Quantify Classification
#     Uncertainty", where prediction uncertainty increases with entropy
#     of the predictive class distribution.

#     For CLIP:
#         logits = logit_scale * image_features @ text_features.T
#         p = softmax(logits)
#         H(p) = -sum_k p_k log p_k

#     Final score:
#         failure_score = H(p) / log(K)

#     Higher score means more uncertain / more likely wrong.

#     Args:
#         query_embeddings:
#             CLIP image embeddings for query/test samples. Shape [N, D].

#         query_labels:
#             Ground-truth labels. Shape [N].

#         query_clip_preds:
#             Existing CLIP predicted labels from your extraction function. Shape [N].

#         query_clip_pred_probs:
#             Existing CLIP max softmax probabilities from your extraction function. Shape [N].

#         text_features:
#             CLIP text embeddings for class names. Shape [K, D].

#         class_names:
#             List of class names.

#         clip_logit_scale:
#             CLIP logit scale. For Hugging Face CLIP, use:
#                 float(model.logit_scale.exp().item())

#         normalize_entropy:
#             If True, divide entropy by log(K), so score lies approximately in [0, 1].
#     """

#     if device is None:
#         device = query_embeddings.device

#     num_classes = len(class_names)

#     q = _normalize(query_embeddings.to(device).float())
#     t = _normalize(text_features.to(device).float())

#     labels = query_labels.to(device).long()
#     preds = query_clip_preds.to(device).long()
#     clip_pred_probs = query_clip_pred_probs.to(device).float()

#     all_failure_scores = []
#     all_entropy_raw = []
#     all_entropy_preds = []
#     all_entropy_pred_probs = []

#     max_entropy = np.log(num_classes)

#     for start in range(0, q.shape[0], chunk_size):
#         end = min(start + chunk_size, q.shape[0])

#         q_chunk = q[start:end]

#         logits = clip_logit_scale * (q_chunk @ t.T)

#         probs = F.softmax(logits, dim=1)
#         log_probs = F.log_softmax(logits, dim=1)

#         entropy = -(probs * log_probs).sum(dim=1)

#         if normalize_entropy:
#             failure_score = entropy / max_entropy
#         else:
#             failure_score = entropy

#         pred_prob, pred = probs.max(dim=1)

#         all_failure_scores.append(failure_score.detach().cpu())
#         all_entropy_raw.append(entropy.detach().cpu())
#         all_entropy_preds.append(pred.detach().cpu())
#         all_entropy_pred_probs.append(pred_prob.detach().cpu())

#     failure_scores = torch.cat(all_failure_scores).numpy()
#     entropy_raw = torch.cat(all_entropy_raw).numpy()
#     entropy_preds = torch.cat(all_entropy_preds).numpy()
#     entropy_pred_probs = torch.cat(all_entropy_pred_probs).numpy()

#     labels_np = labels.detach().cpu().numpy()
#     preds_np = preds.detach().cpu().numpy()
#     clip_probs_np = clip_pred_probs.detach().cpu().numpy()

#     correct = preds_np == labels_np
#     wrong = ~correct
#     y_wrong = wrong.astype(int)

#     metrics = _safe_metrics(
#         y_true=y_wrong,
#         scores=failure_scores,
#         target_tpr=0.95,
#     )

#     if print_debug:
#         quantiles = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1]

#         print("\nEntropy uncertainty score quantiles")

#         if correct.any():
#             print("Correct:", np.quantile(failure_scores[correct], quantiles))
#         else:
#             print("Correct: no correct CLIP predictions")

#         if wrong.any():
#             print("Wrong:  ", np.quantile(failure_scores[wrong], quantiles))
#         else:
#             print("Wrong: no wrong CLIP predictions")

#         pred_mismatch = int(np.sum(entropy_preds != preds_np))
#         print("\nPrediction mismatch between recomputed entropy logits and stored CLIP preds:")
#         print(pred_mismatch, "out of", len(preds_np))

#     summary = {
#         "correct_clip_predictions": {
#             "count": int(correct.sum()),
#             "mean": float(np.mean(failure_scores[correct])) if correct.any() else None,
#             "std": float(np.std(failure_scores[correct])) if correct.any() else None,
#             "min": float(np.min(failure_scores[correct])) if correct.any() else None,
#             "max": float(np.max(failure_scores[correct])) if correct.any() else None,
#         },
#         "wrong_clip_predictions": {
#             "count": int(wrong.sum()),
#             "mean": float(np.mean(failure_scores[wrong])) if wrong.any() else None,
#             "std": float(np.std(failure_scores[wrong])) if wrong.any() else None,
#             "min": float(np.min(failure_scores[wrong])) if wrong.any() else None,
#             "max": float(np.max(failure_scores[wrong])) if wrong.any() else None,
#         },
#     }

#     per_sample = []

#     for i in range(len(labels_np)):
#         true_label = int(labels_np[i])
#         clip_pred = int(preds_np[i])
#         entropy_pred = int(entropy_preds[i])

#         per_sample.append(
#             {
#                 "index": int(i),

#                 "true_label": true_label,
#                 "true_class": class_names[true_label],

#                 "clip_pred": clip_pred,
#                 "clip_pred_class": class_names[clip_pred],
#                 "clip_pred_prob": float(clip_probs_np[i]),
#                 "clip_correct": bool(correct[i]),

#                 "entropy_failure_score": float(failure_scores[i]),
#                 "entropy_raw": float(entropy_raw[i]),
#                 "entropy_normalized": float(failure_scores[i]),

#                 "entropy_recomputed_pred": entropy_pred,
#                 "entropy_recomputed_pred_class": class_names[entropy_pred],
#                 "entropy_recomputed_pred_prob": float(entropy_pred_probs[i]),
#             }
#         )

#     return {
#         "summary": summary,
#         "metrics": metrics,
#         "per_sample": per_sample,
#     }


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def _to_float_tensor_scalar(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    return torch.tensor(float(x), device=device, dtype=torch.float32)



def _fpr_at_tpr(y_true, scores, target_tpr=0.95):
    """
    Compute FPR at target TPR.

    y_true:
        1 = wrong CLIP prediction
        0 = correct CLIP prediction

    scores:
        Higher = more uncertain / more likely wrong.
    """

    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(tpr >= target_tpr)[0]

    if len(valid) == 0:
        return 1.0

    return float(fpr[valid[0]])


def _safe_metrics(y_true, scores, target_tpr=0.95):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc_wrong_positive": None,
            "aupr_wrong_positive": None,
            "fpr_at_95_tpr_wrong_positive": None,
        }

    return {
        "auroc_wrong_positive": float(roc_auc_score(y_true, scores)),
        "aupr_wrong_positive": float(average_precision_score(y_true, scores)),
        "fpr_at_95_tpr_wrong_positive": _fpr_at_tpr(
            y_true=y_true,
            scores=scores,
            target_tpr=target_tpr,
        ),
    }


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x = x.clone()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def compute_entropy_scores(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    query_clip_preds: torch.Tensor,
    query_clip_pred_probs: torch.Tensor,
    text_features: torch.Tensor,
    class_names,
    clip_logit_scale: float = 100.0,
    clip_logit_bias: float = 0.0,
    chunk_size: int = 2048,
    device=None,
    normalize_entropy: bool = True,
    print_debug: bool = True,
):
    """
    Entropy-based uncertainty score for CLIP/SigLIP-style VLMs.

    For CLIP:
        logits = logit_scale * image_features @ text_features.T

    For SigLIP:
        logits = logit_scale * image_features @ text_features.T + logit_bias

    For closed-set single-label classification, we convert logits to a class
    distribution with softmax and compute predictive entropy:

        H(p) = -sum_k p_k log p_k

    Final score:
        failure_score = H(p) / log(K)

    Higher score means more uncertain / more likely wrong.
    """

    if device is None:
        device = query_embeddings.device

    num_classes = len(class_names)

    q = _normalize(query_embeddings.to(device).float())
    t = _normalize(text_features.to(device).float())

    labels = query_labels.to(device).long()
    preds = query_clip_preds.to(device).long()
    clip_pred_probs = query_clip_pred_probs.to(device).float()

    logit_scale = _to_float_tensor_scalar(clip_logit_scale, device)
    logit_bias = _to_float_tensor_scalar(clip_logit_bias, device)

    all_failure_scores = []
    all_entropy_raw = []
    all_entropy_preds = []
    all_entropy_pred_probs = []

    max_entropy = np.log(num_classes)

    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])

        q_chunk = q[start:end]

        logits = logit_scale * (q_chunk @ t.T) + logit_bias

        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        entropy = -(probs * log_probs).sum(dim=1)

        if normalize_entropy:
            failure_score = entropy / max_entropy
        else:
            failure_score = entropy

        pred_prob, pred = probs.max(dim=1)

        all_failure_scores.append(failure_score.detach().cpu())
        all_entropy_raw.append(entropy.detach().cpu())
        all_entropy_preds.append(pred.detach().cpu())
        all_entropy_pred_probs.append(pred_prob.detach().cpu())

    failure_scores = torch.cat(all_failure_scores).numpy()
    entropy_raw = torch.cat(all_entropy_raw).numpy()
    entropy_preds = torch.cat(all_entropy_preds).numpy()
    entropy_pred_probs = torch.cat(all_entropy_pred_probs).numpy()

    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    clip_probs_np = clip_pred_probs.detach().cpu().numpy()

    correct = preds_np == labels_np
    wrong = ~correct
    y_wrong = wrong.astype(int)

    metrics = _safe_metrics(
        y_true=y_wrong,
        scores=failure_scores,
        target_tpr=0.95,
    )

    if print_debug:
        quantiles = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1]

        print("\nEntropy uncertainty score quantiles")

        if correct.any():
            print("Correct:", np.quantile(failure_scores[correct], quantiles))
        else:
            print("Correct: no correct VLM predictions")

        if wrong.any():
            print("Wrong:  ", np.quantile(failure_scores[wrong], quantiles))
        else:
            print("Wrong: no wrong VLM predictions")

        pred_mismatch = int(np.sum(entropy_preds != preds_np))
        print("\nPrediction mismatch between recomputed entropy logits and stored VLM preds:")
        print(pred_mismatch, "out of", len(preds_np))

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
        entropy_pred = int(entropy_preds[i])

        per_sample.append(
            {
                "index": int(i),

                "true_label": true_label,
                "true_class": class_names[true_label],

                "clip_pred": clip_pred,
                "clip_pred_class": class_names[clip_pred],
                "clip_pred_prob": float(clip_probs_np[i]),
                "clip_correct": bool(correct[i]),

                "entropy_failure_score": float(failure_scores[i]),
                "entropy_raw": float(entropy_raw[i]),
                "entropy_normalized": float(failure_scores[i]),

                "entropy_recomputed_pred": entropy_pred,
                "entropy_recomputed_pred_class": class_names[entropy_pred],
                "entropy_recomputed_pred_prob": float(entropy_pred_probs[i]),
            }
        )

    return {
        "summary": summary,
        "metrics": metrics,
        "per_sample": per_sample,
    }