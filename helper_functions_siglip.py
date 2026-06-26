import os
import json
import argparse
from collections import defaultdict
from typing import List

import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def infer_class_names(dataset):
    """
    Works with the new data_helpers datasets.
    """
    if hasattr(dataset, "classes"):
        return list(dataset.classes)

    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return list(dataset.dataset.classes)

    raise AttributeError(
        "Could not infer class names. Dataset must have .classes attribute."
    )


def tensor_collate_fn(batch):
    """
    New data_helpers returns transformed tensors already:
        image: [3, 224, 224]
        label: int

    So we stack tensors directly and do NOT call CLIPProcessor on images.
    """
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def summarize(values):
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

    arr = np.asarray(values, dtype=np.float64)

    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_binary_auroc(scores, labels_pos):
    """
    labels_pos:
        1 = positive = wrong CLIP prediction
        0 = negative = correct CLIP prediction
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels_pos = np.asarray(labels_pos, dtype=np.int64)

    P = int((labels_pos == 1).sum())
    N = int((labels_pos == 0).sum())

    if P == 0 or N == 0:
        return None

    order = np.argsort(-scores)
    y = labels_pos[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    tpr = tp / P
    fpr = fp / N

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = np.trapz(tpr, fpr)
    return float(auc)


def compute_fpr_at_95_tpr(scores, labels_pos):
    scores = np.asarray(scores, dtype=np.float64)
    labels_pos = np.asarray(labels_pos, dtype=np.int64)

    P = int((labels_pos == 1).sum())
    N = int((labels_pos == 0).sum())

    if P == 0 or N == 0:
        return None

    order = np.argsort(-scores)
    y = labels_pos[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    tpr = tp / P
    fpr = fp / N

    valid = np.where(tpr >= 0.95)[0]

    if len(valid) == 0:
        return None

    return float(np.min(fpr[valid]))


# ------------------------------------------------------------
# VLM feature extraction: CLIP / SigLIP
# ------------------------------------------------------------

import torch.nn.functional as F


def _normalize_features(x):
    """L2-normalize image/text features for cosine-similarity scoring."""
    return F.normalize(x, dim=-1)


def _get_logit_scale_and_bias(model, device):
    """
    Return model-specific logit scale and bias.

    CLIP usually uses:
        logits = exp(logit_scale) * image_features @ text_features.T

    SigLIP usually uses:
        logits = exp(logit_scale) * image_features @ text_features.T + logit_bias

    This keeps the same downstream keys as your CLIP pipeline, so the rest
    of the code does not need to change.
    """
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp().to(device)
    else:
        logit_scale = torch.tensor(1.0, device=device)

    if hasattr(model, "logit_bias"):
        logit_bias = model.logit_bias.to(device)
    else:
        logit_bias = torch.tensor(0.0, device=device)

    return logit_scale, logit_bias


@torch.no_grad()
def build_text_bank(
    model,
    processor,
    class_names: List[str],
    prompt_template: str,
    device,
):
    prompts = [prompt_template.format(c.replace("_", " ")) for c in class_names]

    # SigLIP expects max-length padding. This also works for CLIP.
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # Works for both CLIPModel and SiglipModel from Hugging Face.
    text_features = model.get_text_features(**text_inputs)
    text_features = _normalize_features(text_features)

    return text_features


@torch.no_grad()
def extract_bank_embeddings(
    dataset,
    model,
    batch_size,
    num_workers,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tensor_collate_fn,
        pin_memory=True,
    )

    all_embeds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting bank embeddings"):
        pixel_values = images.to(device, non_blocking=True)

        # Works for both CLIP and SigLIP if the dataset transform matches the model.
        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = _normalize_features(image_features)

        all_embeds.append(image_features.cpu())
        all_labels.append(labels.cpu())

    return {
        "embeddings": torch.cat(all_embeds, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


@torch.no_grad()
def extract_query_embeddings_and_clip_preds(
    dataset,
    model,
    text_bank,
    batch_size,
    num_workers,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tensor_collate_fn,
        pin_memory=True,
    )

    all_embeds = []
    all_labels = []
    all_clip_preds = []
    all_clip_pred_probs = []
    all_clip_pred2 = []
    all_clip_pred2_probs = []
    all_clip_margins = []

    logit_scale, logit_bias = _get_logit_scale_and_bias(model, device)

    for images, labels in tqdm(loader, desc="Extracting query embeddings and VLM predictions"):
        pixel_values = images.to(device, non_blocking=True)

        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = _normalize_features(image_features)

        logits = logit_scale * (image_features @ text_bank.t()) + logit_bias

        # Your downstream code needs a normalized class distribution for top-1,
        # top-2, and margins. Softmax keeps this behavior consistent with CLIP.
        probs = torch.softmax(logits, dim=-1)

        top2_probs, top2_idx = probs.topk(k=2, dim=-1)

        preds = top2_idx[:, 0]
        pred2 = top2_idx[:, 1]

        pred_probs = top2_probs[:, 0]
        pred2_probs = top2_probs[:, 1]
        margins = pred_probs - pred2_probs

        all_embeds.append(image_features.cpu())
        all_labels.append(labels.cpu())
        all_clip_preds.append(preds.cpu())
        all_clip_pred_probs.append(pred_probs.cpu())
        all_clip_pred2.append(pred2.cpu())
        all_clip_pred2_probs.append(pred2_probs.cpu())
        all_clip_margins.append(margins.cpu())

    return {
        "embeddings": torch.cat(all_embeds, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "clip_preds": torch.cat(all_clip_preds, dim=0),
        "clip_pred_probs": torch.cat(all_clip_pred_probs, dim=0),
        "clip_pred2": torch.cat(all_clip_pred2, dim=0),
        "clip_pred2_probs": torch.cat(all_clip_pred2_probs, dim=0),
        "clip_margins": torch.cat(all_clip_margins, dim=0),
    }


# ------------------------------------------------------------
# kNN failure scoring
# ------------------------------------------------------------

@torch.no_grad()
def compute_knn_scores_train_bank_test_query(
    bank_embeddings,
    bank_labels,
    query_embeddings,
    query_labels,
    query_clip_preds,
    query_clip_pred_probs,
    query_clip_pred2,
    query_clip_pred2_probs,
    query_clip_margins,
    text_features,
    class_names,
    k=20,
    chunk_size=2048,
    rank_decay_alpha=0.1,
):
    """
    Multimodal kNN failure scoring.

    Idea:
    1. Retrieve top-k bank images using image-image similarity.
    2. For each retrieved neighbor, compute text similarity between:
         - the query's CLIP-predicted class text embedding
         - the neighbor's class text embedding
    3. Combine image similarity and this text similarity into a joint neighbor weight.
    4. Aggregate joint weights per class.
    5. Failure score = 1 - support assigned to CLIP predicted class.

    Returns larger failure_score for more suspicious CLIP predictions.
    """

    device = query_embeddings.device
    Nq = query_embeddings.shape[0]
    Nb = bank_embeddings.shape[0]
    C = len(class_names)

    if k > Nb:
        raise ValueError(f"k={k} is larger than bank size Nb={Nb}")

    records = []
    scores = []
    wrong_as_positive = []
    correct_scores = []
    wrong_scores = []

    for start in tqdm(range(0, Nq, chunk_size), desc="Computing query-to-bank kNN scores"):
        end = min(start + chunk_size, Nq)
        B = end - start

        q = query_embeddings[start:end]                 # [B, d]
        clip_preds = query_clip_preds[start:end]        # [B]
        true_labels = query_labels[start:end]           # [B]
        pred_probs = query_clip_pred_probs[start:end]   # [B]
        pred2 = query_clip_pred2[start:end]             # [B]
        pred2_probs = query_clip_pred2_probs[start:end] # [B]
        margins = query_clip_margins[start:end]         # [B]

        # -------------------------------------------------
        # 1. Top-k retrieval in image space
        # -------------------------------------------------
        sims = q @ bank_embeddings.t()                  # [B, Nb]
        topk_sims, topk_idx = sims.topk(k=k, dim=1, largest=True, sorted=True)
        neighbor_labels = bank_labels[topk_idx]         # [B, k]


        rank_weights = torch.exp(
            -torch.arange(k, dtype=torch.float32, device=device) * float(rank_decay_alpha)
        )

        rank_weights = rank_weights / rank_weights.sum()              # [k]
        rank_weights = rank_weights.unsqueeze(0).expand(B, -1)        # [B, k]


        # map cosine-style sims from [-1, 1] to [0, 1]
        image_weights = (topk_sims + 1.0) * 0.5         # [B, k]

        # -------------------------------------------------
        # 2. Text similarity:
        #    query predicted text  vs  each neighbor class text
        # -------------------------------------------------
        pred_text_features = text_features[clip_preds]              # [B, d]
        neighbor_text_features = text_features[neighbor_labels]     # [B, k, d]

        # cosine similarity via dot product in normalized CLIP space
        text_neighbor_sims = (
            neighbor_text_features * pred_text_features.unsqueeze(1)
        ).sum(dim=2)                                                # [B, k]

        # map from [-1, 1] to [0, 1]
        text_weights = (text_neighbor_sims + 1.0) * 0.5             # [B, k]

        # -------------------------------------------------
        # 3. Aggregate cross-modal evidence per neighbor
        # -------------------------------------------------
        # joint_weights = image_weights * text_weights                # [B, k]
        
        joint_weights = image_weights*rank_weights * text_weights               # [B, k]

        # pred_text_features = text_features[clip_preds]              # [B, d]
        # neighbor_image_features = bank_embeddings[topk_idx]         # [B, k, d]

        # # cosine similarity because both CLIP image and text embeddings are normalized
        # image_text_sims = (
        #     neighbor_image_features * pred_text_features.unsqueeze(1)
        # ).sum(dim=2)                                                # [B, k]

        # # map from [-1, 1] to [0, 1]
        # image_text_weights = (image_text_sims + 1.0) * 0.5          # [B, k]

        # # -------------------------------------------------
        # # 3. Aggregate image-image and image-text evidence
        # # -------------------------------------------------
        # joint_weights = image_weights * image_text_weights          # [B, k]

        class_count = torch.zeros(B, C, dtype=torch.float32, device=device)
        weighted_class_count = torch.zeros(B, C, dtype=torch.float32, device=device)
        joint_class_count = torch.zeros(B, C, dtype=torch.float32, device=device)

        ones = torch.ones_like(neighbor_labels, dtype=torch.float32)

        class_count.scatter_add_(1, neighbor_labels, ones)
        weighted_class_count.scatter_add_(1, neighbor_labels, image_weights)
        joint_class_count.scatter_add_(1, neighbor_labels, joint_weights)

        knn_pred = class_count.argmax(dim=1)
        knn_weighted_pred = weighted_class_count.argmax(dim=1)
        knn_joint_pred = joint_class_count.argmax(dim=1)

        # -------------------------------------------------
        # 4. Standard image-only support
        # -------------------------------------------------
        support_for_clip_pred = (
            class_count.gather(1, clip_preds[:, None]).squeeze(1) / float(k)
        )

        total_image_weight = weighted_class_count.sum(dim=1).clamp(min=1e-12)

        weighted_support_for_clip_pred = (
            weighted_class_count.gather(1, clip_preds[:, None]).squeeze(1)
            / total_image_weight
        )

        weighted_counts_copy = weighted_class_count.clone()
        weighted_counts_copy[torch.arange(B, device=device), clip_preds] = -1e9

        best_other_support = (
            weighted_counts_copy.max(dim=1).values / total_image_weight
        )

        weighted_support_margin = (
            weighted_support_for_clip_pred - best_other_support
        )

        # -------------------------------- -----------------
        # 5. New multimodal support based on:
        #    image sim(query, neighbor) × text sim(pred_text, neighbor_text)
        # -------------------------------------------------
        total_joint_weight = joint_class_count.sum(dim=1).clamp(min=1e-12)

        joint_support_for_clip_pred = (
            joint_class_count.gather(1, clip_preds[:, None]).squeeze(1)
            / total_joint_weight
        )

        joint_counts_copy = joint_class_count.clone()
        joint_counts_copy[torch.arange(B, device=device), clip_preds] = -1e9

        joint_best_other_support = (
            joint_counts_copy.max(dim=1).values / total_joint_weight
        )

        joint_support_margin = (
            joint_support_for_clip_pred - joint_best_other_support
        )

        # -------------------------------------------------
        # 6. Binary helper signals
        # -------------------------------------------------
        top2_counts, top2_classes = torch.topk(class_count, k=2, dim=1)
        clip_pred_in_top2 = (top2_classes == clip_preds[:, None]).any(dim=1)
        knn_match = (knn_pred == clip_preds)
        knn_joint_match = (knn_joint_pred == clip_preds)

        # Main failure score:
        # low multimodal support for CLIP class => high failure score
        # failure_score = 1.0 - joint_support_for_clip_pred
        failure_score = 1-joint_support_margin

        for i in range(B):
            idx_global = start + i

            true_idx = int(true_labels[i].item())
            pred_idx = int(clip_preds[i].item())
            pred2_idx = int(pred2[i].item())

            correct = pred_idx == true_idx
            score = float(failure_score[i].item())

            top1_neighbor_idx = int(top2_classes[i, 0].item())
            top2_neighbor_idx = int(top2_classes[i, 1].item())

            rec = {
                "sample_index": idx_global,

                "true_label_idx": true_idx,
                "true_label_name": class_names[true_idx],

                "clip_pred_idx": pred_idx,
                "clip_pred_name": class_names[pred_idx],
                "clip_pred2_idx": pred2_idx,
                "clip_pred2_name": class_names[pred2_idx],
                "clip_correct": bool(correct),
                "clip_pred_prob": float(pred_probs[i].item()),
                "clip_pred2_prob": float(pred2_probs[i].item()),
                "clip_top1_top2_margin": float(margins[i].item()),

                "knn_pred_idx": int(knn_pred[i].item()),
                "knn_pred_name": class_names[int(knn_pred[i].item())],

                "knn_weighted_pred_idx": int(knn_weighted_pred[i].item()),
                "knn_weighted_pred_name": class_names[int(knn_weighted_pred[i].item())],

                "knn_joint_pred_idx": int(knn_joint_pred[i].item()),
                "knn_joint_pred_name": class_names[int(knn_joint_pred[i].item())],

                "top1_neighbor_majority_idx": top1_neighbor_idx,
                "top1_neighbor_majority_name": class_names[top1_neighbor_idx],
                "top1_neighbor_majority_count": int(top2_counts[i, 0].item()),

                "top2_neighbor_majority_idx": top2_neighbor_idx,
                "top2_neighbor_majority_name": class_names[top2_neighbor_idx],
                "top2_neighbor_majority_count": int(top2_counts[i, 1].item()),

                "clip_pred_in_top2_neighbor_majority": bool(
                    clip_pred_in_top2[i].item()
                ),
                "clip_matches_knn_pred": bool(knn_match[i].item()),
                "clip_matches_knn_joint_pred": bool(knn_joint_match[i].item()),

                "knn_support_for_clip_pred": float(
                    support_for_clip_pred[i].item()
                ),
                "knn_weighted_support_for_clip_pred": float(
                    weighted_support_for_clip_pred[i].item()
                ),
                "knn_best_other_support": float(
                    best_other_support[i].item()
                ),
                "knn_weighted_support_margin": float(
                    weighted_support_margin[i].item()
                ),

                "knn_joint_support_for_clip_pred": float(
                    joint_support_for_clip_pred[i].item()
                ),
                "knn_joint_best_other_support": float(
                    joint_best_other_support[i].item()
                ),
                "knn_joint_support_margin": float(
                    joint_support_margin[i].item()
                ),

                "failure_score": score,
            }

            records.append(rec)
            scores.append(score)
            wrong_as_positive.append(0 if correct else 1)

            if correct:
                correct_scores.append(score)
            else:
                wrong_scores.append(score)

    auroc = compute_binary_auroc(scores, wrong_as_positive)
    fpr95 = compute_fpr_at_95_tpr(scores, wrong_as_positive)

    return {
        "per_sample": records,
        "scores": scores,
        "wrong_as_positive": wrong_as_positive,

        "score_distributions": {
            "correct_scores": correct_scores,
            "wrong_scores": wrong_scores,
        },

        "summary": {
            "correct_clip_predictions": summarize(correct_scores),
            "wrong_clip_predictions": summarize(wrong_scores),
        },

        "metrics": {
            "auroc_wrong_positive": auroc,
            "fpr_at_95_tpr_wrong_positive": fpr95,
        },
    }

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_true_clip_knn_relation_per_class(
    results,
    class_names,
    save_dir="./outputs/true_clip_knn_relation_per_class",
    use_weighted_knn=True,
    max_samples_per_class=None,
    dpi=300,
):
    os.makedirs(save_dir, exist_ok=True)

    records = results["per_sample"]
    C = len(class_names)

    class_to_records = defaultdict(list)

    for rec in records:
        class_to_records[rec["true_label_idx"]].append(rec)

    knn_key = "knn_weighted_pred_idx" if use_weighted_knn else "knn_pred_idx"
    knn_name = "Weighted kNN majority" if use_weighted_knn else "kNN majority"

    for true_class_idx in range(C):
        recs = class_to_records[true_class_idx]

        if len(recs) == 0:
            continue

        if max_samples_per_class is not None:
            recs = recs[:max_samples_per_class]

        true_labels = np.array([r["true_label_idx"] for r in recs])
        clip_preds = np.array([r["clip_pred_idx"] for r in recs])
        knn_preds = np.array([r[knn_key] for r in recs])
        clip_correct = np.array([r["clip_correct"] for r in recs])

        x = np.arange(len(recs))

        plt.figure(figsize=(16, 6))

        for c in range(C):
            plt.axhline(c, linewidth=0.5, alpha=0.25)

        plt.scatter(x, true_labels, marker="o", s=45, label="True label", alpha=0.9)
        plt.scatter(x, clip_preds, marker="x", s=55, label="CLIP predicted label", alpha=0.9)
        plt.scatter(x, knn_preds, marker="^", s=45, label=knn_name, alpha=0.9)

        wrong_positions = np.where(~clip_correct)[0]

        for wp in wrong_positions:
            plt.axvline(wp, linewidth=0.6, alpha=0.18)

        plt.yticks(np.arange(C), class_names)
        plt.xlabel("Samples within this true class")
        plt.ylabel("Class label")
        plt.title(
            f"True vs CLIP Prediction vs Neighbor Majority | True class: {class_names[true_class_idx]}"
        )

        plt.legend(loc="upper right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        filename = f"{class_names[true_class_idx]}_true_clip_knn_relation.png"
        filename = filename.replace(" ", "_").replace("/", "_")

        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")