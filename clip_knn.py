import os
import json
import argparse
from collections import defaultdict
from typing import List

# from ViLU.vilu.models.clip import model
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModel, AutoProcessor


from dataset_helpers_original import create_torchvision_dataset
# from helper_functions import (
#      infer_class_names, build_text_bank, extract_bank_embeddings, extract_query_embeddings_and_clip_preds,
# compute_knn_scores_train_bank_test_query, plot_true_clip_knn_relation_per_class
# )

from helper_functions_siglip import (
     infer_class_names, build_text_bank, extract_bank_embeddings, extract_query_embeddings_and_clip_preds,
compute_knn_scores_train_bank_test_query, plot_true_clip_knn_relation_per_class
)

from icpe_method import compute_icpe_scores_train_bank_test_query
from trustvlm_method import compute_trustvlm_scores
from mcm_method import compute_mcm_scores
from entropy_method import compute_entropy_scores
from bayesvlm_method import compute_bayesvlm_scores
from vilu_method import compute_vilu_scores


from knn_sweep import (
    sweep_k_values_for_knn_failure_detection,
    plot_k_sweep_results,
)

from rank_decay_sweep import (
    sweep_rank_decay_alpha_for_knn_failure_detection,
    plot_rank_decay_alpha_sweep_results,
)
from premise_validation_util import (
    run_premise_validation_analysis,
    save_tsne_bank_query_embedding_data,
    plot_pairwise_distance_distributions,
    plot_query_cluster_distance_correct_vs_wrong,
    plot_cluster_margin_correct_vs_wrong,
    plot_tsne_bank_query,
)


def get_vlm_logit_scale_and_bias(model, device):
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp().detach()
    else:
        logit_scale = torch.tensor(1.0, device=device)

    if hasattr(model, "logit_bias"):
        logit_bias = model.logit_bias.detach()
    else:
        logit_bias = torch.tensor(0.0, device=device)

    return logit_scale.to(device), logit_bias.to(device)

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

#  --root /media/4TB/sgchr/VLM_data
parser.add_argument("--dataset_name", type=str, default="fgvcaircraft")
parser.add_argument("--root", type=str, default="/home/sgchr/Documents/Failure_detection_Clip/data")
parser.add_argument("--bank_split", type=str, default="train")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--download", action="store_true")

# parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
# parser.add_argument("--prompt_template", type=str, default="a photo of a {}")

parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224")
parser.add_argument("--prompt_template", type=str, default="This is a photo of a {}.")


parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--k", type=int, default=20)
parser.add_argument("--chunk_size", type=int, default=2048)

parser.add_argument(
    "--score_mode",
    type=str,
    default="continuous",
    choices=["continuous", "top2_binary"],
    help=(
        "continuous: failure_score = 1 - weighted support for CLIP predicted class. "
        "top2_binary: failure_score = 1 if CLIP pred is not among top-2 neighbor majority classes."
    ),
)

parser.add_argument("--run_k_ablation", action="store_true")

parser.add_argument(
    "--k_ablation_values",
    type=int,
    nargs="+",
    default=[  10, 15, 20, 50,  100],
)

parser.add_argument(
    "--rank_decay_alpha",
    type=float,
    default=0.1,
    help="Coefficient alpha in rank_weights[r] = exp(-alpha * r). Default matches current method.",
)

parser.add_argument(
    "--run_rank_decay_ablation",
    action="store_true",
    help="Sweep rank-decay alpha values for the kNN failure detection method.",
)

parser.add_argument(
    "--rank_decay_alpha_values",
    type=float,
    nargs="+",
    default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    help="Values of alpha to sweep in rank_weights[r] = exp(-alpha * r).",
)


parser.add_argument("--run_premise_validation", action="store_true")
parser.add_argument("--run_premise_tsne", action="store_true")

parser.add_argument(
    "--method",
    type=str,
    nargs="+",
    default=["knn"],
    choices=["knn", "icpe", "trustvlm", "mcm", "entropy", "bayesvlm", "all"],
    help="Choose one or more methods: knn icpe trustvlm, or use all.",
)
parser.add_argument("--icpe_projection_dim", type=int, default=128)
parser.add_argument("--icpe_covariance_reg", type=float, default=1e-4)
parser.add_argument("--icpe_density_temperature", type=float, default=1.0)


parser.add_argument("--mcm_tau", type=float, default=1.0,
                    help="Softmax temperature for MCM. Paper default is 1.0. "
                         "Set to 0 for the no-softmax variant max_i cos(v,t_i).")

parser.add_argument("--trustvlm_variant", type=str, default="base", choices=["base","star"])
parser.add_argument("--trustvlm_n_shot", type=int, default=16)
parser.add_argument("--trustvlm_tau", type=float, default=0.01)


parser.add_argument(
    "--bayesvlm_score_type",
    type=str,
    default="clip_pred_prob",
    choices=["entropy", "msp", "clip_pred_prob"],
    help=(
        "BayesVLM failure score. "
        "entropy: predictive entropy. "
        "msp: 1 - max BayesVLM probability. "
        "clip_pred_prob: 1 - BayesVLM probability assigned to CLIP's predicted class."
    ),
)

parser.add_argument(
    "--bayesvlm_posterior_proxy",
    type=str,
    default="global",
    choices=["global", "local", "class"],
    help=(
        "Proxy for BayesVLM embedding variance when true Laplace/KFAC covariance "
        "is not available. global uses global bank variance. local scales variance "
        "by nearest-neighbor density. class uses the variance of CLIP-predicted class."
    ),
)

parser.add_argument("--bayesvlm_image_var_scale", type=float, default=0.05)
parser.add_argument("--bayesvlm_text_var_scale", type=float, default=0.01)
parser.add_argument("--bayesvlm_var_floor", type=float, default=1e-6)
parser.add_argument("--bayesvlm_local_k", type=int, default=20)
parser.add_argument("--bayesvlm_local_beta", type=float, default=5.0)

parser.add_argument(
    "--vilu_weights_path", type=str, default="./home/sgchr/Documents/ViLU/weights",
    help="Direct path to a .pt ViLU weights file for the current dataset."
)
parser.add_argument(
    "--vilu_weights_dir", type=str, default="./home/sgchr/Documents/ViLU/weights",
    help="Directory containing per-dataset .pt files named {dataset_name}.pt"
)


parser.add_argument("--premise_output_dir", type=str, default="./Plots/premise_validation")
parser.add_argument("--premise_max_intra_pairs_per_class", type=int, default=5000)
parser.add_argument("--premise_max_inter_pairs", type=int, default=100000)
parser.add_argument("--premise_tsne_max_bank", type=int, default=5000)
parser.add_argument("--premise_tsne_max_query", type=int, default=3000)

parser.add_argument("--output_json", type=str, default="./outputs/clip_knn.json")
parser.add_argument("--output_dir", type=str, default="./Plots")
parser.add_argument(
    "--plot_dir",
    type=str,
    default="./outputs/true_clip_knn_relation_per_class",
)
parser.add_argument("--make_plots", action="store_true")



# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def run_single_method_with_shared_inputs(args, method, shared):
    device = shared["device"]
    bank_dataset = shared["bank_dataset"]
    query_dataset = shared["query_dataset"]
    class_names = shared["class_names"]
    text_bank = shared["text_bank"]
    bank_pack = shared["bank_pack"]
    query_pack = shared["query_pack"]

    if method == "knn":
        out = compute_knn_scores_train_bank_test_query(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            query_clip_pred2=query_pack["clip_pred2"],
            query_clip_pred2_probs=query_pack["clip_pred2_probs"],
            query_clip_margins=query_pack["clip_margins"],
            text_features=text_bank,
            class_names=class_names,
            k=args.k,
            chunk_size=args.chunk_size,
            rank_decay_alpha=args.rank_decay_alpha,
        )

    elif method == "icpe":
        out = compute_icpe_scores_train_bank_test_query(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            class_names=class_names,
            projection_dim=args.icpe_projection_dim,
            covariance_reg=args.icpe_covariance_reg,
            density_temperature=args.icpe_density_temperature,
            chunk_size=args.chunk_size,
            device=device,
        )

    elif method == "trustvlm":
        out = compute_trustvlm_scores(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            class_names=class_names,
            n_shot=args.trustvlm_n_shot,
            tau=args.trustvlm_tau,
            variant=args.trustvlm_variant,
            device=device,
        )

    elif method == "mcm":
        out = compute_mcm_scores(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            class_names=class_names,
            text_features=text_bank,
            tau=args.mcm_tau,
            device=device,
        )

    elif method == "entropy":
        out = compute_entropy_scores(
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            text_features=text_bank,
            class_names=class_names,
            clip_logit_scale=shared["vlm_logit_scale"],
            clip_logit_bias=shared["vlm_logit_bias"],
            chunk_size=args.chunk_size,
            device=device,
            normalize_entropy=True,
        )

    elif method == "bayesvlm":
        out = compute_bayesvlm_scores(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            text_features=text_bank,
            class_names=class_names,
            clip_logit_scale=shared["vlm_logit_scale"],
            clip_logit_bias=shared["vlm_logit_bias"],
            chunk_size=args.chunk_size,
            device=device,
            score_type=args.bayesvlm_score_type,
            normalize_entropy=True,
            image_var_scale=args.bayesvlm_image_var_scale,
            text_var_scale=args.bayesvlm_text_var_scale,
            var_floor=args.bayesvlm_var_floor,
            posterior_proxy=args.bayesvlm_posterior_proxy,
            local_k=args.bayesvlm_local_k,
            local_beta=args.bayesvlm_local_beta,
            target_tpr=0.95,
        )

    elif method == "vilu":
        weights_path = args.vilu_weights_path or os.path.join(
            args.vilu_weights_dir, f"{args.dataset_name}.pt"
        )

        out = compute_vilu_scores(
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            text_features=text_bank,
            class_names=class_names,
            weights_path=weights_path,
            chunk_size=args.chunk_size,
            device=device,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    clip_acc = float(
        (query_pack["clip_preds"] == query_pack["labels"])
        .float()
        .mean()
        .item()
    )

    results = {
        "method": method,
        "score_mode": args.score_mode,
        "dataset_name": args.dataset_name,
        "bank_split": args.bank_split,
        "query_split": args.split,
        "num_bank_samples": int(len(bank_dataset)),
        "num_query_samples": int(len(query_dataset)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "model_name": args.model_name,
        "prompt_template": args.prompt_template,
        "k": int(args.k),
        "clip_query_accuracy": clip_acc,
        "summary": out["summary"],
        "metrics": out["metrics"],
        "per_sample": out["per_sample"],
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    return results, class_names
def print_final_metrics_summary(all_results):
    """
    Print only the final metrics table for all methods that were run.

    Expected input:
        all_results: list of results dictionaries returned by
                     run_train_bank_test_query_knn_analysis(args)
    """
    if len(all_results) == 0:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 80)
    print("FINAL FAILURE DETECTION METRICS")
    print("=" * 80)

    header = (
        f"{'Method':<12}"
        f"{'Dataset':<15}"
        f"{'CLIP Acc':>12}"
        f"{'AUROC':>12}"
        f"{'FPR@95TPR':>14}"
    )
    print(header)
    print("-" * len(header))

    for results in all_results:
        method = results.get("method", "unknown")
        dataset = results.get("dataset_name", "unknown")
        clip_acc = results.get("clip_query_accuracy", float("nan"))

        metrics = results.get("metrics", {})
        auroc = metrics.get("auroc_wrong_positive", float("nan"))
        fpr95 = metrics.get("fpr_at_95_tpr_wrong_positive", float("nan"))

        print(
            f"{method:<12}"
            f"{dataset:<15}"
            f"{clip_acc:>12.4f}"
            f"{auroc:>12.4f}"
            f"{fpr95:>14.4f}"
        )

    print("=" * 80)


def prepare_shared_inputs(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank_dataset = create_torchvision_dataset(
        dataset_name=args.dataset_name,
        root=args.root,
        split=args.bank_split,
        download=args.download,
    )

    query_dataset = create_torchvision_dataset(
        dataset_name=args.dataset_name,
        root=args.root,
        split=args.split,
        download=args.download,
    )

    bank_class_names = infer_class_names(bank_dataset)
    query_class_names = infer_class_names(query_dataset)

    if bank_class_names != query_class_names:
        raise ValueError(
            "Bank and query datasets have different class_names.\n"
            f"Bank classes: {bank_class_names}\n"
            f"Query classes: {query_class_names}"
        )

    class_names = query_class_names

    print(f"Dataset: {args.dataset_name}")
    print(f"Bank split: {args.bank_split} | Query split: {args.split}")
    print(f"Number of classes: {len(class_names)}")
    print(f"First 10 classes: {class_names[:10]}")
    print(f"Bank samples: {len(bank_dataset)}")
    print(f"Query samples: {len(query_dataset)}")

    model = AutoModel.from_pretrained(args.model_name).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    text_bank = build_text_bank(
        model=model,
        processor=processor,
        class_names=class_names,
        prompt_template=args.prompt_template,
        device=device,
    )

    bank_pack = extract_bank_embeddings(
        dataset=bank_dataset,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    query_pack = extract_query_embeddings_and_clip_preds(
        dataset=query_dataset,
        model=model,
        text_bank=text_bank,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Move shared tensors to device once
    bank_pack = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in bank_pack.items()
    }

    query_pack = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in query_pack.items()
    }

    text_bank = text_bank.to(device)

    vlm_logit_scale, vlm_logit_bias = get_vlm_logit_scale_and_bias(model, device)

    shared = {
        "device": device,
        "model": model,
        "processor": processor,
        "bank_dataset": bank_dataset,
        "query_dataset": query_dataset,
        "class_names": class_names,
        "text_bank": text_bank,
        "bank_pack": bank_pack,
        "query_pack": query_pack,
        "vlm_logit_scale": vlm_logit_scale,
        "vlm_logit_bias": vlm_logit_bias,
    }

    return shared


def run_shared_ablations_and_premise(args, shared, methods_to_run):
    device = shared["device"]
    class_names = shared["class_names"]
    text_bank = shared["text_bank"]
    bank_pack = shared["bank_pack"]
    query_pack = shared["query_pack"]

    # ------------------------------------------------------------
    # k ablation: run once, because it only applies to your kNN method
    # ------------------------------------------------------------
    if args.run_k_ablation:
        k_results = sweep_k_values_for_knn_failure_detection(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            query_clip_pred2=query_pack["clip_pred2"],
            query_clip_pred2_probs=query_pack["clip_pred2_probs"],
            query_clip_margins=query_pack["clip_margins"],
            text_features=text_bank,
            class_names=class_names,
            k_values=args.k_ablation_values,
            chunk_size=args.chunk_size,
            output_dir=os.path.join(args.output_dir, "k_ablation"),
            output_prefix=f"{args.dataset_name}_knn_k_sweep",
            target_tpr=0.95,
            save_raw_records=False,
            score_fn=compute_knn_scores_train_bank_test_query,
        )

        plot_k_sweep_results(
            k_results,
            output_dir=os.path.join(args.output_dir, "k_ablation"),
            filename=f"{args.dataset_name}_knn_k_sweep_plot.png",
        )

    # ------------------------------------------------------------
    # rank-decay ablation: run only if kNN is part of the selected methods
    # ------------------------------------------------------------
    if args.run_rank_decay_ablation and ("knn" in methods_to_run):
        rank_decay_results = sweep_rank_decay_alpha_for_knn_failure_detection(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            query_clip_pred2=query_pack["clip_pred2"],
            query_clip_pred2_probs=query_pack["clip_pred2_probs"],
            query_clip_margins=query_pack["clip_margins"],
            text_features=text_bank,
            class_names=class_names,
            alpha_values=args.rank_decay_alpha_values,
            k=args.k,
            chunk_size=args.chunk_size,
            output_dir=os.path.join(args.output_dir, "rank_decay_ablation"),
            output_prefix=f"{args.dataset_name}_knn_rank_decay_alpha_sweep",
            target_tpr=0.95,
            save_raw_records=False,
            score_fn=compute_knn_scores_train_bank_test_query,
        )

        plot_rank_decay_alpha_sweep_results(
            rank_decay_results,
            output_dir=os.path.join(args.output_dir, "rank_decay_ablation"),
            filename=f"{args.dataset_name}_knn_rank_decay_alpha_sweep_plot.png",
            show=False,
        )

    # ------------------------------------------------------------
    # premise validation: run once
    # ------------------------------------------------------------
    if args.run_premise_validation:
        premise_results = run_premise_validation_analysis(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            query_clip_pred_probs=query_pack["clip_pred_probs"],
            query_clip_pred2=query_pack["clip_pred2"],
            query_clip_pred2_probs=query_pack["clip_pred2_probs"],
            query_clip_margins=query_pack["clip_margins"],
            class_names=class_names,
            output_dir=args.premise_output_dir,
            max_intra_pairs_per_class=args.premise_max_intra_pairs_per_class,
            max_inter_pairs=args.premise_max_inter_pairs,
            chunk_size=args.chunk_size,
            random_seed=0,
        )

        pairwise_csv = premise_results["paths"]["pairwise_distance_samples"]
        query_cluster_csv = premise_results["paths"]["query_cluster_distances"]

        plot_pairwise_distance_distributions(
            pairwise_csv=pairwise_csv,
            output_dir=args.premise_output_dir,
            filename="pairwise_intra_inter_distance_distribution.png",
            show=False,
        )

        plot_query_cluster_distance_correct_vs_wrong(
            query_cluster_csv=query_cluster_csv,
            output_dir=args.premise_output_dir,
            filename="query_cluster_distance_correct_vs_wrong.png",
            show=False,
        )

        plot_cluster_margin_correct_vs_wrong(
            query_cluster_csv=query_cluster_csv,
            output_dir=args.premise_output_dir,
            filename="cluster_margin_correct_vs_wrong.png",
            show=False,
        )

    # ------------------------------------------------------------
    # t-SNE premise visualization: run once
    # ------------------------------------------------------------
    if args.run_premise_tsne:
        tsne_df, tsne_csv = save_tsne_bank_query_embedding_data(
            bank_embeddings=bank_pack["embeddings"],
            bank_labels=bank_pack["labels"],
            query_embeddings=query_pack["embeddings"],
            query_labels=query_pack["labels"],
            query_clip_preds=query_pack["clip_preds"],
            class_names=class_names,
            output_dir=args.premise_output_dir,
            filename="tsne_bank_query.csv",
            max_bank_samples=args.premise_tsne_max_bank,
            max_query_samples=args.premise_tsne_max_query,
            random_seed=0,
            perplexity=30,
            n_iter=1000,
        )

        plot_tsne_bank_query(
            tsne_csv=tsne_csv,
            output_dir=args.premise_output_dir,
            filename="tsne_bank_query_correct_wrong.png",
            show=False,
            plot_bank=True,
        )


def main():
    args = parser.parse_args()

    if "all" in args.method:
        methods_to_run = ["knn", "icpe", "trustvlm", "mcm", "entropy", "bayesvlm"]
    else:
        methods_to_run = args.method

    base_output_json = args.output_json
    output_dir = os.path.dirname(base_output_json)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Preparing shared VLM inputs once")
    print("=" * 80)

    shared = prepare_shared_inputs(args)

    # ------------------------------------------------------------
    # Run ablations / premise validation once using shared embeddings
    # ------------------------------------------------------------
    run_shared_ablations_and_premise(
        args=args,
        shared=shared,
        methods_to_run=methods_to_run,
    )

    all_results = []

    for method in methods_to_run:
        print("\n" + "=" * 80)
        print(f"Running method: {method}")
        print("=" * 80)

        args.output_json = os.path.join(
            output_dir,
            f"{method}_{args.dataset_name}.json"
        )

        results, class_names = run_single_method_with_shared_inputs(
            args=args,
            method=method,
            shared=shared,
        )

        all_results.append(results)

        if args.make_plots:
            args.plot_dir = os.path.join(
                args.output_dir,
                f"true_clip_relation_{method}_{args.dataset_name}"
            )

            plot_true_clip_knn_relation_per_class(
                results=results,
                class_names=class_names,
                save_dir=args.plot_dir,
                use_weighted_knn=True,
                max_samples_per_class=None,
            )

    print_final_metrics_summary(all_results)
if __name__ == "__main__":
    main()