# TRINE: Training-Free Failure Detection for CLIP-Style Vision-Language Models

This repository contains the implementation of **TRINE**, a training-free, post-hoc uncertainty and failure detection method for zero-shot classification with CLIP-style vision-language models. The code supports both **CLIP** and **SigLIP** backbones and compares TRINE against several uncertainty and reliability baselines.

## Overview

Vision-language models such as CLIP and SigLIP are widely used for zero-shot image classification, but their predictions can still be unreliable under ambiguity, fine-grained class overlap, or dataset shift. TRINE addresses this problem by estimating whether a query image's local neighborhood in the vision-language embedding space supports the model's predicted class.

TRINE uses a labeled reference bank and combines three sources of neighborhood evidence:

1. **Image-side similarity** between the query image and retrieved reference-bank images.
2. **Rank-decayed neighbor importance**, where closer-ranked neighbors contribute more strongly.
3. **Text-guided semantic consistency** between the predicted class and the retrieved neighbor classes in the shared embedding space.

The final failure score is based on the class-conditional support margin. Samples with weak support for the predicted class relative to competing classes are assigned higher failure scores.

## Repository Structure

```text
TRINE/
│
├── main.py
├── helper_functions.py
├── helper_functions_siglip.py
├── dataset_helpers_original.py
│
├── bayesvlm_method.py
├── entropy_method.py
├── icpe_method.py
├── mcm_method.py
├── trustvlm_method.py
├── vilu_method.py
│
└── .gitignore
```

## Supported Backbones

The code currently supports:

* OpenAI CLIP
* Google SigLIP

The backbone can be selected using the `--vlm_type` argument.

## Supported Methods

The repository includes the following failure detection and uncertainty estimation methods:

* `knn`: TRINE / neighborhood-based evidence scoring
* `icpe`: ICPE-based scoring
* `trustvlm`: TrustVLM-style scoring
* `mcm`: Maximum concept matching score
* `entropy`: Predictive entropy
* `bayesvlm`: BayesVLM-style uncertainty scoring
* `all`: Run all available methods

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n trine python=3.10
conda activate trine
```

Install the main dependencies:

```bash
pip install torch torchvision transformers numpy scikit-learn matplotlib tqdm pillow
```

For SigLIP, install SentencePiece if needed:

```bash
pip install sentencepiece
```
<!-- 
If FAISS-based retrieval is used in your helper functions, install either the CPU or GPU version:

```bash
pip install faiss-cpu
```

or, for GPU support:

```bash
conda install -c pytorch -c nvidia faiss-gpu -->
```

## Usage

### Run TRINE with SigLIP

```bash
python main.py \
  --vlm_type siglip \
  --dataset_name fgvcaircraft \
  --root /path/to/data \
  --bank_split train \
  --split test \
  --method knn
```

### Run TRINE with CLIP

```bash
python main.py \
  --vlm_type clip \
  --dataset_name fgvcaircraft \
  --root /path/to/data \
  --bank_split train \
  --split test \
  --method knn
```

### Run all methods

```bash
python main.py \
  --vlm_type siglip \
  --dataset_name fgvcaircraft \
  --root /path/to/data \
  --bank_split train \
  --split test \
  --method all
```

### Run multiple selected methods

```bash
python main.py \
  --vlm_type clip \
  --dataset_name fgvcaircraft \
  --root /path/to/data \
  --method knn entropy mcm bayesvlm
```

## Main Arguments

| Argument             | Description                                                                             |
| -------------------- | --------------------------------------------------------------------------------------- |
| `--vlm_type`         | Backbone type: `clip`, `siglip`, or `auto`                                              |
| `--model_name`       | HuggingFace model name. If omitted, the default model for the selected backbone is used |
| `--prompt_template`  | Text prompt template for class names                                                    |
| `--dataset_name`     | Dataset name used by `dataset_helpers_original.py`                                      |
| `--root`             | Root directory where datasets are stored                                                |
| `--bank_split`       | Split used as the labeled reference bank                                                |
| `--split`            | Query/test split                                                                        |
| `--method`           | Method or methods to run                                                                |
| `--k`                | Number of nearest neighbors for TRINE                                                   |
| `--rank_decay_alpha` | Rank-decay coefficient used by TRINE                                                    |
| `--batch_size`       | Batch size for embedding extraction                                                     |
| `--chunk_size`       | Chunk size for similarity computation                                                   |
| `--output_json`      | Path for saving output metrics and per-sample scores                                    |
| `--output_dir`       | Directory for saving plots and ablation outputs                                         |

<!-- ## Ablation Studies

The script supports ablation studies for the number of neighbors and rank-decay coefficient.

### Sweep k values

```bash
python main.py \
  --vlm_type siglip \
  --dataset_name fgvcaircraft \
  --method knn \
  --run_k_ablation \
  --k_ablation_values 10 15 20 50 100
```

### Sweep rank-decay alpha values

```bash
python main.py \
  --vlm_type siglip \
  --dataset_name fgvcaircraft \
  --method knn \
  --run_rank_decay_ablation \
  --rank_decay_alpha_values 0.0 0.01 0.02 0.05 0.1 0.2 0.5 1.0
``` -->

## Outputs

The code saves:

* Final failure detection metrics
* Per-sample prediction and failure score records
* Method-specific JSON output files
* Optional plots for per-class failure analysis
* Optional ablation results

Typical metrics include:

* AUROC for detecting incorrect zero-shot predictions
* FPR at 95% TPR
* Zero-shot classification accuracy of the selected VLM backbone

## Notes

Datasets, generated outputs, model checkpoints, plots, and cached files are intentionally excluded from the repository through `.gitignore`.

The repository assumes that the dataset loading logic is handled through:

```text
dataset_helpers_original.py
```

If a new dataset is added, update the dataset helper accordingly.



