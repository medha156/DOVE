# DOVE — Detecting Ornithology with Video Embeddings

**D**etection **O**f **V**isual **E**ntities is a research pipeline for bird species classification using a multi-modal fusion of motion-trajectory features and appearance (invariant) features.

## Overview

DOVE classifies 20 bird species from short video clips and/or still images by fusing:

1. **Motion features (139-d)** — Curvature Scale Space (CSS), Turn/heading, Wingbeat FFT, CDF, Vicinity, and Curvature-bearing descriptors extracted from object trajectories via background subtraction.
2. **Invariant features (1440-d)** — Multi-scale appearance embeddings from a Swin-Tiny backbone with Hierarchical Scale-aware Feature Aggregation (HSFA) and Feature Filtering by Activation (FFA).
3. **Fusion** — Concatenation or cross-attention of both feature streams → 512-d representation.
4. **Classification head** — MLP, Linear, Random Forest, or Naive Bayes.

Optional **Filtration-Distillation Learning (FDL)** trains a student model with teacher knowledge distillation guided by FFA confidence scores.

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd DOVE

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
DOVE/
  data/
    inaturalist/    ← iNaturalist images, one subfolder per species
    vb100/          ← VB100 video clips, one subfolder per species
    splits/         ← generated train/val/test CSVs + class weights
  src/
    data/           ← loaders, augmentation, splits
    features/
      motion/       ← trajectory-based feature extractors (139-d)
      invariant/    ← Swin+HSFA appearance features (1440-d)
    fusion/         ← concat and cross-attention fusion
    backbones/      ← VGG19, MobileNet, EfficientNet-B3, Swin-T, Faster/Cascade RCNN, ATSS
    heads/          ← MLP, Linear, RF, NB heads
    training/       ← Trainer, FDLTrainer, Optuna hyperopt
    evaluation/     ← metrics, visualisation, compute analysis
    ablations/      ← 12-ablation runner
  results/
    figures/        ← saved plots
    tables/         ← CSV logs and ablation results
  configs/
    base.yaml       ← base hyperparameters
    experiments/    ← best configs from Optuna runs
  scripts/
    prepare_data.py
    extract_motion.py
    extract_invariant.py
    run_experiments.py
    run_ablations.py
    upload_results.py
```

## Usage

### 1. Prepare Data

Organise your data as:
```
data/inaturalist/<Species Name>/<image.jpg>
data/vb100/<Species Name>/<clip.mp4>
```

Then run:
```bash
python scripts/prepare_data.py
```
This generates `data/splits/train.csv`, `val.csv`, `test.csv`, and `class_weights.npy`.

### 2. Extract Features

```bash
# Motion features from VB100 videos
python scripts/extract_motion.py

# Invariant features from images + videos
python scripts/extract_invariant.py
```

### 3. Run Experiments

```bash
# Full 112-experiment grid
python scripts/run_experiments.py

# Ablation studies
python scripts/run_ablations.py
```

### 4. Upload Results

```bash
python scripts/upload_results.py
```

## Experiment Grid (112 experiments)

| Factor | Options |
|--------|---------|
| Backbone | VGG19, MobileNet, EfficientNet-B3, Swin-T |
| Fusion | Cross-Attention, Concatenation |
| Head | MLP, Linear, Random Forest, Naive Bayes |
| Features | Motion+Invariant, Invariant-only |
| Training | Standard, FDL (knowledge distillation) |

## Results

| Model | Backbone | Fusion | Head | Accuracy | FLOPs |
|-------|----------|--------|------|----------|-------|
| TBD | TBD | TBD | TBD | — | — |

*(Results will be populated after running experiments.)*

## Target Species (20 classes)

| # | Species | # | Species |
|---|---------|---|---------|
| 0 | Acorn Woodpecker | 10 | Cooper's Hawk |
| 1 | American Crow | 11 | Dark-eyed Junco |
| 2 | American Robin | 12 | House Finch |
| 3 | Anna's Hummingbird | 13 | Lesser Goldfinch |
| 4 | Black Phoebe | 14 | Mourning Dove |
| 5 | Brewer's Blackbird | 15 | Northern Mockingbird |
| 6 | Bushtit | 16 | Oak Titmouse |
| 7 | California Scrub-Jay | 17 | Red-tailed Hawk |
| 8 | California Towhee | 18 | White-crowned Sparrow |
| 9 | Chestnut-backed Chickadee | 19 | Yellow-rumped Warbler |

## Key Hyperparameters

See `configs/base.yaml` for full configuration. Defaults:

| Parameter | Value |
|-----------|-------|
| Backbone | Swin-T |
| Fusion | Cross-Attention |
| d_model | 512 |
| lr | 1e-4 |
| Batch size | 32 |
| Epochs | 50 |
| FDL temperature | 4 |
| FFA k | 100 |
| Motion window | 64 |
