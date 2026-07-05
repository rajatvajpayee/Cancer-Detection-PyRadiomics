# Cancer-Detection-PyRadiomics

## Overview

This repository provides a modular experimentation framework for predicting chemotherapy response from CT-derived radiomics features.

It supports multiple feature selection techniques and modern tabular learning models, enabling reproducible benchmarking under a unified evaluation pipeline.

Implemented models include:

- XGBoost
- TabICLv2
- Google TabFM

---

## Project Structure

```
Cancer-Detection-PyRadiomics/
├── data/
│   └── data.csv                    ← your extracted radiomics CSV
├── configs/
│   ├── base_config.yaml            ← shared defaults (inherited by all)
│   ├── phase1_covariance.yaml      ← Phase 1: covariance filter only
│   ├── phase1_mi.yaml              ← Phase 1: MI only
│   ├── phase1_hsic.yaml            ← Phase 1: HSIC only
│   ├── phase2_covar_mi.yaml        ← Phase 2: covariance → MI (sequential)
│   └── phase2_covar_hsic.yaml      ← Phase 2: covariance → HSIC (sequential)
│   └── .....<MORE CONFIGS>
|   
├── src/
│   ├── features/
│   │   ├── covariance_filter.py    ← Pearson correlation filter
│   │   ├── mutual_information.py   ← MI-based top-k selector
│   │   ├── hsic.py                 ← HSIC-based top-k selector
│   │   └── feature_pipeline.py     ← orchestrates sequential selection
│   ├── models/
│   │   ├── xgboost_wrapper.py      ← CPU XGBoost (hist, scale_pos_weight)
│   │   ├── tabicl_wrapper.py       ← TabICL with VRAM cleanup
│   │   ├── tabfm_wrapper.py        ← TabFM with VRAM cleanup
│   │   └── model_factory.py        ← get_model(name, cfg)
│   ├── evaluation/
│   │   ├── cv_runner.py            ← stratified 5-fold CV loop
│   │   ├── metrics.py              ← AUC, F1, CM, ROC plots
│   │   └── threshold_sweep.py      ← global OOF threshold sweep
│   └── utils/
│       ├── data_loader.py          ← load_data() → (X, y)
│       ├── logger.py               ← file + console logging
│       └── seed.py                 ← global reproducibility seed
├── outputs/
│   ├── phase1/{covariance,mi,hsic}/{tabicl,tabfm, xgboost}/
│   └── phase2/{covar_mi,covar_hsic}/{tabicl,tabfm, xgboost}/
│       Each folder gets: *_metrics.csv, *_roc.png, *_cm.png
├── logs/                           ← timestamped .log files
└── run_experiment.py               ← unified argparse entry point
```

---
## Models

### 1. XGBoost
- Gradient boosting algorithm that builds an ensemble of decision trees for **classification and regression** on structured/tabular data.
- Known for **high accuracy, fast training, regularization, and efficient handling of missing values**, making it an industry standard for tabular ML.
- Requires **model training and hyperparameter tuning**, but remains one of the strongest baselines for tabular prediction tasks.

### 2. TabICL
- A **Transformer-based tabular foundation model** that performs prediction using **in-context learning**, without retraining for each new dataset.
- Learns from a few labeled examples provided in the input context, reducing the need for task-specific optimization.
- Enables **zero-shot and few-shot learning** for tabular prediction, reducing dependence on extensive hyperparameter tuning.

**Paper:**  
- *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model*  
  https://arxiv.org/abs/2602.11139

---

### 3. TabFM (Google Research, v1.0.0)
- Google's **latest tabular foundation model** for classification and regression, released in **2026** with publicly available pretrained checkpoints.
- Uses **zero-shot in-context learning**, eliminating the need for dataset-specific training, feature engineering, or hyperparameter tuning.
- Employs a **hybrid attention architecture** to achieve competitive performance across diverse tabular tasks.

**Resources:**
- Google Research Blog: https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/
- GitHub: https://github.com/google-research/tabfm


| Model | Year | Approach |
|-------|------|----------|
| XGBoost | 2016 | Gradient Boosted Decision Trees |
| TabICLv2 | 2025/2026 | Transformer + In-Context Learning |
| TabFM | 2026 | Foundation Model for Tabular Data |

---
## Usage

### Environment Setup

```bash
# Create a new Conda environment
conda create -n cancer-detection python=3.11 -y

# Activate the environment
conda activate cancer-detection

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

Experiments are configured using YAML configuration files in the `configs/` directory. You can run a specific experiment by selecting a configuration file and model, or execute all predefined experiments in a single command.

```bash
# Run a single experiment
python run_experiment.py --config configs/phase1_mi.yaml --model tabicl

# Run all experiments
python run_experiment.py --all
```

- `--config`: Specifies the experiment configuration (feature selection pipeline, preprocessing, and evaluation settings).
- `--model`: Selects the model to evaluate (`xgboost`, `tabicl`, or `tabfm`).
- `--all`: Runs every predefined configuration across all supported models and saves the results under the `outputs/` directory.

---

## Architecture 
```
CT Images
      │
      ▼
PyRadiomics
      │
      ▼
Feature Selection
(Covariance / MI / HSIC)
      │
      ▼
Model
(XGBoost / TabICL / TabFM)
      │
      ▼
5-Fold CV
      │
      ▼
Metrics
```


## Feature Selection

Three feature selection strategies are implemented:

- **Covariance Filter:** Removes highly correlated features to reduce redundancy.
- **Mutual Information:** Selects features with the highest dependency on the target variable.
- **HSIC:** Captures nonlinear dependence between features and labels.