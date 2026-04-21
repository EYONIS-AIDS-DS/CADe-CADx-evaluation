# CADe/CADx Performance Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/EYONIS-AIDS-DS/CADe-CADx-evaluation)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2512.00281-b31b1b.svg)](https://arxiv.org/abs/2512.00281)

This repository provides a complete, standardised performance evaluation framework for **CADe** (Computer-Aided Detection) and **CADx** (Computer-Aided Diagnosis) systems.  
It targets binary detection and classification systems (e.g. malignant nodule detection) and computes ROC curves, Precision-Recall curves, FROCs (with configurable operating points), bootstrap confidence intervals, and statistical comparison tests across arbitrary sets of predictions.

It serves two independent purposes:

1. **Reproduce** all results of the preprint *"Rethinking Lung Cancer Screening: AI Nodule Detection and Diagnosis Outperforms Radiologists, Leading Models, and Standards Beyond Size and Growth"* ([arXiv:2512.00281](https://arxiv.org/abs/2512.00281)).
2. **Provide** a reusable, generic evaluation pipeline for any CADe/CADx system on any dataset.

---

## Table of Contents

1. [Installation](#1-installation)
   - 1.1 [uv — Package Manager & Git](#11-uv--package-manager--git)
   - 1.2 [Clone the Repository](#12-clone-the-repository)
   - 1.3 [Install Dependencies](#13-install-dependencies)
   - 1.4 [Data Files Setup](#14-data-files-setup)
2. [Repository Structure](#2-repository-structure)
   - 2.1 [Configuration](#21-configuration)
   - 2.2 [Input Format](#22-input-format)
   - 2.3 [Modules evaluate_series & evaluate_lesions](#23-modules-evaluate_series--evaluate_lesions)
   - 2.4 [Module statistical_tests](#24-module-statistical_tests)
   - 2.5 [Outputs](#25-outputs)
3. [Reproduce All Paper Results](#3-reproduce-all-paper-results)
4. [Independent Submodule Usage](#4-independent-submodule-usage)
   - 4.1 [Submodule evaluate_series](#41-submodule-evaluate_series)
   - 4.2 [Submodule evaluate_lesions](#42-submodule-evaluate_lesions)
   - 4.3 [Submodule statistical_tests](#43-submodule-statistical_tests)
5. [Citation](#5-citation)

---

## 1. Installation

### 1.1 uv — Package Manager & Git

This repository uses **uv**, an ultra-fast Python package and environment manager (see the [complete uv installation guide](https://uv.pypa.io/en/stable/installation/)).  
If you do not have it yet, install it with:

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or, if curl is not available:
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If you do not have **Git**, follow the [Git installation guide](https://git-scm.com/downloads).  
On Linux/macOS: `sudo apt-get install git`  
On Windows: download the [installer](https://git-scm.com/download/win) or use `choco install git`.

### 1.2 Clone the Repository

```bash
git clone https://github.com/EYONIS-AIDS-DS/CADe-CADx-evaluation.git
cd CADe-CADx-evaluation
```

Alternatively, download the ZIP from the GitHub *Code* button and unzip it.

### 1.3 Install Dependencies

Create the virtual environment and install all dependencies in one step:

```bash
uv venv
uv sync
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

> **Python version:** the project requires Python >= 3.11, < 3.12. `uv` will download the correct interpreter automatically if needed.

### 1.4 Data Files Setup

Large CSV input files are stored via **Git LFS**. Pull them after cloning:

```bash
# Install Git LFS (once per machine)
# macOS:   brew install git-lfs
# Windows: choco install git-lfs  (or download from https://git-lfs.github.com/)
# Ubuntu:  sudo apt-get install git-lfs

git lfs install
git lfs pull
```

Verify the files were correctly downloaded (the first line should be a CSV header, not a Git LFS pointer):

```bash
head -1 data/data_lesions/lesions.csv
```

**If Git LFS is unavailable**, download the files manually (PowerShell):

```powershell
$files = @(
    "lesions.csv",
    "lesions_radiologists.csv",
    "lesions_longitudinal.csv",
    "lesions_model_CADe.csv",
    "lesions_nndetection_baumgartner_CADe.csv",
    "lesions_nndetection_CADe.csv",
    "lesions_nndetection_CADx.csv"
)
$base_url = "https://media.githubusercontent.com/media/EYONIS-AIDS-DS/CADe-CADx-evaluation/refs/heads/main/data/data_lesions"
foreach ($file in $files) {
    Invoke-WebRequest -Uri "$base_url/$file" -OutFile "data/data_lesions/$file"
}
```

---

## 2. Repository Structure

```
CADe-CADx-evaluation/
│
│   README.md
│   config_paper.py          <- all evaluation parameters for the paper
│   run_paper_evaluation.py  <- entry point to reproduce the full paper
│   pyproject.toml
│
├── data/
│   │   CADe_CADx_evaluate.log       (OUTPUT — created at runtime)
│   │
│   ├── data_series/                 (INPUT)
│   │       series.csv
│   │
│   ├── data_lesions/                (INPUT)
│   │       lesions.csv
│   │       lesions_radiologists.csv
│   │       lesions_longitudinal.csv
│   │       lesions_model_CADe.csv
│   │       lesions_nndetection_CADe.csv
│   │       lesions_nndetection_CADex.csv
│   │       lesions_nndetection_baumgartner_CADe.csv
│   │
│   ├── evaluate_series/             (OUTPUT — created at runtime)
│   │   └── figure_1/
│   │       └── model_prediction_test1/
│   │               roc_curve_with_op_test1.png
│   │               roc_CI_bootstrap_hanley_test1.csv
│   │               operating_point_performances_test1.csv
│   │               AUC_array_5000_bootstrap_test1.npy
│   │               ...
│   │
│   ├── evaluate_lesions/            (OUTPUT — created at runtime)
│   │   └── figure_4/
│   │       └── model_prediction_test1/
│   │               froc_curve_with_2_op_test1.png
│   │               operating_point_FROC_scores_at_0.5_and_1_FP_per_scan_test1.csv
│   │               ...
│   │
│   └── statistical_tests/          (OUTPUT — created at runtime)
│       └── figure_1/
│               test_results_AUC_model_vs_4radiolog_test3.csv
│               distribution_BOOTSTRAPP_AUC_model_vs_4radiolog_test3.png
│               ...
│
└── CADe_CADx_evaluation/           (PACKAGE)
    ├── evaluate_common/
    │       roc.py
    │       roc_confidence_interval.py
    │       precision_recall.py
    │       sens_spec.py
    │       sample_size_analysis.py
    │       plot_score_distribution_benign_cancer.py
    │       logger.py
    │
    ├── evaluate_series/
    │       evaluate_series.py
    │
    ├── evaluate_lesions/
    │       evaluate_lesions.py
    │       froc.py
    │       plot_diameter_prediction_distributions.py
    │
    └── statistical_tests/
            statistical_tests.py
```

### 2.1 Configuration

All parameters are centralised in `config_paper.py`. It defines:

- **Input / output paths** for data and results.
- **Prediction names** — column names in the CSV files corresponding to model or reader scores.
- **Label names** — column names for ground-truth binary labels.
- **Subset indicator names** — boolean columns that select evaluation subsets (e.g. `nlst_test_1`).
- **Bootstrap settings** — number of bootstrap samples (`nb_bootstrap_samples`) and confidence level (`confidence_threshold`).
- **Operating point thresholds and labels** — threshold values and human-readable names for each operating point on the ROC/FROC.
- **Fast computation flag** — when `fast_computation=True`, bootstrap samples are reduced to 50 and FROC resolution is lowered, allowing a full run in ~8 minutes instead of ~24 hours.
- **Evaluation lists** — lists of `(set_name, prediction, label, thresholds, labels)` tuples, grouped by figure.

To apply this package to a **new model or dataset**, either:
- Rewrite `config_paper.py` following the same structure (recommended for multi-figure evaluations), or
- Call the submodule functions directly (see [Section 4](#4-independent-submodule-usage)).

### 2.2 Input Format

Inputs are CSV files located in `data/data_series/` (patient/series level) and `data/data_lesions/` (nodule/lesion level). Each row is one sample; each column is a feature. The evaluation uses four types of columns:

| Type | Description | Examples |
|------|-------------|---------|
| **Predictions** | Numeric score (float) — model probability, reader score, or any predictive variable | `model_prediction`, `radiologist_1_max_malignancy`, `diameter_lad` |
| **Labels** | Binary ground-truth: `0` = negative, `1` = positive | `label`, `label_nodule` |
| **Identifiers** | Unique sample IDs used to compute per-scan FP rates | `series_uid`, `patient_id`, `detection_id` |
| **Subset indicators** | Boolean columns that define evaluation subsets | `nlst_test_1`, `nlst_test_3`, `radiologist_1` |

> **Lesion-level only:** an additional `detection_status` column is required, taking values `"TP"`, `"FP"`, or `"FN"` (True Positive, False Positive, False Negative) resulting from prior IoU-based pairing of detections with ground-truth lesions. False negative detections are assigned a prediction score of 0 (worst score) by the evaluation algorithm.

### 2.3 Modules `evaluate_series` & `evaluate_lesions`

These two modules perform the same computations at different levels:

- **`evaluate_series`** — patient/series level. Ground truth is straightforward (e.g. cancer diagnosis per scan).
- **`evaluate_lesions`** — nodule/lesion level. Requires prior detection–GT pairing (not included; see [Jaeger et al., nnDetection](https://arxiv.org/abs/1811.08661) for a standard IoU >= 0.1 pairing). Lesion-level evaluation additionally includes FROC curves.

Both modules call shared utilities from `evaluate_common` (ROC, Precision-Recall, bootstrap CI, etc.).

### 2.4 Module `statistical_tests`

Implements non-parametric bootstrap-based superiority tests ([Efron & Tibshirani](https://www.hms.harvard.edu/bss/neuro/bornlab/nb204/statistics/bootstrap.pdf)) to compare pairs of models or readers.

**Input:** pairs of `.npy` bootstrap vector files produced by `evaluate_series` or `evaluate_lesions`.  
**Tests performed** (using [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)):
- One-sided Welch t-test (unequal variance) — H1: prediction 1 > prediction 2
- One-sided Student t-test (equal variance) — H1: prediction 1 > prediction 2

**Output per pair:** a CSV with p-values and significance labels (`extremely strong`, `very strong`, `strong`, `moderate`, `weak`, `no evidence`), plus a figure showing the two bootstrap distributions.

> This module must be run **after** `evaluate_series` / `evaluate_lesions`, as it depends on their `.npy` bootstrap output files.

### 2.5 Outputs

All outputs are written under `data/evaluate_series/`, `data/evaluate_lesions/`, and `data/statistical_tests/`, organised by figure name and analysis name.

**Figures** (saved as `.png` and `.svg` for vector editing):
- ROC curve with operating points (Youden index maximum by default, or any custom threshold list)
- Precision-Recall curve
- Distribution of malignancy predictions (benign vs. malignant)
- FROC curve with operating points at 0.5 FP/scan and 1 FP/scan *(lesion level only)*
- Bootstrap distribution comparison *(statistical tests only)*

**CSV files:**
- `sample_size_<set>.csv` — total, positive, and negative sample counts with imbalance ratio
- `roc_CI_bootstrap_hanley_<set>.csv` — AUC with bootstrap and [Hanley & McNeil](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747) confidence intervals
- `operating_point_performances_<set>.csv` — sensitivity, specificity, accuracy with bootstrap mean and CI at each operating point
- `operating_point_FROC_scores_at_0.5_and_1_FP_per_scan_<set>.csv` — FROC sensitivity and FP/scan with CI *(lesion level only)*
- `test_results_<comparison>.csv` — p-values and significance conclusions *(statistical tests only)*

**NumPy arrays (`.npy`):**
- `AUC_array_5000_bootstrap_<set>.npy` — bootstrap AUC samples
- `sensitivity_array_5000_bootstrap_for_each_OP_ROC_<set>.npy` — bootstrap sensitivity samples per operating point
- `specificity_array_5000_bootstrap_for_each_OP_ROC_<set>.npy`
- `accuracy_array_5000_bootstrap_for_each_OP_ROC_<set>.npy`
- `at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_<set>.npy` *(lesion level only)*

---

## 3. Reproduce All Paper Results

The `run_paper_evaluation.py` script integrates all evaluation modules and reproduces every figure, performance table, and statistical test in the paper in a single command.

**Fast run** (~8 minutes, 50 bootstrap samples — qualitatively equivalent):

```bash
# with uv
uv run python run_paper_evaluation.py --fast_computation=true

# or with the activated virtual environment
python run_paper_evaluation.py --fast_computation=true
```

**Full paper reproduction** (~24 hours, 5000 bootstrap samples):

```bash
uv run python run_paper_evaluation.py --fast_computation=false
```

**From a Python script:**

```python
from run_paper_evaluation import paper_evaluation

paper_evaluation(fast_computation=True)   # fast demo
# paper_evaluation(fast_computation=False)  # full paper reproduction
```

---

## 4. Independent Submodule Usage

The three submodules can be used independently on any dataset. The example below generates a synthetic CSV and evaluates it.

**Create a synthetic series CSV:**

```python
import numpy as np
import pandas as pd
from pathlib import Path

path_data_series = Path("data/data_series/series_example.csv")
sample_size = 1000
np.random.seed(42)

df = pd.DataFrame({
    "series_uid": np.arange(1, sample_size + 1),
    "label_col":  np.random.randint(0, 2, size=sample_size),
    "subset_col": np.random.choice([True, False], size=sample_size, p=[0.9, 0.1]),
})
# Make predictions correlated with labels
df["prediction_col"] = (
    ((df["label_col"] * 2) - 1) * np.random.rand(sample_size) * 0.5 + 0.5
)
df.to_csv(path_data_series, index=False)
```

### 4.1 Submodule `evaluate_series`

Evaluates model/reader performance at the **patient/series level**.

```python
from pathlib import Path
from CADe_CADx_evaluation.evaluate_series.evaluate_series import evaluate_serie_main

evaluate_serie_main(
    path_to_load_csv_serie=Path("data/data_series/series_example.csv"),
    expdir=Path("data/evaluate_series/my_experiment"),
    set_name="subset_col",                  # boolean column defining the evaluation subset
    prediction="prediction_col",            # numeric score column
    label_name="label_col",                 # binary ground-truth column (0/1)
    operating_point_thresholds=[0],         # 0 = Youden Index Maximum; or e.g. [0.3, 0.5, 0]
    operating_point_labels=["Youden Max"],  # human-readable label per threshold
    nb_bootstrap_samples=500,
    confidence_threshold=0.95,
)
```

All outputs (ROC, Precision-Recall, CSV scores, `.npy` bootstrap arrays) are saved under
`data/evaluate_series/my_experiment/<prediction_col>_<subset_col>/`.

### 4.2 Submodule `evaluate_lesions`

Evaluates model/reader performance at the **nodule/lesion level**, including FROC curves.

> **Prerequisite:** the input CSV must contain a `detection_status` column (`"TP"`, `"FP"`, `"FN"`) and a `patient_id` column (used to compute FP per scan). Detection–GT pairing must be performed externally before calling this module (e.g. using [nnDetection](https://github.com/MIC-DKFZ/nnDetection) with IoU >= 0.1).

```python
from pathlib import Path
from CADe_CADx_evaluation.evaluate_lesions.evaluate_lesions import evaluate_lesions_main

evaluate_lesions_main(
    fast_computation=True,                    # True: 1 000-point FROC grid; False: 500 000-point grid
    path_to_load_csv_lesion=Path("data/data_lesions/lesions.csv"),
    expdir=Path("data/evaluate_lesions/my_experiment"),
    set_name="nlst_test_1",                   # boolean subset column, or any set_name (uses full df if unknown)
    prediction="model_prediction",            # numeric score column
    label_name="label",                       # binary ground-truth column (0/1)
    operating_point_thresholds=[0],           # 0 = Youden Index Maximum
    operating_point_labels=["Youden Max"],
    nb_bootstrap_samples=500,
    confidence_threshold=0.95,
)
```

Outputs are saved under `data/evaluate_lesions/my_experiment/<prediction>_<set_name>/` and include
everything from `evaluate_series` plus FROC figures and FROC operating-point CSV files.

### 4.3 Submodule `statistical_tests`

Performs bootstrap-based superiority tests between pairs of models or readers.

> **Prerequisite:** `.npy` bootstrap files must already exist, produced by `evaluate_serie_main` or `evaluate_lesions_main`.

```python
from pathlib import Path
from CADe_CADx_evaluation.statistical_tests.statistical_tests import statistical_test_main

root = Path("data/evaluate_series/my_experiment")

# Each tuple: (comparison_name, path_to_bootstrap_array_1, path_to_bootstrap_array_2)
# H1: array_1 > array_2  (one-sided superiority test)
comparisons = [
    (
        "AUC_model_vs_reader",
        root / "model_prediction_test1" / "AUC_array_5000_bootstrap_test1.npy",
        root / "reader_prediction_test1" / "AUC_array_5000_bootstrap_test1.npy",
    ),
    (
        "SENS_model_vs_reader",
        root / "model_prediction_test1" / "sensitivity_array_5000_bootstrap_for_each_OP_ROC_test1.npy",
        root / "reader_prediction_test1" / "sensitivity_array_5000_bootstrap_for_each_OP_ROC_test1.npy",
    ),
]

statistical_test_main(
    list_of_tuples_of_pairs_of_bootstrap_paths=comparisons,
    expdir_analysis=Path("data/statistical_tests/my_experiment"),
)
```

For each comparison the module saves:
- `test_results_<comparison_name>.csv` — p-values and significance conclusions for both equal- and unequal-variance t-tests.
- `distribution_BOOTSTRAPP_<comparison_name>.png/.svg` — overlaid histogram of the two bootstrap distributions.

When the input arrays are 2-D (one column per operating point), one test and one figure are produced per operating point.

---

## 5. Citation

If you use this repository in your research, please cite the associated preprint:

```bibtex
@misc{baudot2024rethinking,
  title         = {Rethinking Lung Cancer Screening: AI Nodule Detection and Diagnosis
                   Outperforms Radiologists, Leading Models, and Standards Beyond Size and Growth},
  author        = {Pierre Baudot and others},
  year          = {2024},
  eprint        = {2512.00281},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2512.00281}
}
```
