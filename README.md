# Intelligent Threat Detection for MQTT Protocol in Resource-Constrained Edge Computing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-lightblue.svg)](https://imbalanced-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-013243.svg?logo=numpy)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.5+-150458.svg?logo=pandas)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.10+-8CAAE6.svg)](https://scipy.org/)
[![Reviewer R4C2](https://img.shields.io/badge/Revision-R4C2-red.svg)]()
[![Journal: IoT Elsevier](https://img.shields.io/badge/Journal-Internet%20of%20Things%20(Elsevier)-brightgreen.svg)](https://www.journals.elsevier.com/internet-of-things)
[![Dataset: Google Drive](https://img.shields.io/badge/Dataset-Google%20Drive-yellow.svg?logo=googledrive)](https://drive.google.com/drive/folders/1XGhSibCliOYjynWwBJfsvNyaxGe_CzPY?usp=sharing)
[![CV: 10-Fold Stratified](https://img.shields.io/badge/CV-10--Fold%20Stratified-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## Overview
This repository is a **reproducibility package** for a lightweight, tree-based MQTT Intrusion Detection System (IDS) evaluated under strict timing transparency. It provides the complete experimental pipeline used in the manuscript submission to *Internet of Things (Elsevier)*, including baseline replications, the proposed method, and external validation. The focus is on **transparent, CPU-only timing measurements** and **10-fold stratified cross-validation**.

---
## Reproducibility Checklist
- [x] Deterministic environment documentation (`phase0_env/document_environment.py`)
- [x] 10-fold stratified cross-validation across all experiments
- [x] CPU-only inference timing with explicit warm-up handling
- [x] Statistical reporting (mean ± std, 95% CI, Wilcoxon signed-rank)
- [x] External validation on independent MQTT datasets

---
## Key Experiments (Scripts)

| Experiment | Script |
|---|---|
| Environment documentation | `phase0_env/document_environment.py` |
| Paper [4] baseline reproduction | `phase1_baselines/reproduce_paper4.py` |
| Paper [5] baseline reproduction | `phase1_baselines/reproduce_paper5.py` |
| Paper [26] baseline (SMOTE per-fold) | `phase1_baselines/reproduce_paper26.py` |
| Proposed approach (ENN + SMOTE + multi-model) | `phase1_baselines/our_approach.py` |
| External validation — MQTT-IoT-IDS2020 | `phase2_external_validation/validate_ids2020.py` |
| External validation — MQTTset | `phase2_external_validation/validate_mqttset.py` |

All datasets used across all reproduced papers are available in the shared repository:  
- [Google Drive — Datasets](https://drive.google.com/drive/folders/1XGhSibCliOYjynWwBJfsvNyaxGe_CzPY?usp=sharing)

---
## Project Structure

```
MQTT-Edge-Threat-Detection/
├── utils/
│   └── evaluation.py          # Shared: timing_stats, evaluate_with_cv_timing,
│                              #         print_timing_report, wilcoxon_test
├── phase0_env/
│   └── document_environment.py
├── phase1_baselines/
│   ├── reproduce_paper4.py
│   ├── reproduce_paper5.py
│   ├── reproduce_paper26.py
│   └── our_approach.py
├── phase2_external_validation/
│   ├── validate_ids2020.py
│   └── validate_mqttset.py
├── models/                    # Saved .joblib model files (git-ignored)
├── requirements.txt
└── README.md
```

---
## Setup

```bash
git clone https://github.com/Hamza-ALLAGA/MQTT-Edge-Threat-Detection.git
cd MQTT-Edge-Threat-Detection
pip install -r requirements.txt
```

---
## Data Availability and Provenance
Datasets are **not included** in this repository due to size and licensing constraints. Please download the datasets from the shared drive link above and place the following CSV files in the project root before running the Phase 1 scripts:

| File | Used by |
|---|---|
| `R2_Prepared_Train_[4].csv` | `reproduce_paper4.py` |
| `R2_Prepared_Test_[4].csv` | `reproduce_paper4.py` |
| `OriginalDatasetCleanV11_[5].csv` | `reproduce_paper5.py` |
| `Dadataset_Paper_2026_[26].csv` | `reproduce_paper26.py` |
| `Our_Approache.csv` | `our_approach.py` |

For **Phase 2 (external validation)**, pre-processed `.npz` and `.joblib` files should be placed in the directories specified at the top of each validation script.

---
## Timing Protocol (R4C2)
All timing follows the protocol described in the manuscript:

- **Timer**: `time.perf_counter()` (nanosecond resolution)
- **Warm-up**: First 10 inference passes discarded per model
- **Batch size**: Full test fold for accuracy; fixed 13,000-sample subsample for latency in our approach
- **Reporting**: mean ± std across k=10 folds, 95% CI (Student’s t), throughput (samples/sec)
- **Significance**: paired Wilcoxon signed-rank test, α = 0.05

Run the environment documentation script to record hardware/software context:

```bash
python phase0_env/document_environment.py
```

---
## Running the Experiments

```bash
# Phase 0 — document environment
python phase0_env/document_environment.py

# Phase 1 — baselines
python phase1_baselines/reproduce_paper4.py
python phase1_baselines/reproduce_paper5.py
python phase1_baselines/reproduce_paper26.py
python phase1_baselines/our_approach.py

# Phase 2 — external validation
python phase2_external_validation/validate_ids2020.py
python phase2_external_validation/validate_mqttset.py
```

---
## Expected Outputs
Each script reports classification metrics and timing summaries to stdout. Where applicable, the scripts save trained models under `models/` (git-ignored) and may emit serialized artifacts noted in the script headers.

---
## Citation
If you use this code, please cite the manuscript:

```
@article{Allaga2026,
  author  = {Allaga, Hamza and Farchane, Abderrazak and Biniz, Mohamed},
  title   = {Intelligent threat detection for {MQTT} protocol in resource constrained edge computing},
  journal = {Pervasive and Mobile Computing},
  year    = {2026},
  publisher = {Elsevier}
}
```

A BibTeX entry will be added when the article is published.

---
## License
Code released for reproducibility purposes in support of peer review.  
Contact the corresponding author for further inquiries.
