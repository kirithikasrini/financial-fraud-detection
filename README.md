# Explainable Financial Fraud Detection Using SEC Financial Data and MD&A Text

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-008080?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen)](https://shap.readthedocs.io/)

> **Academic Context**: M.Tech Final-Year Project investigating multimodal machine learning architectures that combine structured financial-statement accounting ratios with unstructured Management's Discussion and Analysis (MD&A) textual disclosures from SEC Form 10-K filings to identify fraudulent financial reporting, explained using SHAP (SHapley Additive exPlanations).

> [!WARNING]
> **Regulatory & Risk Disclaimer**: Model outputs represent algorithmic risk estimations based on historical SEC filing characteristics and enforcement releases. A high prediction score indicates statistical anomaly/risk, **not** definitive legal or forensic proof of fraudulent reporting.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Datasets & Sources](#datasets--sources)
- [Methodology & Architecture](#methodology--architecture)
  - [Experimental Setups](#experimental-setups)
  - [Data Leakage Prevention](#data-leakage-prevention)
  - [Class Imbalance & Threshold Tuning](#class-imbalance--threshold-tuning)
- [Evaluation & Test Results](#evaluation--test-results)
- [Explainability (SHAP Analysis)](#explainability-shap-analysis)
- [Repository Structure](#repository-structure)
- [Installation & Getting Started](#installation--getting-started)
- [Execution Pipeline](#execution-pipeline)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)

---

## Project Overview

Detecting fraudulent financial reporting is a critical challenge in corporate governance, auditing, and fintech. Traditional detection systems depend primarily on quantitative accounting metrics and financial ratios (e.g., Beneish M-score, Altman Z-score), often missing qualitative early-warning indicators embedded within textual disclosures.

This project implements a multi-paradigm machine learning framework that compares:
1. **Financial-Only Models (E1)**: Tabular accounting items and engineered financial ratios.
2. **Text-Only Models (E2)**: Unstructured Management's Discussion and Analysis (MD&A) sections extracted from Form 10-K filings, vectorized via TF-IDF.
3. **Multimodal Hybrid Models (E3)**: Joint feature representations fusing structured financial indicators with dimensionality-reduced latent MD&A textual representations (Truncated SVD).
4. **Model Interpretability**: Post-hoc SHAP KernelExplainer to quantify the relative attribution between qualitative disclosure language and quantitative financial statement metrics.

---

## Key Findings

- **Textual disclosures carry dominant discriminative power**: The **E2 TF-IDF + Logistic Regression** model achieved the highest overall **ROC-AUC (0.7300)**, **PR-AUC (0.0321)**, and **Recall (0.3438)** on the unseen test set, significantly outperforming tabular financial-only baselines.
- **Hybrid synergy improves precision**: The **E3 Hybrid Logistic Regression** architecture attained the highest **Precision (0.0426)** and **F1-Score (0.0635)** among all evaluated configurations.
- **Multimodal attribution balance**: In the SHAP KernelExplainer evaluation of the **E3 Hybrid MLP**, latent text components (SVD dimensions) accounted for **~88.09%** of the summed mean absolute attribution, compared to **~11.91%** for structured financial ratios within the examined sample.

---

## Datasets & Sources

The project integrates two primary public data sources to construct a company-year panel:

| Dataset | Provider / Source | Primary Content Used |
| :--- | :--- | :--- |
| **SEC Financial Statement Data Sets** | [Kaggle Dataset](https://www.kaggle.com/datasets/dominicmalouf/sec-financial-statement-data-sets) (Dominic Malouf) | Fundamental financial statements extracted from SEC EDGAR filings (`sub.txt`, `num.txt`, etc.), yielding balance sheet, income statement, and cash flow items. |
| **10-K Fraud Detection Dataset** | [Zenodo Record 17121948](https://zenodo.org/records/17121948) | Pre-extracted MD&A sections from 10-K reports (`firm_years.json`) and fraud ground-truth labels derived from SEC Accounting and Auditing Enforcement Releases (`aaer_mark5.csv`). |

### Data Integration Pipeline

- **Entity Linking**: Records were merged using the SEC **Central Index Key (CIK)** and matching **fiscal year (`fyear`)**.
- **Dataset Scale**:
  - **31,241** company-year observations across **2,931** distinct firms.
  - Target Variable: Binary indicator ($y \in \{0, 1\}$) denoting whether the filing coincides with an identified SEC AAER fraud period.


> [!NOTE]
> In compliance with distribution guidelines, raw dataset files are not hosted inside this git repository. Please download the respective files from Kaggle and Zenodo and place them in the project data path or adjust paths in `notebook/fintech_data_merger.ipynb`.

---

## Methodology & Architecture

### Experimental Setups

| ID | Experiment | Input Modality | Evaluated Models |
| :--- | :--- | :--- | :--- |
| **E1** | Financial-only | Engineered accounting ratios & quantitative items | XGBoost, LightGBM, Multi-Layer Perceptron (MLP) |
| **E2** | MD&A-only | Textual features vectorized via TF-IDF | Logistic Regression, Multi-Layer Perceptron (MLP) |
| **E3** | Hybrid Multimodal | Financial features concatenated with SVD-reduced TF-IDF text features | Logistic Regression, Multi-Layer Perceptron (MLP) |

### Data Leakage Prevention
To prevent cross-period and intra-firm information leakage, data partitioning was performed at the **company level (by CIK)**:
- Companies allocated to the training split never appear in the validation or testing splits.
- All normalization scalers, imputation values, TF-IDF vocabularies, and Truncated SVD projection matrices were fitted **strictly on the training split** and subsequently applied to validation and test partitions.

### Class Imbalance & Threshold Tuning
Because fraudulent reporting constitutes a minuscule fraction of all corporate filings (severe positive-class rarity):
- Cost-sensitive learning and class weighting (`class_weight='balanced'`) were applied.
- Decision thresholds were optimized on the **Validation Split** to maximize the precision-recall trade-off rather than defaulting to $0.50$, then held fixed for test evaluation.

---

## Evaluation & Test Results

Given the extreme class imbalance, **PR-AUC (Precision-Recall AUC)** and the **Precision-Recall trade-off** represent the most reliable indicators of real-world audit viability.

| Experiment | Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E1: Financial-only** | XGBoost | 0.0175 | 0.1875 | 0.0321 | 0.5220 | 0.0136 |
| **E1: Financial-only** | LightGBM | 0.0139 | 0.0781 | 0.0236 | 0.5264 | 0.0118 |
| **E1: Financial-only** | Multi-Layer Perceptron (MLP) | 0.0286 | 0.2344 | 0.0510 | 0.5588 | 0.0269 |
| **E2: MD&A-only** | TF-IDF + Logistic Regression | 0.0309 | **0.3438** | 0.0566 | **0.7300** | **0.0321** |
| **E2: MD&A-only** | TF-IDF + MLP | 0.0317 | 0.0625 | 0.0421 | 0.6491 | 0.0198 |
| **E3: Hybrid Multimodal** | Hybrid Logistic Regression | **0.0426** | 0.1250 | **0.0635** | 0.5209 | 0.0161 |
| **E3: Hybrid Multimodal** | Hybrid MLP | 0.0258 | 0.1875 | 0.0454 | 0.6840 | 0.0252 |

> **Reading PR-AUC in Imbalanced Domains**: Baseline PR-AUC equals the positive class prevalence in the test set (~1-2%). A PR-AUC of 0.0321 indicates substantive uplift over random guessing under severe imbalance.

---

## Explainability (SHAP Analysis)

To prevent the multimodal neural network from acting as an opaque black-box, **SHAP (SHapley Additive exPlanations)** with `KernelExplainer` was applied to the **E3 Hybrid MLP**:

- **Relative Attribution**:
  - **Latent Text Components (SVD)**: **88.09%** of the summed mean absolute attribution.
  - **Financial Metrics & Ratios**: **11.91%** of the summed mean absolute attribution.
- **Interpretation**:
  - Linguistic choices, sentiment modulation, structural omissions, and disclosure phrasing in MD&A sections provide stronger risk signals than raw accounting ratios alone.
  - SVD components represent latent dimensional clusters rather than standalone keywords; future extensions can map these back to semantic dictionaries (e.g., Loughran-McDonald financial sentiment categories).

---

## Repository Structure

```text
financial-fraud-detection/
├── .gitignore                          # Git ignore rules for datasets, venv, and checkpoints
├── README.md                           # Comprehensive project documentation
├── requirements.txt                    # Project dependencies and environment specs
├── notebook/
│   ├── fintech_data_merger.ipynb       # Ingestion and alignment of Kaggle financial data & Zenodo 10-K text
│   ├── datasplit.ipynb                 # Leakage-free company-level train/val/test splitting
│   ├── E1_XGBoost.ipynb                # Experiment 1: XGBoost on tabular financial features
│   ├── E1_LightGBM.ipynb               # Experiment 1: LightGBM on tabular financial features
│   ├── E1_MLP_.ipynb                   # Experiment 1: Multi-Layer Perceptron on tabular financial features
│   ├── E2_MDA_TF.ipynb                 # Experiment 2: TF-IDF + Logistic Regression & MLP on MD&A text
│   ├── E3_experiment.ipynb             # Experiment 3: Multimodal Hybrid fusion (Financial + SVD Text)
│   ├── Comparative_Analysis.ipynb      # Cross-model evaluation, metrics aggregation, ROC/PR curves
│   ├── SHAP_Explainability.ipynb       # KernelExplainer SHAP attribution on Hybrid MLP
│   └── Final_Verification_ipynb.ipynb  # End-to-end verification and sanity checks
└── results/                            # Performance comparison plots and SHAP visual summary
    ├── ROC_AUC_Comparison_Clean (2).png
    ├── PR_AUC_Comparison_Clean (2).png
    ├── MLP_Precision_Comparison.png
    ├── MLP_Recall_Comparison.png
    ├── MLP_F1_Comparison.png
    └── E3_Hybrid_MLP_SHAP_Summary (3).png
```

---

## Installation & Getting Started

### Prerequisites

- **Python 3.9+** (recommended: 3.10 or 3.11)
- **JupyterLab** or **Google Colab** (with GPU acceleration for faster MLP / SVD computation)

### Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/financial-fraud-detection.git
cd financial-fraud-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Execution Pipeline

Follow the structured pipeline below to replicate results:

1. **Acquire Source Datasets**:
   - Download the SEC financial data from [Kaggle](https://www.kaggle.com/datasets/dominicmalouf/sec-financial-statement-data-sets).
   - Download the 10-K fraud labels and MD&A text from [Zenodo](https://zenodo.org/records/17121948).
2. **Data Integration**:
   - Execute `notebook/fintech_data_merger.ipynb` to match CIK and fiscal year entries and output `final_financial_mda_dataset.csv`.
3. **Partitioning**:
   - Run `notebook/datasplit.ipynb` to generate company-isolated `train.csv`, `val.csv`, and `test.csv`.
4. **Model Training & Experiments**:
   - **E1**: Run `E1_XGBoost.ipynb`, `E1_LightGBM.ipynb`, and `E1_MLP_.ipynb`.
   - **E2**: Run `E2_MDA_TF.ipynb`.
   - **E3**: Run `E3_experiment.ipynb`.
5. **Comparative Analysis & Explainability**:
   - Run `Comparative_Analysis.ipynb` to generate side-by-side performance comparison figures and PR curves.
   - Run `SHAP_Explainability.ipynb` to extract feature attributions and generate SHAP summary plots.

---

## Limitations & Ethical Considerations

- **AAER Label Incompleteness**: Fraud ground-truth is derived from formal SEC Accounting and Auditing Enforcement Releases. Undetected fraud, unprosecuted irregularities, or settled cases without formal AAERs remain labeled as non-fraud ($0$).
- **Extreme Class Imbalance**: With fraud incidents representing a very small percentage of filings, models face high false-positive rates; precision remains modest across all configurations.
- **SVD Interpretability Trade-off**: While Truncated SVD reduces memory complexity for multimodal neural nets, latent dimensions lack direct word-level readability compared to bag-of-words or attention maps.
- **Temporal Generalization**: Shifting macroeconomic conditions, changes in accounting standards (GAAP/IFRS), and evolving SEC disclosure regulations may require periodic retraining and domain adaptation.

---

## References

1. Malouf, D. *SEC Financial Statement Data Sets*. [Kaggle](https://www.kaggle.com/datasets/dominicmalouf/sec-financial-statement-data-sets).
2. *10-K Fraud Detection Dataset*. [Zenodo](https://zenodo.org/records/17121948).
3. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*.
4. Loughran, T., & McDonald, B. (2011). When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
