# MS Blood RNA ML Benchmark

A publication-style machine learning benchmarking framework for classifying Multiple Sclerosis (MS) using whole blood RNA expression data.

This project evaluates multiple machine learning algorithms on the publicly available GSE17048 cohort and introduces the **MS Diagnostic Score** — a clinically weighted composite evaluation metric designed to assess model utility beyond standard accuracy alone.

---

## Overview

Machine learning in biomedical research is often evaluated using isolated metrics such as accuracy or AUC-ROC. However, in real clinical settings, different metrics carry different levels of importance.

This project introduces a more clinically aware benchmarking framework by combining:

- Diagnostic discrimination
- Sensitivity to disease detection
- Precision
- Calibration quality
- Generalization performance

into a single weighted scoring system.

The framework benchmarks multiple machine learning models under a standardized evaluation pipeline with:

- Leakage-safe preprocessing
- Feature selection
- Nested cross-validation
- Bootstrap confidence intervals
- Calibration analysis
- ROC analysis
- Statistical comparison testing

---

## Models Included

The benchmark currently evaluates:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Neural Network (MLP)
- Gradient Boosting
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Stacking Ensemble

---

## Dataset

### Source

This project uses the publicly available GEO dataset:

**GSE17048 — Whole Blood RNA Expression Profiles**

NCBI GEO:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17048

### Cohort

- 144 patient samples
- Healthy controls
- Relapsing-remitting MS
- Primary progressive MS
- Secondary progressive MS

### Important

The dataset is **not included** in this repository.

To use this project:

1. Download `GSE17048_series_matrix.txt.gz`
2. Place it in the project root directory

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ms-blood-rna-ml-benchmark.git
cd ms-blood-rna-ml-benchmark
