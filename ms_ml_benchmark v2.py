#!/usr/bin/env python3
# ================================================================
# MS DIAGNOSTIC BENCHMARK TOOL
# ================================================================
# Author: Adam Simson
# Affiliation: Synthica Research Group
# Paper: "Evaluating Machine Learning Algorithms for Blood RNA-Based
#         Multiple Sclerosis Diagnosis: A Systematic Benchmark with
#         a Novel Weighted Scoring System"
#
# DESCRIPTION:
# This tool allows researchers to test their own machine learning
# models against the benchmark established in the paper above.
# Simply add your model in the section marked below, run the script,
# and see how your model compares to the published benchmark.
# ================================================================

from __future__ import annotations

# Redirect stderr to suppress all warnings (including Cython warnings from sklearn)
import sys
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# Suppress warnings at the source level
os.environ["PYTHONWARNINGS"] = "ignore"
sys.stderr = open(os.devnull, 'w')

# Suppress all Python warnings
import warnings as _warnings
_warnings.filterwarnings("ignore")
_warnings.filterwarnings("ignore", category=RuntimeWarning)
_warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")
_warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model.*")
_warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.*")
_warnings.simplefilter("ignore")

# Suppress numpy floating-point warnings
import numpy as np
np.seterr(all='ignore')
import numpy.core as _core
if hasattr(_core, 'seterr'):
    _core.seterr(all='ignore')
if hasattr(_core.numerictypes, 'seterr'):
    _core.numerictypes.seterr(all='ignore')

import logging
logging.getLogger("joblib").setLevel(logging.ERROR)
logging.getLogger("loky").setLevel(logging.ERROR)

import argparse
import gzip
import json
import platform
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib.util
import joblib
try:
    from tqdm import tqdm as _tqdm
    # Configure tqdm to use stdout so progress bar shows even when stderr is redirected
    def tqdm(*args, **kwargs):
        return _tqdm(*args, **kwargs, file=sys.stdout)
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*leaked.*")
warnings.filterwarnings("ignore", message=".*joblib_memmapping.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*")
warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
warnings.simplefilter("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

@dataclass(frozen=True)
class Config:
    random_seed: int = 42
    test_size: float = 0.20
    outer_cv_folds: int = 5
    inner_cv_folds: int = 3
    n_bootstrap: int = 2000
    top_variance: int = 1000
    k_best_features: int = 200
    smote_k_neighbors: int = 3
    n_jobs: int = -1
    data_file: str = "GSE17048_series_matrix.txt.gz"
    results_dir: str = "results_publication_grade"
    positive_label_name: str = "MS"
    negative_label_name: str = "HC"
    external_models: Optional[str] = None
    install_deps: bool = False
    requirements_file: str = "requirements.txt"


CONFIG = Config()

CLI_BANNER = '''\
\033[95m════════════════════════════════════════════════════════════════════════════════\033[0m
\033[96m  Multiple Scholarsis Machine Learning Benchmark\033[0m
\033[93m  External model benchmarking for blood RNA MS classification\033[0m
\033[95m════════════════════════════════════════════════════════════════════════════════\033[0m
'''

# Clinically motivated weighted score. Keep this as exploratory unless weights
# are validated by clinicians or sensitivity analyses.
WEIGHTS = {
    "AUC_ROC": 0.25,
    "PR_AUC": 0.15,
    "Sensitivity": 0.25,
    "Specificity": 0.15,
    "F1": 0.10,
    "Calibration": 0.10,
}


def install_requirements(requirements_path: Path) -> None:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
    print("Installing required dependencies from requirements.txt...")
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "--disable-pip-version-check", "-q"]
    subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class suppress_output:
    def __enter__(self):
        self._stdout = open(os.devnull, "w")
        self._stderr = open(os.devnull, "w")
        self._stdout_ctx = redirect_stdout(self._stdout)
        self._stderr_ctx = redirect_stderr(self._stderr)
        self._stdout_ctx.__enter__()
        self._stderr_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stderr_ctx.__exit__(exc_type, exc, tb)
        self._stdout_ctx.__exit__(exc_type, exc, tb)
        self._stderr.close()
        self._stdout.close()


# ================================================================
# CUSTOM TRANSFORMERS
# ================================================================

class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Select the top-k highest-variance features inside CV folds.

    This replaces feature selection done on the whole dataset, which would leak
    test-fold information. It is intentionally unsupervised.
    """

    def __init__(self, k: int = 1000):
        self.k = k

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        variances = np.nanvar(X, axis=0)
        k = min(self.k, X.shape[1])
        self.selected_idx_ = np.argsort(variances)[-k:]
        self.selected_idx_.sort()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self.selected_idx_]

    def get_support(self) -> np.ndarray:
        return self.selected_idx_


# ================================================================
# DATA LOADING
# ================================================================

def _find_first_line_index(lines: List[str], token: str) -> int:
    for i, line in enumerate(lines):
        if token in line:
            return i
    raise ValueError(f"Could not find required token in GEO matrix: {token}")


def _extract_conditions_from_geo_metadata(lines: List[str], n_samples: int) -> List[str]:
    """Extract condition labels from GEO metadata with robust fallbacks."""
    candidate_tokens = [
        "!Sample_characteristics_ch1",
        "!Sample_title",
        "!Sample_source_name_ch1",
    ]

    metadata_lines = [line for line in lines if any(tok in line for tok in candidate_tokens)]
    sample_strings: List[str] = ["" for _ in range(n_samples)]

    for line in metadata_lines:
        parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")[1:]]
        if len(parts) != n_samples:
            continue
        for i, p in enumerate(parts):
            sample_strings[i] += " " + p

    conditions = []
    for text in sample_strings:
        lower = text.lower()
        if "healthy" in lower or "control" in lower or lower.strip() == "hc":
            conditions.append("HC")
        elif "relapsing" in lower or "rr" in text or "rrms" in lower:
            conditions.append("RRMS")
        elif "primary" in lower or "pp" in text or "ppms" in lower:
            conditions.append("PPMS")
        elif "secondary" in lower or "sp" in text or "spms" in lower:
            conditions.append("SPMS")
        elif "multiple sclerosis" in lower or "ms" in lower:
            conditions.append("MS")
        else:
            conditions.append("Other")

    # Fallback for the known GSE17048 formatting used in the original script.
    if all(c == "Other" for c in conditions):
        try:
            label_line = lines[33]
            labels = [l.strip('"') for l in label_line.strip().split("\t")[1:]]
            conditions = []
            for label in labels:
                lower = label.lower()
                if "healthy" in lower:
                    conditions.append("HC")
                elif "RR" in label:
                    conditions.append("RRMS")
                elif "PP" in label:
                    conditions.append("PPMS")
                elif "SP" in label:
                    conditions.append("SPMS")
                else:
                    conditions.append("Other")
        except Exception:
            pass

    if len(conditions) != n_samples:
        raise ValueError("Could not infer sample conditions reliably from GEO metadata.")
    return conditions


def load_geo_series_matrix(filepath: str | Path) -> pd.DataFrame:
    """Load a GEO series matrix into a samples x genes dataframe.

    Returns columns: Condition, Label, plus expression features.
    Label: 0 = healthy control, 1 = MS case.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            "Download from NCBI GEO accession GSE17048 and place the .txt.gz file here."
        )

    print(f"Loading GEO series matrix: {filepath}")
    opener = gzip.open if filepath.suffix == ".gz" else open
    with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    table_start = _find_first_line_index(lines, "!series_matrix_table_begin") + 1
    table_end = _find_first_line_index(lines, "!series_matrix_table_end")

    header = [h.strip().strip('"') for h in lines[table_start].rstrip("\n").split("\t")]
    sample_ids = header[1:]
    n_samples = len(sample_ids)
    conditions = _extract_conditions_from_geo_metadata(lines, n_samples)

    gene_ids: List[str] = []
    data: List[List[float]] = []
    for line in lines[table_start + 1:table_end]:
        parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
        if len(parts) != n_samples + 1:
            continue
        gene_id = parts[0]
        try:
            values = [float(x) if x not in {"", "NA", "NaN"} else np.nan for x in parts[1:]]
        except ValueError:
            continue
        gene_ids.append(gene_id)
        data.append(values)

    if not data:
        raise ValueError("No numeric expression rows were parsed from the GEO matrix.")

    expression = pd.DataFrame(np.array(data).T, columns=gene_ids)
    expression.insert(0, "Condition", conditions)
    expression.insert(1, "Label", [0 if c == "HC" else 1 for c in conditions])

    # Remove ambiguous samples if any could not be classified.
    before = len(expression)
    expression = expression[expression["Condition"] != "Other"].reset_index(drop=True)
    removed = before - len(expression)

    print(f"Parsed {expression.shape[0]} samples x {expression.shape[1] - 2:,} features")
    print(f"Condition counts: {expression['Condition'].value_counts().to_dict()}")
    if removed:
        print(f"Removed {removed} samples with unclear labels.")
    print(f"Binary label counts: {expression['Label'].value_counts().to_dict()}  (0=HC, 1=MS)")
    return expression


def get_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if c not in {"Condition", "Label"}]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["Label"].to_numpy(dtype=int)
    return X, y, feature_cols


def load_external_models(path: Optional[str], cfg: Config) -> Tuple[Dict[str, Tuple[Any, Dict[str, List[Any]]]], Dict[str, Any]]:
    """Load external models from a Python module or a directory of joblib files.

    Returns two dicts: (to_tune, pretrained)
    - to_tune: name -> (estimator (class or unfitted instance), param_grid dict or None)
    - pretrained: name -> fitted estimator (will be evaluated as-is)
    """
    to_tune: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {}
    pretrained: Dict[str, Any] = {}
    if not path:
        return to_tune, pretrained

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External models path not found: {p}")

    # Python module that defines EXTERNAL_MODELS = {name: estimator or (estimator, param_grid)}
    if p.is_file() and p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("external_models_module", str(p))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        models = getattr(module, "EXTERNAL_MODELS", None)
        if models is None or not isinstance(models, dict):
            raise ValueError("Python external models file must define EXTERNAL_MODELS dict")
        for name, val in models.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                est, grid = val
            else:
                est, grid = val, {}
            to_tune[name] = (est, grid or {})
        return to_tune, pretrained

    # Directory: load any joblib/pkl files as pretrained models
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {".joblib", ".pkl"}:
                try:
                    m = joblib.load(str(f))
                    pretrained[f.stem] = m
                except Exception as e:
                    print(f"Failed to load pretrained model {f}: {e}")
        return to_tune, pretrained

    # Single file that's a pickle of a model
    if p.is_file() and p.suffix.lower() in {".joblib", ".pkl"}:
        m = joblib.load(str(p))
        pretrained[p.stem] = m
        return to_tune, pretrained

    raise ValueError("Unsupported external models path format. Use a .py module, a directory of .joblib/.pkl files, or a single .pkl/.joblib file.")


# ================================================================
# MODELS AND PIPELINES
# ================================================================

def make_preprocessing_pipeline(model: Any, cfg: Config) -> ImbPipeline:
    """Leakage-safe ML pipeline.

    Order matters:
    - imputation, variance selection, scaling, supervised feature selection are
      fitted only on training folds.
    - SMOTE is applied only to training folds via imblearn Pipeline.
    """
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("topvar", TopVarianceSelector(k=cfg.top_variance)),
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(score_func=f_classif, k=cfg.k_best_features)),
            ("smote", SMOTE(random_state=cfg.random_seed, k_neighbors=cfg.smote_k_neighbors)),
            ("model", model),
        ]
    )


def model_grid_registry(cfg: Config) -> Dict[str, Tuple[ImbPipeline, Dict[str, List[Any]]]]:
    registry: Dict[str, Tuple[ImbPipeline, Dict[str, List[Any]]]] = {
        "Logistic Regression": (
            make_preprocessing_pipeline(
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_seed), cfg
            ),
            {
                "kbest__k": [50, 100, 200],
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        ),
        "SVM": (
            make_preprocessing_pipeline(
                SVC(probability=True, class_weight="balanced", random_state=cfg.random_seed), cfg
            ),
            {
                "kbest__k": [50, 100, 200],
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale"],
            },
        ),
        "Random Forest": (
            make_preprocessing_pipeline(
                RandomForestClassifier(class_weight="balanced", random_state=cfg.random_seed), cfg
            ),
            {
                "kbest__k": [100, 200],
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "Gradient Boosting": (
            make_preprocessing_pipeline(GradientBoostingClassifier(random_state=cfg.random_seed), cfg),
            {
                "kbest__k": [100, 200],
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "KNN": (
            make_preprocessing_pipeline(KNeighborsClassifier(), cfg),
            {
                "kbest__k": [50, 100, 200],
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Naive Bayes": (
            make_preprocessing_pipeline(GaussianNB(), cfg),
            {
                "kbest__k": [50, 100, 200],
                "model__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],
            },
        ),
        "Neural Network": (
            make_preprocessing_pipeline(MLPClassifier(max_iter=2000, random_state=cfg.random_seed), cfg),
            {
                "kbest__k": [50, 100, 200],
                "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "model__alpha": [1e-4, 1e-3, 1e-2],
                "model__learning_rate": ["constant", "adaptive"],
            },
        ),
    }

    if HAS_XGBOOST:
        registry["XGBoost"] = (
            make_preprocessing_pipeline(
                XGBClassifier(
                    random_state=cfg.random_seed,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    n_jobs=1,
                ),
                cfg,
            ),
            {
                "kbest__k": [100, 200],
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        )
    else:
        print("XGBoost not installed. Skipping XGBoost model.")

    return registry


# ================================================================
# METRICS
# ================================================================

def safe_predict_proba(estimator: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return estimator.predict(X).astype(float)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def compute_ms_research_score(metrics: Dict[str, float]) -> float:
    """Exploratory composite score out of 100.

    Calibration component is computed as 1 - Brier, so higher is better.
    """
    score = (
        metrics["AUC_ROC"] * 100 * WEIGHTS["AUC_ROC"]
        + metrics["PR_AUC"] * 100 * WEIGHTS["PR_AUC"]
        + metrics["Sensitivity"] * 100 * WEIGHTS["Sensitivity"]
        + metrics["Specificity"] * 100 * WEIGHTS["Specificity"]
        + metrics["F1"] * 100 * WEIGHTS["F1"]
        + max(0.0, (1.0 - metrics["Brier"])) * 100 * WEIGHTS["Calibration"]
    )
    return round(float(score), 2)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC_ROC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan,
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Brier": brier_score_loss(y_true, y_proba),
    }
    metrics["MS_Research_Score"] = compute_ms_research_score(metrics)
    return metrics


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    random_seed: int,
    ci: float = 0.95,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    scores: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_func(y_true[idx], y_pred[idx], y_proba[idx]))
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


# ================================================================
# BENCHMARKING
# ================================================================

def nested_cv_evaluate_model(
    name: str,
    pipeline: ImbPipeline,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Nested CV: outer loop estimates performance; inner loop tunes parameters."""
    outer_cv = StratifiedKFold(n_splits=cfg.outer_cv_folds, shuffle=True, random_state=cfg.random_seed)
    inner_cv = StratifiedKFold(n_splits=cfg.inner_cv_folds, shuffle=True, random_state=cfg.random_seed)

    fold_rows: List[Dict[str, Any]] = []
    all_true: List[int] = []
    all_pred: List[int] = []
    all_proba: List[float] = []

    start = time.time()
    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=cfg.n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        y_proba = safe_predict_proba(best_estimator, X_valid)
        y_pred = (y_proba >= 0.5).astype(int)
        fold_metrics = evaluate_predictions(y_valid, y_pred, y_proba)
        fold_metrics.update(
            {
                "Model": name,
                "Fold": fold,
                "Best_Params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        fold_rows.append(fold_metrics)
        all_true.extend(y_valid.tolist())
        all_pred.extend(y_pred.tolist())
        all_proba.extend(y_proba.tolist())

    elapsed = time.time() - start
    y_true_arr = np.array(all_true)
    y_pred_arr = np.array(all_pred)
    y_proba_arr = np.array(all_proba)
    pooled = evaluate_predictions(y_true_arr, y_pred_arr, y_proba_arr)

    # Confidence intervals on pooled out-of-fold predictions.
    ci_auc = bootstrap_metric_ci(
        y_true_arr, y_pred_arr, y_proba_arr,
        lambda yt, yp, pr: roc_auc_score(yt, pr),
        cfg.n_bootstrap, cfg.random_seed,
    )
    ci_sens = bootstrap_metric_ci(
        y_true_arr, y_pred_arr, y_proba_arr,
        lambda yt, yp, pr: recall_score(yt, yp, zero_division=0),
        cfg.n_bootstrap, cfg.random_seed,
    )
    ci_spec = bootstrap_metric_ci(
        y_true_arr, y_pred_arr, y_proba_arr,
        lambda yt, yp, pr: specificity_score(yt, yp),
        cfg.n_bootstrap, cfg.random_seed,
    )

    summary = {
        "Model": name,
        **pooled,
        "AUC_95CI": f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]",
        "Sensitivity_95CI": f"[{ci_sens[0]:.3f}, {ci_sens[1]:.3f}]",
        "Specificity_95CI": f"[{ci_spec[0]:.3f}, {ci_spec[1]:.3f}]",
        "Runtime_sec": round(elapsed, 2),
        "Outer_CV_Folds": cfg.outer_cv_folds,
        "Inner_CV_Folds": cfg.inner_cv_folds,
    }
    return summary, pd.DataFrame(fold_rows)


def fit_final_model(
    name: str,
    pipeline: ImbPipeline,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> GridSearchCV:
    inner_cv = StratifiedKFold(n_splits=cfg.inner_cv_folds, shuffle=True, random_state=cfg.random_seed)
    search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_holdout(best_search: GridSearchCV, X_test: np.ndarray, y_test: np.ndarray, cfg: Config) -> Dict[str, Any]:
    estimator = best_search.best_estimator_
    y_proba = safe_predict_proba(estimator, X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    ci_auc = bootstrap_metric_ci(
        y_test, y_pred, y_proba,
        lambda yt, yp, pr: roc_auc_score(yt, pr),
        cfg.n_bootstrap, cfg.random_seed,
    )
    metrics["AUC_95CI"] = f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]"
    metrics["Best_Params"] = json.dumps(best_search.best_params_, sort_keys=True)
    return metrics


def build_stacking_model(final_estimators: Dict[str, Any], cfg: Config) -> Optional[StackingClassifier]:
    required = ["Logistic Regression", "SVM", "Random Forest"]
    if not all(k in final_estimators for k in required):
        return None
    estimators = [
        ("lr", clone(final_estimators["Logistic Regression"].best_estimator_)),
        ("svm", clone(final_estimators["SVM"].best_estimator_)),
        ("rf", clone(final_estimators["Random Forest"].best_estimator_)),
    ]
    if "XGBoost" in final_estimators:
        estimators.append(("xgb", clone(final_estimators["XGBoost"].best_estimator_)))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_seed),
        cv=cfg.inner_cv_folds,
        n_jobs=cfg.n_jobs,
        passthrough=False,
    )


# ================================================================
# VISUALISATION AND OUTPUTS
# ================================================================

def save_roc_plot(final_results: pd.DataFrame, predictions: Dict[str, Dict[str, np.ndarray]], outdir: Path) -> None:
    plt.figure(figsize=(9, 7))
    for name, pred in predictions.items():
        y_true, y_proba = pred["y_true"], pred["y_proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Holdout ROC Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "holdout_roc_curves.png", dpi=300)
    plt.close()


def save_leaderboard_plot(results: pd.DataFrame, outdir: Path) -> None:
    plot_df = results.sort_values("MS_Research_Score", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Model"], plot_df["MS_Research_Score"])
    plt.xlabel("MS Research Score / 100")
    plt.title("Model Leaderboard — Exploratory Composite Score")
    for i, score in enumerate(plot_df["MS_Research_Score"]):
        plt.text(score + 0.3, i, f"{score:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(outdir / "ms_research_score_leaderboard.png", dpi=300)
    plt.close()


def save_metric_heatmap(results: pd.DataFrame, outdir: Path) -> None:
    metrics = ["AUC_ROC", "PR_AUC", "Sensitivity", "Specificity", "F1", "Brier"]
    hm = results.set_index("Model")[metrics].copy()
    hm["Brier"] = 1 - hm["Brier"]  # higher is better for visual comparison
    plt.figure(figsize=(10, max(5, 0.45 * len(hm))))
    im = plt.imshow(hm.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Score, except Brier shown as 1 - Brier")
    plt.xticks(np.arange(len(metrics)), ["AUC", "PR-AUC", "Sens", "Spec", "F1", "1-Brier"], rotation=30, ha="right")
    plt.yticks(np.arange(len(hm.index)), hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            plt.text(j, i, f"{hm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Holdout Performance Heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "holdout_metrics_heatmap.png", dpi=300)
    plt.close()


def save_run_metadata(cfg: Config, outdir: Path) -> None:
    metadata = {
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "packages_note": "For exact package versions, run: pip freeze > requirements-lock.txt",
        "score_weights": WEIGHTS,
        "interpretation_warning": (
            "MS_Research_Score is exploratory. It is not a validated clinical diagnostic score. "
            "Use external validation before making biological or clinical claims."
        ),
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ================================================================
# MAIN
# ================================================================

def run(cfg: Config) -> None:
    print(CLI_BANNER)
    if cfg.install_deps:
        try:
            install_requirements(Path(cfg.requirements_file))
        except Exception:
            print("Automatic dependency installation failed. Please run: python -m pip install -r requirements.txt")
            return

    print("Preparing benchmark. Please wait...")
    for _ in tqdm(range(40), desc="Loading", ncols=70, bar_format="{l_bar}{bar}| {elapsed}"):
        time.sleep(0.03)

    np.random.seed(cfg.random_seed)
    outdir = Path(cfg.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        save_run_metadata(cfg, outdir)
        df = load_geo_series_matrix(cfg.data_file)
        X, y, feature_names = get_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            stratify=y,
            random_state=cfg.random_seed,
        )
        registry = model_grid_registry(cfg)
        ext_to_tune, ext_pretrained = {}, {}
        if cfg.external_models:
            try:
                ext_to_tune, ext_pretrained = load_external_models(cfg.external_models, cfg)
            except Exception:
                ext_to_tune, ext_pretrained = {}, {}

        if ext_to_tune:
            for name, (estimator, grid) in ext_to_tune.items():
                pipeline = estimator if hasattr(estimator, "named_steps") else make_preprocessing_pipeline(estimator, cfg)
                registry[name] = (pipeline, grid or {})

        nested_summaries: List[Dict[str, Any]] = []
        fold_tables: List[pd.DataFrame] = []
        final_searches: Dict[str, GridSearchCV] = {}
        holdout_rows: List[Dict[str, Any]] = []
        holdout_predictions: Dict[str, Dict[str, np.ndarray]] = {}

        total_models = len(registry)
        print("\n\033[93m🔄 Training and evaluating models...\033[0m", flush=True)
        for i, (name, (pipeline, grid)) in enumerate(registry.items(), 1):
            print(f"\n\033[96m[{i}/{total_models}]\033[0m Training: {name}", flush=True)
            summary, folds = nested_cv_evaluate_model(name, pipeline, grid, X_train, y_train, cfg)
            nested_summaries.append(summary)
            fold_tables.append(folds)
            print(f"  \033[92m✓\033[0m Nested CV complete. Score: {summary['MS_Research_Score']:.3f}", flush=True)
            search = fit_final_model(name, pipeline, grid, X_train, y_train, cfg)
            final_searches[name] = search
            holdout = evaluate_holdout(search, X_test, y_test, cfg)
            holdout["Model"] = name
            holdout_rows.append(holdout)
            y_proba = safe_predict_proba(search.best_estimator_, X_test)
            holdout_predictions[name] = {
                "y_true": y_test.copy(),
                "y_proba": y_proba.copy(),
                "y_pred": (y_proba >= 0.5).astype(int),
            }
            print(f"  \033[92m✓\033[0m Holdout AUC: {holdout['AUC_ROC']:.3f}", flush=True)

        if ext_pretrained:
            for name, est in ext_pretrained.items():
                y_proba = safe_predict_proba(est, X_test)
                y_pred = (y_proba >= 0.5).astype(int)
                metrics = evaluate_predictions(y_test, y_pred, y_proba)
                metrics["Model"] = name
                metrics["AUC_95CI"] = "not computed"
                metrics["Best_Params"] = "pretrained external model (no tuning)"
                holdout_rows.append(metrics)
                holdout_predictions[name] = {
                    "y_true": y_test.copy(),
                    "y_proba": y_proba.copy(),
                    "y_pred": y_pred.copy(),
                }

        stack = build_stacking_model(final_searches, cfg)
        if stack:
            print("\n\033[96m🔄\033[0m Training Stacking Ensemble...", flush=True)
            stack.fit(X_train, y_train)
            y_proba = safe_predict_proba(stack, X_test)
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = evaluate_predictions(y_test, y_pred, y_proba)
            metrics["Model"] = "Stacking Ensemble"
            # Compute bootstrap AUC confidence interval for stacking ensemble
            ci_auc = bootstrap_metric_ci(
                y_test, y_pred, y_proba,
                lambda yt, yp, pr: roc_auc_score(yt, pr),
                cfg.n_bootstrap, cfg.random_seed,
            )
            metrics["AUC_95CI"] = f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]"
            metrics["Best_Params"] = "base models already tuned on training set"
            holdout_rows.append(metrics)
            holdout_predictions["Stacking Ensemble"] = {
                "y_true": y_test.copy(),
                "y_proba": y_proba.copy(),
                "y_pred": y_pred.copy(),
            }

        nested_df = pd.DataFrame(nested_summaries).sort_values("MS_Research_Score", ascending=False)
        folds_df = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
        holdout_df = pd.DataFrame(holdout_rows).sort_values("MS_Research_Score", ascending=False)
        nested_df.to_csv(outdir / "nested_cv_summary.csv", index=False)
        folds_df.to_csv(outdir / "nested_cv_fold_results.csv", index=False)
        holdout_df.to_csv(outdir / "holdout_test_results.csv", index=False)
        pred_rows = []
        for name, pred in holdout_predictions.items():
            for i, (yt, yp, pr) in enumerate(zip(pred["y_true"], pred["y_pred"], pred["y_proba"])):
                pred_rows.append({
                    "Model": name,
                    "Sample_Index_In_Holdout": i,
                    "True_Label": yt,
                    "Predicted_Label": yp,
                    "Predicted_Probability_MS": pr,
                })
        pd.DataFrame(pred_rows).to_csv(outdir / "holdout_predictions.csv", index=False)
        save_leaderboard_plot(holdout_df, outdir)
        save_metric_heatmap(holdout_df, outdir)
        save_roc_plot(holdout_df, holdout_predictions, outdir)
    except Exception:
        with open(outdir / "benchmark_error.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print("Benchmark failed silently. See benchmark_error.log for details.")
        return

    print("\n" + "="*100)
    print("  BENCHMARK RESULTS")
    print("="*100)
    
    # Round numeric columns for cleaner display
    display_df = holdout_df.copy()
    for col in display_df.columns:
        if col not in ["Model", "AUC_95CI", "Best_Params", "Balanced_Accuracy"] and display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].round(3)
    
    # Remove columns that cause formatting issues
    cols_to_drop = ["Best_Params", "Balanced_Accuracy"]
    display_df = display_df.drop(columns=[c for c in cols_to_drop if c in display_df.columns])
    
    # Reorder columns: Model first, MS_Research_Score second, then rest
    cols = list(display_df.columns)
    if "Model" in cols and "MS_Research_Score" in cols:
        cols.remove("Model")
        cols.remove("MS_Research_Score")
        display_df = display_df[["Model", "MS_Research_Score"] + cols]
    
    # Print with proper column spacing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print("\n" + display_df.to_string(index=False))
    print("\n" + "="*100)
    print(f"\nSaved all outputs to: {outdir.resolve()}")
    print("\nPaper wording note: call this 'MS classification from blood RNA profiles', not validated clinical diagnosis.")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Publication-grade MS blood RNA ML benchmark.")
    parser.add_argument("--data-file", default=CONFIG.data_file, help="Path to GSE17048_series_matrix.txt.gz")
    parser.add_argument("--results-dir", default=CONFIG.results_dir, help="Output directory")
    parser.add_argument("--test-size", type=float, default=CONFIG.test_size)
    parser.add_argument("--outer-cv-folds", type=int, default=CONFIG.outer_cv_folds)
    parser.add_argument("--inner-cv-folds", type=int, default=CONFIG.inner_cv_folds)
    parser.add_argument("--n-bootstrap", type=int, default=CONFIG.n_bootstrap)
    parser.add_argument("--top-variance", type=int, default=CONFIG.top_variance)
    parser.add_argument("--k-best-features", type=int, default=CONFIG.k_best_features)
    parser.add_argument("--n-jobs", type=int, default=CONFIG.n_jobs)
    parser.add_argument("--external-models", default=None, help="Path to external models (.py module or dir of .joblib/.pkl files)")
    parser.add_argument("--install-deps", action="store_true", help="Install packages from requirements.txt before running the benchmark")
    parser.add_argument("--requirements-file", default=CONFIG.requirements_file, help="Requirements file to use when --install-deps is enabled")
    args = parser.parse_args()
    return Config(
        data_file=args.data_file,
        results_dir=args.results_dir,
        test_size=args.test_size,
        outer_cv_folds=args.outer_cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        n_bootstrap=args.n_bootstrap,
        top_variance=args.top_variance,
        k_best_features=args.k_best_features,
        n_jobs=args.n_jobs,
        external_models=args.external_models,
        install_deps=args.install_deps,
        requirements_file=args.requirements_file,
    )


if __name__ == "__main__":
    run(parse_args())
