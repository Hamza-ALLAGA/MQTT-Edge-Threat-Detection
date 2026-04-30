"""
phase1_baselines/reproduce_paper26.py
========================================
PHASE 1 — Baseline Reproduction: Paper [26]  (2026)
R4C2 Update: 10-fold stratified CV + SMOTE applied per-fold (no leakage).

Dataset file expected
---------------------
  Dadataset_Paper_2026_[26].csv

Place it in the project root, or adjust the path below.

Usage
-----
    pip install imbalanced-learn
    python phase1_baselines/reproduce_paper26.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    timing_stats,
    print_timing_report,
    RANDOM_SEED,
    N_FOLDS,
    N_WARMUP,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE = "Dadataset_Paper_2026_[26].csv"


def hex_to_int(hex_val) -> int:
    try:
        return int(str(hex_val), 16)
    except Exception:
        return 0


def load_and_prepare() -> tuple:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.dropna(subset=["label"], inplace=True)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    for col in ["tcp_flags", "mqtt_conack_flags", "mqtt_conflags", "mqtt_hdrflags"]:
        if col in df.columns:
            df[col] = df[col].apply(hex_to_int)

    df.fillna(0, inplace=True)

    X_raw = df.drop(columns=["label"]).values
    y     = df["label"].values

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    print(f"[R4C2] Paper [26] — Full CV pool: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"[R4C2] SMOTE oversampling applied INSIDE each CV fold (no leakage)")
    print(f"[R4C2] ~{X.shape[0] // N_FOLDS:,} samples per test fold")
    return X, y


def build_models() -> list:
    """Paper [26] model suite."""
    return [
        ("Decision Tree (Slow) [26]", Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("tree", DecisionTreeClassifier(
                criterion="entropy", splitter="best",
                max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features=None,
                random_state=42)),
        ])),
        ("CART [26]",              DecisionTreeClassifier(criterion="gini", random_state=42)),
        ("Random Forest [26]",     RandomForestClassifier(n_estimators=100, random_state=42)),
        ("KNN [26]",               KNeighborsClassifier(n_neighbors=5)),
        ("Gradient Boosting [26]", GradientBoostingClassifier(random_state=42)),
    ]


def evaluate_with_smote_cv(name: str, model, X: np.ndarray, y: np.ndarray) -> dict:
    """10-fold CV with SMOTE applied inside each fold to prevent data leakage."""
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds  = list(skf.split(X, y))

    # Warm-up on fold 0 (SMOTE + fit + N_WARMUP silent predictions)
    tr0, te0 = folds[0]
    smote_wu = SMOTE(random_state=RANDOM_SEED)
    X_wu, y_wu = smote_wu.fit_resample(X[tr0], y[tr0])
    model.fit(X_wu, y_wu)
    for _ in range(N_WARMUP):
        model.predict(X[te0[:1]])
    print(f"  [{name}] Warm-up complete ({N_WARMUP} passes discarded).")

    fold_acc, fold_f1, fold_times, fold_sizes = [], [], [], []

    for tr_idx, te_idx in folds:
        smote_f = SMOTE(random_state=RANDOM_SEED)
        X_tr_r, y_tr_r = smote_f.fit_resample(X[tr_idx], y[tr_idx])
        model.fit(X_tr_r, y_tr_r)

        t0     = time.perf_counter()
        y_pred = model.predict(X[te_idx])
        t1     = time.perf_counter()

        fold_times.append(t1 - t0)
        fold_sizes.append(len(te_idx))
        fold_acc.append(accuracy_score(y[te_idx], y_pred))
        fold_f1.append(f1_score(y[te_idx], y_pred, average="weighted", zero_division=0))

    median_n = int(np.median(fold_sizes))
    ts = timing_stats(fold_times, median_n)

    return {
        "Model"                   : name,
        "Acc_mean"                : np.mean(fold_acc) * 100,
        "Acc_std"                 : np.std(fold_acc, ddof=1) * 100,
        "F1_mean"                 : np.mean(fold_f1) * 100,
        "F1_std"                  : np.std(fold_f1, ddof=1) * 100,
        "TestTime_mean_s"         : ts["mean_total_s"],
        "TestTime_std_s"          : ts["std_total_s"],
        "TestTime_CI95_low_s"     : ts["ci95_low_s"],
        "TestTime_CI95_high_s"    : ts["ci95_high_s"],
        "PerSample_mean_ms"       : ts["mean_per_sample_ms"],
        "PerSample_std_ms"        : ts["std_per_sample_ms"],
        "PerSample_CI95_low_ms"   : ts["ci95_low_per_sample_ms"],
        "PerSample_CI95_high_ms"  : ts["ci95_high_per_sample_ms"],
        "Throughput_mean_sps"     : ts["mean_throughput_sps"],
        "Throughput_std_sps"      : ts["std_throughput_sps"],
        "n_test_samples_per_fold" : median_n,
        "n_folds"                 : N_FOLDS,
        "_fold_test_times"        : fold_times,
        "_fold_f1"                : fold_f1,
    }


def run() -> None:
    print("=" * 70)
    print("PAPER [26] — 10-Fold CV with SMOTE (per-fold) + Timing Stats (R4C2)")
    print(f"Warm-up: first {N_WARMUP} inference passes discarded per model")
    print("=" * 70)

    X, y   = load_and_prepare()
    models = build_models()
    results = []

    for name, model in models:
        print(f"\n[Processing: {name}]")
        r = evaluate_with_smote_cv(name, model, X, y)
        results.append(r)
        print_timing_report(r)

    cols = [
        "Model", "Acc_mean", "Acc_std", "F1_mean", "F1_std",
        "PerSample_mean_ms", "PerSample_std_ms",
        "PerSample_CI95_low_ms", "PerSample_CI95_high_ms",
        "Throughput_mean_sps",
    ]
    print("\n" + "=" * 70)
    print("PAPER [26] — SUMMARY TABLE")
    print("=" * 70)
    print(pd.DataFrame(results)[cols].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    run()
