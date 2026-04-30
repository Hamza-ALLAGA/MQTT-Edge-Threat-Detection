"""
phase1_baselines/reproduce_paper4.py
======================================
PHASE 1 — Baseline Reproduction: Paper [4]
R4C2 Update: 10-fold stratified CV with full timing statistics
             (mean ± std, 95% CI, per-sample latency, throughput).

Dataset files expected
----------------------
  R2_Prepared_Train_[4].csv
  R2_Prepared_Test_[4].csv

Place them in the project root, or adjust the paths below.

Usage
-----
    python phase1_baselines/reproduce_paper4.py
"""

import re
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Add project root to path so utils can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    evaluate_with_cv_timing,
    print_timing_report,
    RANDOM_SEED,
    N_FOLDS,
    N_WARMUP,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "R2_Prepared_Train_[4].csv"
TEST_FILE  = "R2_Prepared_Test_[4].csv"
MODEL_DIR  = os.path.join("models", "paper4")


def load_and_prepare() -> tuple:
    """Load CSVs, encode, combine into a single CV pool."""
    df_train = pd.read_csv(TRAIN_FILE)
    df_test  = pd.read_csv(TEST_FILE)
    print(f"[✔] Train shape: {df_train.shape}")
    print(f"[✔] Test  shape: {df_test.shape}")

    # Auto-detect target column
    candidates = ["target", "label", "Label", "Target", "class", "Class", "Attack", "attack"]
    target_col = next((c for c in candidates if c in df_train.columns), df_train.columns[-1])
    print(f"[✔] Target column: '{target_col}'")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].copy()
    X_test  = df_test.drop(columns=[target_col])
    y_test  = df_test[target_col].copy()

    # Encode object feature columns
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        enc = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        enc.fit(combined)
        X_train[col] = enc.transform(X_train[col].astype(str))
        X_test[col]  = enc.transform(X_test[col].astype(str))

    # Clean inf / NaN
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # Combine into CV pool
    X_cv = np.vstack([X_train.values, X_test.values])
    y_cv = np.concatenate([y_train.values, y_test.values])

    # Drop NaN labels
    nan_mask = pd.isnull(y_cv)
    if nan_mask.any():
        print(f"⚠  Removing {nan_mask.sum()} NaN labels from CV pool")
        X_cv = X_cv[~nan_mask]
        y_cv = y_cv[~nan_mask]

    le = LabelEncoder()
    y_cv = le.fit_transform(y_cv.astype(str))

    print(f"[✔] Combined CV pool: {X_cv.shape[0]:,} samples, {X_cv.shape[1]} features")
    print(f"[R4C2] ~{X_cv.shape[0] // N_FOLDS:,} samples per test fold")

    print("\nClass distribution (FULL CV POOL):")
    for u, c in zip(*np.unique(y_cv, return_counts=True)):
        print(f"  {le.classes_[int(u)]!s:<20s} {c:>12,}")
    print(f"  {'TOTAL':<20s} {y_cv.shape[0]:>12,}")

    return X_cv, y_cv


def build_models() -> dict:
    """Paper [4] exact hyperparameters."""
    return {
        "Decision Tree [4]" : DecisionTreeClassifier(
            criterion="gini", splitter="best",
            max_depth=None, random_state=RANDOM_SEED),
        "Random Forest [4]" : RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boost [4]": GradientBoostingClassifier(
            n_estimators=20, random_state=RANDOM_SEED),
    }


def run() -> None:
    print("=" * 70)
    print("PAPER [4] — 10-Fold CV Evaluation with Timing Statistics (R4C2)")
    print(f"Warm-up: first {N_WARMUP} inference passes discarded per model")
    print("=" * 70)

    X_cv, y_cv = load_and_prepare()
    models     = build_models()

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []

    for name, model in models.items():
        print(f"\n[Processing: {name}]")
        r = evaluate_with_cv_timing(name, model, X_cv, y_cv)
        results.append(r)
        print_timing_report(r)

        # Save final model trained on full pool
        model.fit(X_cv, y_cv)
        safe_name  = re.sub(r"[^a-zA-Z0-9_]+", "_", name.lower()).strip("_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.joblib")
        joblib.dump(model, model_path)
        print(f"  Model saved → {model_path}")

    # Summary table
    cols = [
        "Model", "Acc_mean", "Acc_std", "F1_mean", "F1_std",
        "PerSample_mean_ms", "PerSample_std_ms",
        "Throughput_mean_sps", "n_test_samples_per_fold",
    ]
    df_out = pd.DataFrame(results)[cols].copy()
    df_out.columns = [
        "Model", "Acc(%)", "±Acc", "F1(%)", "±F1",
        "PerSample(ms)", "±PS", "Throughput(s/s)", "N_test",
    ]
    print("\n" + "=" * 70)
    print("PAPER [4] — FINAL SUMMARY TABLE")
    print("=" * 70)
    print(df_out.to_string(index=False, float_format="%.4f"))
    print("\n[NOTE] All timing values are means across k=10 stratified folds.")
    print("[NOTE] Per-sample latency = total_fold_time / fold_test_size.")


if __name__ == "__main__":
    run()
