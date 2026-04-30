"""
phase1_baselines/reproduce_paper5.py
======================================
PHASE 1 — Baseline Reproduction: Paper [5]
R4C2 Update: 10-fold stratified CV with full timing statistics.

Dataset file expected
---------------------
  OriginalDatasetCleanV11_[5].csv

Place it in the project root, or adjust the path below.

Usage
-----
    python phase1_baselines/reproduce_paper5.py
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
from sklearn.ensemble import RandomForestClassifier

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
DATA_FILE = "OriginalDatasetCleanV11_[5].csv"
MODEL_DIR = os.path.join("models", "paper5")

# Protocols excluded per Paper [5] methodology
PROTOCOLS_TO_AVOID = [
    "MDNS", "PORTMAP", "RIP", "DHCP", "NBNS", "XDMCP", "NTP",
    "protocol", "NFS", "CLDAP", "RADIUS", "SNMP", "ISAKMP",
    "SRVLOC", "MPEG_PAT", "protocol_DNS",
]


def load_and_prepare() -> tuple:
    df = pd.read_csv(DATA_FILE)
    print(f"[✔] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Drop metadata columns
    drop_cols = [
        "Unnamed: 0", "mqtt_flag_uname", "mqtt_flag_passwd", "mqtt_flag_retain",
        "mqtt_flag_qos", "mqtt_flag_willflag", "mqtt_flag_clean",
        "mqtt_flag_reserved", "src_ip", "dst_ip", "timestamp",
    ]
    removed = [c for c in drop_cols if c in df.columns]
    df.drop(columns=removed, inplace=True, errors="ignore")
    print(f"[2] Dropped {len(removed)} metadata columns")

    # Impute numeric columns
    for col in df.columns:
        if col != "is_attack" and df[col].dtype in ["float64", "int64"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # Remove noisy protocols
    if "protocol" in df.columns:
        before = len(df)
        df = df[~df["protocol"].isin(PROTOCOLS_TO_AVOID)]
        print(f"[5] Removed {before - len(df):,} noisy-protocol rows | Remaining: {len(df):,}")

    df.fillna(0, inplace=True)
    print(f"[✔] Preprocessing done. Final shape: {df.shape}")

    target_col = "is_attack"
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found. Columns: {list(df.columns)}")

    X_df = df.drop(columns=[target_col]).replace({"True": 1, "False": 0, True: 1, False: 0})
    for col in X_df.select_dtypes(include=["object"]).columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)

    X = X_df.values.astype(np.float64)
    y = LabelEncoder().fit_transform(df[target_col].values)

    print(f"[✔] X: {X.shape} | dtype: {X.dtype} | NaN: {np.isnan(X).any()}")
    print(f"[✔] y: {y.shape} | classes: {np.unique(y)}")
    print(f"[R4C2] ~{X.shape[0] // N_FOLDS:,} samples per test fold")
    return X, y


def build_models() -> dict:
    """Paper [5] exact hyperparameters."""
    return {
        "Decision Tree [5]": DecisionTreeClassifier(
            criterion="entropy", splitter="best", random_state=42),
        "Random Forest [5]": RandomForestClassifier(
            criterion="entropy", bootstrap=True,
            n_estimators=100, n_jobs=-1, random_state=42),
    }


def run() -> None:
    print("=" * 70)
    print("PAPER [5] — 10-Fold CV Evaluation with Timing Statistics (R4C2)")
    print(f"Warm-up: first {N_WARMUP} inference passes discarded per model")
    print("=" * 70)

    X, y   = load_and_prepare()
    models = build_models()

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []

    for name, model in models.items():
        print(f"\n[Processing: {name}]")
        r = evaluate_with_cv_timing(name, model, X, y)
        results.append(r)
        print_timing_report(r)

        model.fit(X, y)
        safe_name  = re.sub(r"[^a-zA-Z0-9_]+", "_", name.lower()).strip("_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.joblib")
        joblib.dump(model, model_path)
        print(f"  Model saved → {model_path}")

    cols = [
        "Model", "Acc_mean", "Acc_std", "F1_mean", "F1_std",
        "TestTime_mean_s", "TestTime_std_s",
        "TestTime_CI95_low_s", "TestTime_CI95_high_s",
        "PerSample_mean_ms", "PerSample_std_ms", "Throughput_mean_sps",
    ]
    df_out = pd.DataFrame(results)[cols].copy()
    df_out.columns = [
        "Model", "Acc(%)", "±Acc", "F1(%)", "±F1",
        "TestTime(s)", "±TT", "CI95_low(s)", "CI95_high(s)",
        "PerSample(ms)", "±PS", "Throughput(s/s)",
    ]
    print("\n" + "=" * 70)
    print("PAPER [5] — FINAL SUMMARY TABLE")
    print("=" * 70)
    print(df_out.to_string(index=False, float_format="%.4f"))
    print("\n[NOTE] All timing values are means across k=10 stratified folds.")
    print("[NOTE] Per-sample latency = total_fold_time / fold_test_size.")


if __name__ == "__main__":
    run()
