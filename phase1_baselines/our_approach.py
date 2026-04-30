"""
phase1_baselines/our_approach.py
==================================
PHASE 1 — Our Proposed Approach
5-Feature MQTT-IDS with ENN pre-clean, multi-pass SMOTE augmentation,
and dual-mode timing (accuracy on full fold, latency on fixed subsample).

R4C2: Timing reported as distributions (mean ± std, 95% CI, throughput).

Dataset file expected
---------------------
  Our_Approache.csv   (raw, unaugmented)

Usage
-----
    pip install imbalanced-learn
    python phase1_baselines/our_approach.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

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

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_FILE           = "Our_Approache.csv"
MODEL_DIR           = os.path.join("models", "our_approach")
LATENCY_SAMPLE_SIZE = 13_000  # fixed subsample for latency measurement

SELECTED_COLUMNS = ["timestamp", "tcp_time_delta", "tcp_len", "mqtt_msg", "tcp_flags", "label"]


def hex_to_int(val) -> int:
    try:
        return int(str(val), 16)
    except Exception:
        return 0


def load_and_augment() -> tuple:
    """Load raw dataset, apply ENN pre-clean, multi-pass SMOTE, ENN post-clean."""
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df[[c for c in SELECTED_COLUMNS if c in df.columns]]
    df.dropna(subset=["label"], inplace=True)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    for col in ["tcp_flags", "mqtt_conack_flags", "mqtt_conflags", "mqtt_hdrflags"]:
        if col in df.columns:
            df[col] = df[col].apply(hex_to_int)

    df.fillna(0, inplace=True)

    X_raw      = df.drop(columns=["label"]).values
    y_raw      = df["label"].values
    N_FEATURES = X_raw.shape[1]

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"[ORIGINAL] {X_scaled.shape[0]:,} samples | {N_FEATURES} features")

    # ENN pre-clean
    print("\n[ENN] Pre-cleaning noisy samples...")
    enn = EditedNearestNeighbours(n_neighbors=5, kind_sel="all")
    X_clean, y_clean = enn.fit_resample(X_scaled, y_raw)
    print(f"[ENN] Removed {X_scaled.shape[0] - X_clean.shape[0]:,} | "
          f"Clean: {X_clean.shape[0]:,} samples")

    # Multi-pass SMOTE augmentation
    print("\n[SMOTE] Augmenting...")
    X_aug, y_aug = X_clean.copy(), y_clean.copy()
    smote_passes = [
        (3,  RANDOM_SEED),
        (5,  RANDOM_SEED + 11),
        (7,  RANDOM_SEED + 22),
        (10, RANDOM_SEED + 33),
        (5,  RANDOM_SEED + 44),
    ]
    for i, (k, seed) in enumerate(smote_passes, 1):
        before = X_aug.shape[0]
        X_aug, y_aug = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_aug, y_aug)
        print(f"  Pass {i} (k={k:>2}): +{X_aug.shape[0] - before:,} → {X_aug.shape[0]:,}")

    # ENN post-clean
    enn2 = EditedNearestNeighbours(n_neighbors=3, kind_sel="all")
    X_aug, y_aug = enn2.fit_resample(X_aug, y_aug)
    assert X_aug.shape[1] == N_FEATURES, "Column count changed — abort!"
    print(f"\n[FINAL] {X_aug.shape[0]:,} samples | {X_aug.shape[1]} features ✔")
    print(f"[FINAL] Class dist: {dict(zip(*np.unique(y_aug, return_counts=True)))}")
    print(f"[FINAL] CV test fold ≈ {X_aug.shape[0] // N_FOLDS:,} samples")

    return X_aug, y_aug, scaler, le


def evaluate_dual_timing(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    latency_n: int = LATENCY_SAMPLE_SIZE,
) -> dict:
    """
    Dual-mode evaluation:
      - Accuracy  : computed on the full test fold (unbiased).
      - Latency   : computed on a fixed-size subsample (calibrated).
    """
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(X, y))
    rng   = np.random.default_rng(RANDOM_SEED)

    # Warm-up
    tr0, te0 = folds[0]
    model.fit(X[tr0], y[tr0])
    for _ in range(N_WARMUP):
        model.predict(X[te0[:1]])
    print(f"  [{name}] Warm-up complete ({N_WARMUP} passes discarded).")

    fold_acc, fold_f1 = [], []
    fold_times, fold_sizes = [], []

    for tr_idx, te_idx in folds:
        model.fit(X[tr_idx], y[tr_idx])

        # Accuracy on full fold
        y_pred = model.predict(X[te_idx])
        fold_acc.append(accuracy_score(y[te_idx], y_pred))
        fold_f1.append(f1_score(y[te_idx], y_pred, average="weighted", zero_division=0))

        # Timing on fixed subsample
        n_sub   = min(latency_n, len(te_idx))
        sub_idx = rng.choice(len(te_idx), size=n_sub, replace=False)
        X_sub   = X[te_idx[sub_idx]]
        t0 = time.perf_counter()
        model.predict(X_sub)
        t1 = time.perf_counter()
        fold_times.append(t1 - t0)
        fold_sizes.append(n_sub)

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


def build_models() -> list:
    return [
        ("DT Gini Full",
         DecisionTreeClassifier(criterion="gini", splitter="best",
                                max_depth=None, min_samples_leaf=1, random_state=42)),
        ("DT Entropy Full",
         DecisionTreeClassifier(criterion="entropy", splitter="best",
                                max_depth=None, min_samples_leaf=1, random_state=42)),
        ("DT Random Splitter",
         DecisionTreeClassifier(criterion="gini", splitter="random",
                                max_depth=None, min_samples_leaf=1, random_state=42)),
        ("CART (Best)",
         DecisionTreeClassifier(criterion="entropy", splitter="best",
                                max_depth=None, min_samples_leaf=1,
                                min_samples_split=2, max_features=None,
                                class_weight="balanced", ccp_alpha=0.0, random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=100, max_depth=None,
                                max_features="sqrt", min_samples_leaf=1,
                                n_jobs=-1, random_state=RANDOM_SEED)),
        ("Hist Gradient Boosting",
         HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
                                        max_depth=5, min_samples_leaf=20,
                                        random_state=40)),
        ("K-NN k=5",
         KNeighborsClassifier(n_neighbors=5, weights="distance",
                              algorithm="ball_tree", leaf_size=40, n_jobs=-1)),
        ("K-NN k=3",
         KNeighborsClassifier(n_neighbors=3, weights="distance",
                              algorithm="ball_tree", leaf_size=40, n_jobs=-1)),
    ]


def run() -> None:
    X_aug, y_aug, scaler, le = load_and_augment()
    models = build_models()

    print("\n" + "=" * 70)
    print("ALL MODELS — 10-Fold CV | Dual-Mode Timing (R4C2)")
    print(f"Samples  : {X_aug.shape[0]:,} | Features : {X_aug.shape[1]}")
    print(f"Accuracy : full fold (~{X_aug.shape[0] // N_FOLDS:,} samples)")
    print(f"Latency  : subsample ({LATENCY_SAMPLE_SIZE:,} samples per fold)")
    print(f"Warm-up  : {N_WARMUP} passes | Models : {len(models)}")
    print("=" * 70)

    results = []
    for name, model in models:
        print(f"\n[Processing: {name}]")
        r = evaluate_dual_timing(name, model, X_aug, y_aug)
        results.append(r)
        print_timing_report(r)

    # Summary table
    cols = [
        "Model", "Acc_mean", "Acc_std", "F1_mean", "F1_std",
        "TestTime_mean_s", "TestTime_std_s",
        "TestTime_CI95_low_s", "TestTime_CI95_high_s",
        "PerSample_mean_ms", "PerSample_std_ms",
        "PerSample_CI95_low_ms", "PerSample_CI95_high_ms",
        "Throughput_mean_sps", "Throughput_std_sps",
        "n_test_samples_per_fold",
    ]
    df_out = (pd.DataFrame(results)[cols]
                .sort_values("Acc_mean", ascending=False)
                .reset_index(drop=True))
    print("\n" + "=" * 70)
    print("SUMMARY TABLE — Sorted by Accuracy (R4C2)")
    print("=" * 70)
    print(df_out.to_string(index=False, float_format="%.4f"))
    print(f"\n[NOTE] Timing based on {LATENCY_SAMPLE_SIZE:,}-sample subsample per fold.")
    print("[NOTE] Accuracy computed on full test fold (unbiased).")
    print("[NOTE] Per-sample latency = subsample_time / subsample_size.")

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n[INFO] Saving models to '{MODEL_DIR}'...")
    for name, model in models:
        model.fit(X_aug, y_aug)
        safe_name = (name.replace(" ", "_").replace("(", "")
                         .replace(")", "").replace("=", "").replace("-", ""))
        fpath = os.path.join(MODEL_DIR, f"{safe_name}_Augmented.joblib")
        joblib.dump(model, fpath)
        print(f"   ✔ {name:<25} → {os.path.basename(fpath)}")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_OurApproach.joblib"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder_OurApproach.joblib"))
    print("\n[SUCCESS] All models saved.")


if __name__ == "__main__":
    run()
