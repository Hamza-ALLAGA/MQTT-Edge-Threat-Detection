"""
phase2_external_validation/validate_mqttset.py
================================================
PHASE 2 — External Validation: MQTTset Dataset
5-Feature selection approach applied to MQTTset (Vaccari et al. 2020).
R4C2: 10-fold stratified CV with full timing statistics.

Pre-processed dataset files expected:
  /content/drive/MyDrive/R3-PMC/Dataset_for_Phase_2/processed_mqttset/
      processed_dataset.npz
      label_encoder.joblib
      scaler.joblib
      metadata.joblib

Adjust SAVE_DIR and MODEL_DIR paths as needed.

Usage
-----
    python phase2_external_validation/validate_mqttset.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    evaluate_with_cv_timing,
    print_timing_report,
    wilcoxon_test,
    RANDOM_SEED,
    N_FOLDS,
    N_WARMUP,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# ── Paths — adjust to your environment ───────────────────────────────────────
SAVE_DIR  = "/content/drive/MyDrive/R3-PMC/Dataset_for_Phase_2/processed_mqttset"
MODEL_DIR = "/content/drive/MyDrive/R3-PMC/Dataset_for_Phase_2/models_mqttset"

# ── Hyperparameters (tuned for MQTTset) ──────────────────────────────────────
CART_HP = dict(criterion="gini", max_depth=10, min_samples_split=2,
               min_samples_leaf=1, class_weight="balanced")
DT_HP   = dict(criterion="gini", max_depth=8,  min_samples_split=3,
               min_samples_leaf=1, max_features="sqrt")
RF_HP   = dict(n_estimators=2,   max_depth=7,  min_samples_split=14,
               min_samples_leaf=1, max_features=2)


def load_dataset() -> tuple:
    data     = np.load(os.path.join(SAVE_DIR, "processed_dataset.npz"))
    X_proc   = data["X_proc"]
    y        = data["y"]
    le       = joblib.load(os.path.join(SAVE_DIR, "label_encoder.joblib"))
    scaler   = joblib.load(os.path.join(SAVE_DIR, "scaler.joblib"))
    metadata = joblib.load(os.path.join(SAVE_DIR, "metadata.joblib"))

    print("=" * 70)
    print("MQTTset — LOADING DATASET")
    print("=" * 70)
    print(f"  X shape     : {X_proc.shape}")
    print(f"  y shape     : {y.shape}")
    print(f"  n_classes   : {len(le.classes_)}  →  {list(le.classes_)}")
    return X_proc, y, le


def build_models() -> dict:
    return {
        "CART (MQTTset)"          : DecisionTreeClassifier(**CART_HP, random_state=RANDOM_SEED),
        "Decision Tree (MQTTset)" : DecisionTreeClassifier(**DT_HP,   random_state=RANDOM_SEED),
        "Random Forest (MQTTset)" : RandomForestClassifier(**RF_HP, n_jobs=1, random_state=RANDOM_SEED),
    }


def run() -> None:
    X_proc, y, le = load_dataset()
    model_defs    = build_models()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("MQTTset — TRAINING  (10-fold stratified CV, R4C2)")
    print(f"Warm-up: first {N_WARMUP} inference passes discarded per model")
    print("=" * 70)

    results, trained = [], {}
    for mname, mdl in model_defs.items():
        print(f"\n  [{mname}]")
        res = evaluate_with_cv_timing(mname, mdl, X_proc, y)
        results.append(res)
        trained[mname] = mdl
        print_timing_report(res)

    # Summary table
    df_res = (pd.DataFrame(results)
                .sort_values("Acc_mean", ascending=False)
                .reset_index(drop=True))
    print("\n" + "=" * 70)
    print("MQTTset — SUMMARY TABLE")
    print("=" * 70)
    cols = ["Model", "Acc_mean", "Acc_std", "F1_mean", "F1_std",
            "TestTime_mean_s", "PerSample_mean_ms", "Throughput_mean_sps"]
    print(df_res[cols].to_string(index=False, float_format="%.4f"))

    # Wilcoxon between best two models
    if len(results) >= 2:
        w = wilcoxon_test(results[0], results[1])
        print(f"\n[Wilcoxon] {results[0]['Model']} vs {results[1]['Model']}: "
              f"p={w['p_value']:.4f} | significant={w['significant']}")

    # Save models + artefacts
    for mname, mdl in trained.items():
        mdl.fit(X_proc, y)
        safe = mname.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        joblib.dump(mdl, os.path.join(MODEL_DIR, f"{safe}.joblib"))

    joblib.dump({"CART": CART_HP, "DT": DT_HP, "RF": RF_HP},
                os.path.join(MODEL_DIR, "model_hyperparams.joblib"))
    joblib.dump(df_res, os.path.join(MODEL_DIR, "results_summary.joblib"))
    print("\n[SUCCESS] MQTTset models saved to:", MODEL_DIR)


if __name__ == "__main__":
    run()
