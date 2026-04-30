"""
utils/evaluation.py
====================
Shared statistical and evaluation utilities used across all experiment phases.
Addresses: Reviewer 4, Comment 2 (R4C2) — Timing Protocol Transparency.

All timing values are measured with time.perf_counter() (nanosecond resolution).
Warm-up inference calls are discarded before timing begins.
"""

import time
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS     = 10    # k=10 stratified CV for all evaluations
N_WARMUP    = 10    # warm-up inference calls discarded per model
ALPHA       = 0.05  # significance threshold
np.random.seed(RANDOM_SEED)


def timing_stats(times_seconds: list, n_samples: int) -> dict:
    """
    Compute full timing statistics from per-fold inference times.

    Parameters
    ----------
    times_seconds : list of float
        Per-fold total inference time in seconds.
    n_samples : int
        Number of test samples (used to derive per-sample latency).

    Returns
    -------
    dict with keys:
        mean_total_s, std_total_s, ci95_low_s, ci95_high_s,
        mean_per_sample_ms, std_per_sample_ms,
        ci95_low_per_sample_ms, ci95_high_per_sample_ms,
        mean_throughput_sps, std_throughput_sps
    """
    t      = np.array(times_seconds, dtype=float)
    n      = len(t)
    mean_t = np.mean(t)
    std_t  = np.std(t, ddof=1)
    se     = std_t / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)   # two-tailed 95% CI
    ci_low  = mean_t - t_crit * se
    ci_high = mean_t + t_crit * se

    per_sample_ms = t / n_samples * 1000.0
    mean_ps = np.mean(per_sample_ms)
    std_ps  = np.std(per_sample_ms, ddof=1)
    se_ps   = std_ps / np.sqrt(n)

    throughput = n_samples / t

    return {
        "mean_total_s"             : mean_t,
        "std_total_s"              : std_t,
        "ci95_low_s"               : ci_low,
        "ci95_high_s"              : ci_high,
        "mean_per_sample_ms"       : mean_ps,
        "std_per_sample_ms"        : std_ps,
        "ci95_low_per_sample_ms"   : mean_ps - t_crit * se_ps,
        "ci95_high_per_sample_ms"  : mean_ps + t_crit * se_ps,
        "mean_throughput_sps"      : np.mean(throughput),
        "std_throughput_sps"       : np.std(throughput, ddof=1),
    }


def evaluate_with_cv_timing(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_FOLDS,
    n_warmup: int = N_WARMUP,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a classifier via Stratified k-Fold CV with full timing statistics.

    Warm-up: Before recording timings, the model runs `n_warmup` silent
    single-sample predictions on fold 0 to discard Python/framework
    initialisation overhead.

    Batch size: inference is performed on the entire held-out fold at once.
    Per-sample latency is derived as total_time / fold_size.

    Parameters
    ----------
    name     : Model label used in result dict and printed output.
    model    : sklearn-compatible estimator.
    X, y     : Full dataset (combined train+test pool for CV).
    n_splits : Number of CV folds (default 10).
    n_warmup : Number of warm-up passes discarded (default 10).
    verbose  : Print warm-up confirmation line.

    Returns
    -------
    dict with performance, timing, and raw per-fold arrays.
    """
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(X, y))

    # Warm-up on fold 0
    tr0, te0 = folds[0]
    model.fit(X[tr0], y[tr0])
    for _ in range(n_warmup):
        model.predict(X[te0[:1]])
    if verbose:
        print(f"  [{name}] Warm-up complete ({n_warmup} passes discarded).")

    fold_acc, fold_f1, fold_times, fold_sizes = [], [], [], []

    for tr_idx, te_idx in folds:
        model.fit(X[tr_idx], y[tr_idx])

        t0     = time.perf_counter()
        y_pred = model.predict(X[te_idx])
        t1     = time.perf_counter()

        fold_times.append(t1 - t0)
        fold_sizes.append(len(te_idx))
        fold_acc.append(accuracy_score(y[te_idx], y_pred))
        fold_f1.append(f1_score(y[te_idx], y_pred, average="weighted", zero_division=0))

    median_n = int(np.median(fold_sizes))
    ts       = timing_stats(fold_times, median_n)

    return {
        "Model"                      : name,
        "Acc_mean"                   : np.mean(fold_acc) * 100,
        "Acc_std"                    : np.std(fold_acc, ddof=1) * 100,
        "F1_mean"                    : np.mean(fold_f1) * 100,
        "F1_std"                     : np.std(fold_f1, ddof=1) * 100,
        "TestTime_mean_s"            : ts["mean_total_s"],
        "TestTime_std_s"             : ts["std_total_s"],
        "TestTime_CI95_low_s"        : ts["ci95_low_s"],
        "TestTime_CI95_high_s"       : ts["ci95_high_s"],
        "PerSample_mean_ms"          : ts["mean_per_sample_ms"],
        "PerSample_std_ms"           : ts["std_per_sample_ms"],
        "PerSample_CI95_low_ms"      : ts["ci95_low_per_sample_ms"],
        "PerSample_CI95_high_ms"     : ts["ci95_high_per_sample_ms"],
        "Throughput_mean_sps"        : ts["mean_throughput_sps"],
        "Throughput_std_sps"         : ts["std_throughput_sps"],
        "n_test_samples_per_fold"    : median_n,
        "n_folds"                    : n_splits,
        "_fold_test_times"           : fold_times,
        "_fold_f1"                   : fold_f1,
    }


def print_timing_report(r: dict) -> None:
    """Print a standardised timing + performance report for one model."""
    print(f"  Model                  : {r['Model']}")
    print(f"  Accuracy               : {r['Acc_mean']:.2f}% ± {r['Acc_std']:.2f}%")
    print(f"  F1-Score (weighted)    : {r['F1_mean']:.2f}% ± {r['F1_std']:.2f}%")
    print(
        f"  Total Test Time        : {r['TestTime_mean_s']*1000:.2f} ms "
        f"± {r['TestTime_std_s']*1000:.2f} ms  "
        f"(95% CI: [{r['TestTime_CI95_low_s']*1000:.2f}, "
        f"{r['TestTime_CI95_high_s']*1000:.2f}] ms)"
    )
    print(
        f"  Per-Sample Latency     : {r['PerSample_mean_ms']:.4f} ms "
        f"± {r['PerSample_std_ms']:.4f} ms  "
        f"(95% CI: [{r['PerSample_CI95_low_ms']:.4f}, "
        f"{r['PerSample_CI95_high_ms']:.4f}] ms)"
    )
    print(
        f"  Throughput             : {r['Throughput_mean_sps']:,.0f} "
        f"± {r['Throughput_std_sps']:,.0f} samples/sec"
    )
    print(f"  Test dataset size      : {r['n_test_samples_per_fold']:,} samples per fold")
    print(f"  CV folds               : k = {r['n_folds']}")


def wilcoxon_test(r_proposed: dict, r_baseline: dict, metric: str = "F1") -> dict:
    """
    Paired Wilcoxon signed-rank test between two models on per-fold scores.

    Parameters
    ----------
    r_proposed : result dict from evaluate_with_cv_timing (proposed model).
    r_baseline : result dict from evaluate_with_cv_timing (baseline model).
    metric     : 'F1' uses _fold_f1; 'time' uses _fold_test_times.

    Returns
    -------
    dict with keys: statistic, p_value, significant, label.
    """
    if metric == "F1":
        a = np.array(r_proposed["_fold_f1"])
        b = np.array(r_baseline["_fold_f1"])
        label = "F1-Score"
    else:
        a = np.array(r_proposed["_fold_test_times"])
        b = np.array(r_baseline["_fold_test_times"])
        label = "Test Time"

    diff = a - b
    if np.all(diff == 0):
        return {"p_value": 1.0, "significant": False, "label": label}

    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return {
        "statistic"  : stat,
        "p_value"    : p,
        "significant": p < ALPHA,
        "label"      : label,
    }
