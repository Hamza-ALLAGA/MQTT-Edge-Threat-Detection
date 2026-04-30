"""
phase0_env/document_environment.py
====================================
PHASE 0 — Environment Documentation (R4C2)
Addresses: Reviewer 4, Comment 2 — Timing Protocol Transparency.

Records all software versions, OS, and hardware context
to ensure full reproducibility of timing measurements.

Usage
-----
    python phase0_env/document_environment.py

Output can be included in the supplementary materials of the manuscript.
"""

import sys
import platform
import os
import importlib


def document_environment() -> None:
    sep = "=" * 70

    print(sep)
    print("EXPERIMENTAL ENVIRONMENT DOCUMENTATION")
    print("Addressing: Reviewer 4, Comment 2 — Timing Protocol Transparency")
    print(sep)

    # [1/4] Python runtime
    print("\n[1/4] Python Runtime")
    print(f"  Python version : {sys.version}")
    print(f"  Platform       : {platform.platform()}")
    print(f"  Architecture   : {platform.machine()}")
    print(f"  Processor      : {platform.processor()}")

    # [2/4] Core scientific stack
    print("\n[2/4] Core Scientific Stack")
    libs = ["numpy", "pandas", "scipy", "sklearn", "joblib", "imbalanced_learn"]
    for lib in libs:
        try:
            if lib == "sklearn":
                import sklearn
                print(f"  scikit-learn    : {sklearn.__version__}")
            elif lib == "imbalanced_learn":
                import imblearn
                print(f"  imbalanced-learn: {imblearn.__version__}")
            else:
                mod = importlib.import_module(lib)
                print(f"  {lib:<16}: {mod.__version__}")
        except Exception as e:
            print(f"  {lib:<16}: NOT INSTALLED ({e})")

    # [3/4] CPU information
    print("\n[3/4] CPU Information (Inference Hardware)")
    cpu_count = os.cpu_count()
    print(f"  Logical CPU cores : {cpu_count}")
    print(f"  NOTE: All timing measurements were conducted on CPU only.")
    print(f"  No GPU acceleration was used during inference.")

    # [4/4] Timing protocol
    print("\n[4/4] Timing Protocol Summary")
    print(f"  Cross-validation strategy  : Stratified 10-Fold (k=10)")
    print(f"  Timing scope               : Per-fold inference time on held-out test fold")
    print(f"  Warm-up procedure          : First 10 inference calls discarded per model")
    print(f"  Batch size during inference: Full fold (online batch mode)")
    print(f"  Timer resolution           : time.perf_counter() [nanosecond resolution]")
    print(f"  Statistical reporting      : mean ± std, 95% CI, Wilcoxon p-value")
    print(f"  Significance threshold     : α = 0.05")

    print(f"\n{sep}")
    print("Environment documentation complete.")
    print("Include this output in supplementary materials of the revised manuscript.")
    print(sep)


if __name__ == "__main__":
    document_environment()
