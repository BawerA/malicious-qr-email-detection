"""
Step 5: Evaluate XGBoost — Compare to Rivas et al. (2025)

Loads training results and generates comparison tables:
1. Our XGBoost (24 features) vs Rivas's published XGBoost results
2. Our XGBoost: 24 features → 63 features improvement

Input:  pipeline/results/training_results.csv
Output: pipeline/results/rivas_comparison.csv
        pipeline/results/model_comparison.csv
"""

import pandas as pd
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
RESULTS_DIR = PIPELINE_DIR / "results"

TRAINING_RESULTS = RESULTS_DIR / "training_results.csv"

# Rivas et al. (2025) published XGBoost results — Table II
RIVAS_XGBOOST = {"accuracy": 0.947, "precision": 0.95, "recall": 0.94, "f1": 0.94}


def main():
    print("=" * 60)
    print("STEP 5: EVALUATE XGBOOST — Compare to Rivas et al. (2025)")
    print("=" * 60)
    print()

    if not TRAINING_RESULTS.exists():
        print(f"[!] Missing: {TRAINING_RESULTS}")
        print("    Run '04_train_models.py' first.")
        return

    results = pd.read_csv(TRAINING_RESULTS)

    our_24 = results[results["feature_set"] == "rivas_24"]
    our_63 = results[results["feature_set"] == "our_63"]

    xgb_24 = our_24[our_24["model"] == "XGBoost"].iloc[0] if len(our_24[our_24["model"] == "XGBoost"]) > 0 else None
    xgb_63 = our_63[our_63["model"] == "XGBoost"].iloc[0] if len(our_63[our_63["model"] == "XGBoost"]) > 0 else None

    # ── Table A: Our 24-feature XGBoost vs Rivas's published XGBoost ──
    print("=" * 70)
    print("TABLE A: Our XGBoost (24 feat) vs Rivas Published XGBoost")
    print("=" * 70)
    print(f"\n{'Metric':<12} {'Rivas Published':>16} {'Our 24-feat':>16} {'Difference':>12}")
    print("-" * 56)

    comparison_rows = []

    if xgb_24 is not None:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            rivas_val = RIVAS_XGBOOST[metric]
            our_val = xgb_24[metric]
            diff = our_val - rivas_val

            print(f"{metric:<12} {rivas_val:>16.4f} {our_val:>16.4f} {diff:>+12.4f}")

            comparison_rows.append({
                "metric": metric,
                "rivas_published": rivas_val,
                "our_24_features": our_val,
                "difference": diff,
            })

    rivas_comp_df = pd.DataFrame(comparison_rows)
    rivas_comp_df.to_csv(RESULTS_DIR / "rivas_comparison.csv", index=False)

    # ── Table B: Our 24 features vs Our 63 features ──
    print("\n\n" + "=" * 70)
    print("TABLE B: XGBoost — 24 Features vs 63 Features")
    print("=" * 70)
    print(f"\n{'Metric':<12} {'24 Features':>14} {'63 Features':>14} {'Improvement':>14}")
    print("-" * 54)

    improvement_rows = []

    if xgb_24 is not None and xgb_63 is not None:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            val_24 = xgb_24[metric]
            val_63 = xgb_63[metric]
            improvement = val_63 - val_24

            print(f"{metric:<12} {val_24:>14.4f} {val_63:>14.4f} {improvement:>+14.4f}")

            improvement_rows.append({
                "metric": metric,
                "features_24": val_24,
                "features_63": val_63,
                "improvement": improvement,
            })

        # FPR comparison
        if xgb_24["fpr"] is not None and xgb_63["fpr"] is not None:
            print(f"{'fpr':<12} {xgb_24['fpr']:>14.4f} {xgb_63['fpr']:>14.4f} {xgb_63['fpr'] - xgb_24['fpr']:>+14.4f}")

    improvement_df = pd.DataFrame(improvement_rows)
    improvement_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("XGBOOST RESULTS SUMMARY")
    print("=" * 70)

    if xgb_63 is not None:
        print(f"\n  XGBoost (63 features):")
        print(f"    Accuracy:  {xgb_63['accuracy']:.4f}")
        print(f"    Precision: {xgb_63['precision']:.4f}")
        print(f"    Recall:    {xgb_63['recall']:.4f}")
        print(f"    F1:        {xgb_63['f1']:.4f}")
        if xgb_63["fpr"] is not None:
            print(f"    FPR:       {xgb_63['fpr']:.4f}")
        if xgb_63.get("val_f1") is not None:
            print(f"    Val F1:    {xgb_63['val_f1']:.4f}")

        # H1 check
        p = xgb_63["precision"]
        r = xgb_63["recall"]
        print(f"\n  H1 CHECK: Precision={p:.4f}, Recall={r:.4f}")
        if p >= 0.90 and r >= 0.90:
            print(f"    H1 SUPPORTED: Both >= 90%")
        else:
            print(f"    H1 NOT MET: Need both >= 90%")

    print(f"\n[+] Saved: {RESULTS_DIR / 'rivas_comparison.csv'}")
    print(f"[+] Saved: {RESULTS_DIR / 'model_comparison.csv'}")
    print("\n    Next step: Run '06_generate_email_qr_codes.py'")


if __name__ == "__main__":
    main()
