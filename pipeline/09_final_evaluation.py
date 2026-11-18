"""
Step 9: Final Evaluation — Test All Three Hypotheses

Consolidates all results and formally evaluates:
  H1: Lexical XGBoost achieves ≥90% precision and ≥90% recall
  H2: Reputation enrichment improves precision by ≥3 percentage points
  H3: Pipeline maintains ≤5% false-positive rate

Generates final tables and summary for the praxis paper.

Input:  pipeline/results/training_results.csv
        pipeline/results/email_pipeline_results.csv
        pipeline/results/enriched_results.csv
        pipeline/results/enrichment_comparison.csv
Output: pipeline/results/final_evaluation.txt
        pipeline/results/hypothesis_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PIPELINE_DIR = Path(__file__).parent
RESULTS_DIR = PIPELINE_DIR / "results"

TRAINING_RESULTS = RESULTS_DIR / "training_results.csv"
EMAIL_RESULTS = RESULTS_DIR / "email_pipeline_results.csv"
ENRICHED_RESULTS = RESULTS_DIR / "enriched_results.csv"
COMPARISON_FILE = RESULTS_DIR / "enrichment_comparison.csv"

FINAL_REPORT = RESULTS_DIR / "final_evaluation.txt"
HYPOTHESIS_FILE = RESULTS_DIR / "hypothesis_results.csv"


def evaluate_h1(training_results):
    """H1: Lexical XGBoost achieves ≥90% precision and ≥90% recall."""

    print("=" * 70)
    print("HYPOTHESIS 1: Lexical-Feature XGBoost Performance")
    print("  Target: ≥90% precision AND ≥90% recall")
    print("=" * 70)

    results = {}

    # Check both feature sets
    for feature_set in ["rivas_24", "our_59"]:
        xgb = training_results[
            (training_results["model"] == "XGBoost") &
            (training_results["feature_set"] == feature_set)
        ]

        if len(xgb) == 0:
            print(f"\n  [{feature_set}] XGBoost not found in results")
            continue

        row = xgb.iloc[0]
        precision = row["precision"]
        recall = row["recall"]
        f1 = row["f1"]
        accuracy = row["accuracy"]

        precision_pass = precision >= 0.90
        recall_pass = recall >= 0.90
        h1_pass = precision_pass and recall_pass

        print(f"\n  [{feature_set}] XGBoost Results:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}  {'PASS' if precision_pass else 'FAIL'} (≥0.90)")
        print(f"    Recall:    {recall:.4f}  {'PASS' if recall_pass else 'FAIL'} (≥0.90)")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    H1 Status: {'SUPPORTED' if h1_pass else 'NOT SUPPORTED'}")

        results[feature_set] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "h1_supported": h1_pass,
        }

    # Also check all models for comparison
    print(f"\n  {'─'*60}")
    print(f"  All Models Comparison (59 features):")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*60}")

    our_59 = training_results[training_results["feature_set"] == "our_59"]
    for _, row in our_59.iterrows():
        print(f"  {row['model']:<20} {row['accuracy']:>10.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")

    return results


def evaluate_h2(comparison_data, enriched_data=None):
    """H2: Reputation enrichment improves precision by ≥3 percentage points."""

    print(f"\n\n{'='*70}")
    print("HYPOTHESIS 2: Reputation Enrichment Improvement")
    print("  Target: ≥3 percentage point increase in precision")
    print("=" * 70)

    results = {}

    if comparison_data is not None and len(comparison_data) >= 2:
        stage1 = comparison_data[comparison_data["stage"].str.contains("Stage 1")].iloc[0]
        stage2 = comparison_data[comparison_data["stage"].str.contains("Stage 2")].iloc[0]

        precision_s1 = stage1["precision"]
        precision_s2 = stage2["precision"]
        improvement = (precision_s2 - precision_s1) * 100  # percentage points

        recall_s1 = stage1["recall"]
        recall_s2 = stage2["recall"]

        f1_s1 = stage1["f1"]
        f1_s2 = stage2["f1"]

        h2_pass = improvement >= 3.0

        print(f"\n  Stage 1 (Lexical Only):")
        print(f"    Precision: {precision_s1:.4f}")
        print(f"    Recall:    {recall_s1:.4f}")
        print(f"    F1:        {f1_s1:.4f}")

        print(f"\n  Stage 2 (Lexical + Reputation):")
        print(f"    Precision: {precision_s2:.4f}")
        print(f"    Recall:    {recall_s2:.4f}")
        print(f"    F1:        {f1_s2:.4f}")

        print(f"\n  Improvement:")
        print(f"    Precision: {improvement:+.2f} percentage points")
        print(f"    Recall:    {(recall_s2 - recall_s1)*100:+.2f} percentage points")
        print(f"    F1:        {(f1_s2 - f1_s1)*100:+.2f} percentage points")
        print(f"\n  H2 Status: {'SUPPORTED' if h2_pass else 'NOT SUPPORTED'}")

        results = {
            "precision_stage1": precision_s1,
            "precision_stage2": precision_s2,
            "improvement_pp": improvement,
            "h2_supported": h2_pass,
        }
    else:
        print("\n  [!] Enrichment comparison data not available.")
        print("      Run '08_reputation_enrichment.py' first.")
        results = {"h2_supported": None}

    return results


def evaluate_h3(email_results, enriched_results=None):
    """H3: Pipeline maintains ≤5% false-positive rate."""

    print(f"\n\n{'='*70}")
    print("HYPOTHESIS 3: False-Positive Rate")
    print("  Target: ≤5% FPR")
    print("=" * 70)

    results = {}

    # Check FPR from Stage 1 (email pipeline)
    if email_results is not None:
        classified = email_results[
            email_results["predicted_label"].isin(["benign", "malicious"]) &
            email_results["true_label"].isin(["benign", "malicious"])
        ]

        if len(classified) > 0:
            fp = ((classified["true_label"] == "benign") & (classified["predicted_label"] == "malicious")).sum()
            tn = ((classified["true_label"] == "benign") & (classified["predicted_label"] == "benign")).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            h3_pass_s1 = fpr <= 0.05

            print(f"\n  Stage 1 (Lexical Only) — Email Pipeline:")
            print(f"    False Positives: {fp}")
            print(f"    True Negatives:  {tn}")
            print(f"    FPR: {fpr:.4f} ({fpr*100:.2f}%)")
            print(f"    H3 Status: {'SUPPORTED' if h3_pass_s1 else 'NOT SUPPORTED'} (≤5%)")

            results["fpr_stage1"] = fpr
            results["h3_stage1_supported"] = h3_pass_s1

    # Check FPR from Stage 2 (enriched)
    if enriched_results is not None:
        classified_e = enriched_results[
            enriched_results["hybrid_label"].isin(["benign", "malicious"]) &
            enriched_results["true_label"].isin(["benign", "malicious"])
        ]

        if len(classified_e) > 0:
            fp_e = ((classified_e["true_label"] == "benign") & (classified_e["hybrid_label"] == "malicious")).sum()
            tn_e = ((classified_e["true_label"] == "benign") & (classified_e["hybrid_label"] == "benign")).sum()
            fpr_e = fp_e / (fp_e + tn_e) if (fp_e + tn_e) > 0 else 0

            h3_pass_s2 = fpr_e <= 0.05

            print(f"\n  Stage 2 (Hybrid) — Email Pipeline:")
            print(f"    False Positives: {fp_e}")
            print(f"    True Negatives:  {tn_e}")
            print(f"    FPR: {fpr_e:.4f} ({fpr_e*100:.2f}%)")
            print(f"    H3 Status: {'SUPPORTED' if h3_pass_s2 else 'NOT SUPPORTED'} (≤5%)")

            results["fpr_stage2"] = fpr_e
            results["h3_stage2_supported"] = h3_pass_s2

    if not results:
        print("\n  [!] No email pipeline results available.")
        print("      Run '07_email_qr_pipeline.py' first.")

    return results


def generate_final_report(h1_results, h2_results, h3_results):
    """Generate a comprehensive final report for the praxis paper."""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FINAL EVALUATION REPORT")
    report_lines.append("Detection of Malicious QR Codes Embedded in Emails")
    report_lines.append("through URL Analysis and Machine-Learning Threat Classification")
    report_lines.append("=" * 70)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # H1
    report_lines.append("-" * 70)
    report_lines.append("HYPOTHESIS 1")
    report_lines.append("A lexical-feature XGBoost model will reach at least 90% precision")
    report_lines.append("and 90% recall on the email-QR test set.")
    report_lines.append("-" * 70)

    if "our_59" in h1_results:
        r = h1_results["our_59"]
        report_lines.append(f"  XGBoost (59 features):")
        report_lines.append(f"    Precision: {r['precision']:.4f}")
        report_lines.append(f"    Recall:    {r['recall']:.4f}")
        report_lines.append(f"    F1:        {r['f1']:.4f}")
        report_lines.append(f"    Accuracy:  {r['accuracy']:.4f}")
        report_lines.append(f"  RESULT: {'SUPPORTED' if r['h1_supported'] else 'NOT SUPPORTED'}")
    report_lines.append("")

    # H2
    report_lines.append("-" * 70)
    report_lines.append("HYPOTHESIS 2")
    report_lines.append("Adding multi-source reputation scores will increase precision by")
    report_lines.append("at least 3 percentage points over the lexical model alone.")
    report_lines.append("-" * 70)

    if h2_results.get("h2_supported") is not None:
        report_lines.append(f"  Precision (Stage 1): {h2_results['precision_stage1']:.4f}")
        report_lines.append(f"  Precision (Stage 2): {h2_results['precision_stage2']:.4f}")
        report_lines.append(f"  Improvement: {h2_results['improvement_pp']:+.2f} pp")
        report_lines.append(f"  RESULT: {'SUPPORTED' if h2_results['h2_supported'] else 'NOT SUPPORTED'}")
    else:
        report_lines.append("  RESULT: NOT EVALUATED (reputation enrichment not run)")
    report_lines.append("")

    # H3
    report_lines.append("-" * 70)
    report_lines.append("HYPOTHESIS 3")
    report_lines.append("The detection pipeline will maintain a false-positive rate of 5%")
    report_lines.append("or lower when classifying email-embedded QR code URLs.")
    report_lines.append("-" * 70)

    if "fpr_stage1" in h3_results:
        report_lines.append(f"  FPR (Stage 1): {h3_results['fpr_stage1']:.4f} ({h3_results['fpr_stage1']*100:.2f}%)")
        report_lines.append(f"  RESULT (Stage 1): {'SUPPORTED' if h3_results.get('h3_stage1_supported') else 'NOT SUPPORTED'}")

    if "fpr_stage2" in h3_results:
        report_lines.append(f"  FPR (Stage 2): {h3_results['fpr_stage2']:.4f} ({h3_results['fpr_stage2']*100:.2f}%)")
        report_lines.append(f"  RESULT (Stage 2): {'SUPPORTED' if h3_results.get('h3_stage2_supported') else 'NOT SUPPORTED'}")

    if not h3_results:
        report_lines.append("  RESULT: NOT EVALUATED (email pipeline not run)")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def main():
    print("=" * 60)
    print("STEP 9: FINAL EVALUATION — TEST ALL HYPOTHESES")
    print("=" * 60)
    print()

    # Load all available results
    training_results = None
    email_results = None
    enriched_results = None
    comparison_data = None

    if TRAINING_RESULTS.exists():
        training_results = pd.read_csv(TRAINING_RESULTS)
        print(f"[+] Loaded training results: {len(training_results)} rows")
    else:
        print(f"[!] Training results not found: {TRAINING_RESULTS}")

    if EMAIL_RESULTS.exists():
        email_results = pd.read_csv(EMAIL_RESULTS)
        print(f"[+] Loaded email pipeline results: {len(email_results)} rows")
    else:
        print(f"[!] Email pipeline results not found: {EMAIL_RESULTS}")

    if ENRICHED_RESULTS.exists():
        enriched_results = pd.read_csv(ENRICHED_RESULTS)
        print(f"[+] Loaded enriched results: {len(enriched_results)} rows")
    else:
        print(f"[!] Enriched results not found: {ENRICHED_RESULTS}")

    if COMPARISON_FILE.exists():
        comparison_data = pd.read_csv(COMPARISON_FILE)
        print(f"[+] Loaded enrichment comparison: {len(comparison_data)} rows")
    else:
        print(f"[!] Enrichment comparison not found: {COMPARISON_FILE}")

    # ── Evaluate Hypotheses ──

    # H1: Lexical model performance
    h1_results = {}
    if training_results is not None:
        h1_results = evaluate_h1(training_results)
    else:
        print("\n[!] Cannot evaluate H1: No training results available.")

    # H2: Reputation enrichment improvement
    h2_results = evaluate_h2(comparison_data, enriched_results)

    # H3: False-positive rate
    h3_results = evaluate_h3(email_results, enriched_results)

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print("HYPOTHESIS SUMMARY")
    print(f"{'='*70}")

    hypothesis_rows = []

    # H1
    if "our_59" in h1_results:
        h1_status = "SUPPORTED" if h1_results["our_59"]["h1_supported"] else "NOT SUPPORTED"
        h1_detail = f"P={h1_results['our_59']['precision']:.4f}, R={h1_results['our_59']['recall']:.4f}"
    else:
        h1_status = "NOT EVALUATED"
        h1_detail = "Training results not available"

    print(f"\n  H1 (≥90% P/R):     {h1_status}")
    print(f"     {h1_detail}")
    hypothesis_rows.append({"hypothesis": "H1", "target": "≥90% precision and recall",
                           "status": h1_status, "detail": h1_detail})

    # H2
    if h2_results.get("h2_supported") is not None:
        h2_status = "SUPPORTED" if h2_results["h2_supported"] else "NOT SUPPORTED"
        h2_detail = f"Improvement: {h2_results['improvement_pp']:+.2f} pp"
    else:
        h2_status = "NOT EVALUATED"
        h2_detail = "Enrichment not run"

    print(f"  H2 (≥3pp improve): {h2_status}")
    print(f"     {h2_detail}")
    hypothesis_rows.append({"hypothesis": "H2", "target": "≥3pp precision improvement",
                           "status": h2_status, "detail": h2_detail})

    # H3
    if "fpr_stage1" in h3_results:
        fpr_val = h3_results.get("fpr_stage2", h3_results.get("fpr_stage1", None))
        h3_supported = h3_results.get("h3_stage2_supported", h3_results.get("h3_stage1_supported", None))
        h3_status = "SUPPORTED" if h3_supported else "NOT SUPPORTED"
        h3_detail = f"FPR: {fpr_val:.4f} ({fpr_val*100:.2f}%)"
    else:
        h3_status = "NOT EVALUATED"
        h3_detail = "Email pipeline not run"

    print(f"  H3 (≤5% FPR):     {h3_status}")
    print(f"     {h3_detail}")
    hypothesis_rows.append({"hypothesis": "H3", "target": "≤5% FPR",
                           "status": h3_status, "detail": h3_detail})

    # Save hypothesis results
    hypothesis_df = pd.DataFrame(hypothesis_rows)
    hypothesis_df.to_csv(HYPOTHESIS_FILE, index=False)

    # Generate and save final report
    report = generate_final_report(h1_results, h2_results, h3_results)
    with open(FINAL_REPORT, "w") as f:
        f.write(report)

    print(f"\n[+] Final report saved to: {FINAL_REPORT}")
    print(f"[+] Hypothesis results saved to: {HYPOTHESIS_FILE}")
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
