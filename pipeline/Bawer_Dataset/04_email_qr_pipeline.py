"""
Bawer_Dataset — Step 4: Email QR Code Detection Pipeline

Same email pipeline as Step 7, but using the Bawer model instead of Rivas.
Uses the SAME test emails (generated in Step 6) so results are directly comparable.

This is the key test: does the Bawer model (trained with bare-domain URLs)
correctly classify the Tranco bare domains that the Rivas model failed on?

Input:  pipeline/data/test_emails/benign/*.eml  (same test emails)
        pipeline/data/test_emails/malicious/*.eml
        pipeline/Bawer_Dataset/models/bawer_63_XGBoost.joblib
Output: pipeline/Bawer_Dataset/results/email_pipeline_results.csv
        pipeline/Bawer_Dataset/results/email_pipeline_summary.txt
"""

import sys
import io
import email
import joblib
import pandas as pd
import numpy as np
from email import policy
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch.qr_decode import decode_qr_bytes
from src.fetch.url_features import url_features_enhanced as extract_url_features

# Paths — use SAME test emails as Rivas pipeline
EMAIL_DIR = PIPELINE_DIR / "data" / "test_emails"
BENIGN_DIR = EMAIL_DIR / "benign"
MALICIOUS_DIR = EMAIL_DIR / "malicious"

# Bawer model and results
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def extract_images_from_eml(eml_path):
    images = []
    with open(eml_path, "r", encoding="utf-8", errors="replace") as f:
        msg = email.message_from_file(f, policy=policy.default)
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type.startswith("image/"):
            payload = part.get_payload(decode=True)
            if payload:
                images.append({"content_type": content_type,
                               "filename": part.get_filename() or "unknown",
                               "data": payload})
    return images


def decode_qr_from_image_bytes(image_bytes):
    try:
        decoded_url, engine = decode_qr_bytes(image_bytes)
        return decoded_url, engine
    except Exception:
        return None, "error"


def classify_url(url, model_bundle):
    try:
        features = extract_url_features(str(url))
        feature_names = model_bundle["feature_names"]
        X = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
        scaler = model_bundle["scaler"]
        X_scaled = scaler.transform(X.values)
        model = model_bundle["model"]
        threshold = model_bundle.get("optimal_threshold", 0.5)

        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            prob = float(probs[1]) if len(probs) == 2 else float(max(probs))
            prediction = 1 if prob >= threshold else 0
        else:
            prediction = model.predict(X_scaled)[0]

        label_encoder = model_bundle["label_encoder"]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        # Map integer labels back to strings if needed
        if predicted_label == 0 or predicted_label == "0":
            predicted_label = "benign"
        elif predicted_label == 1 or predicted_label == "1":
            predicted_label = "malicious"
        return predicted_label, prob
    except Exception:
        return "error", None


def process_email_directory(email_dir, true_label, model_bundle, model_name):
    results = []
    eml_files = sorted(email_dir.glob("*.eml"))

    if not eml_files:
        print(f"    No .eml files found in {email_dir}")
        return results

    success = no_qr = errors = 0

    for eml_path in eml_files:
        try:
            images = extract_images_from_eml(eml_path)
            if not images:
                no_qr += 1
                results.append({"email_file": eml_path.name, "true_label": true_label,
                                "predicted_label": "no_image", "probability": None,
                                "decoded_url": None, "qr_engine": None,
                                "model_used": model_name, "num_images": 0})
                continue

            decoded_url = qr_engine = None
            for img_data in images:
                url, engine = decode_qr_from_image_bytes(img_data["data"])
                if url:
                    decoded_url, qr_engine = url, engine
                    break

            if not decoded_url:
                no_qr += 1
                results.append({"email_file": eml_path.name, "true_label": true_label,
                                "predicted_label": "no_qr_detected", "probability": None,
                                "decoded_url": None, "qr_engine": None,
                                "model_used": model_name, "num_images": len(images)})
                continue

            predicted_label, prob = classify_url(decoded_url, model_bundle)
            success += 1
            results.append({"email_file": eml_path.name, "true_label": true_label,
                            "predicted_label": predicted_label, "probability": prob,
                            "decoded_url": decoded_url, "qr_engine": qr_engine,
                            "model_used": model_name, "num_images": len(images)})

        except Exception:
            errors += 1
            results.append({"email_file": eml_path.name, "true_label": true_label,
                            "predicted_label": "error", "probability": None,
                            "decoded_url": None, "qr_engine": None,
                            "model_used": model_name, "num_images": 0})

    print(f"    Processed: {len(eml_files)} | Decoded: {success} | No QR: {no_qr} | Errors: {errors}")
    return results


def main():
    print("=" * 60)
    print("BAWER_DATASET — STEP 4: EMAIL QR PIPELINE")
    print("=" * 60)
    print()

    if not EMAIL_DIR.exists():
        print(f"[!] Email directory not found: {EMAIL_DIR}")
        print("    Run 'pipeline/06_generate_email_qr_codes.py' first.")
        return

    # Load BAWER model
    print("[+] Loading Bawer model...")
    model_path = MODELS_DIR / "bawer_63_XGBoost.joblib"
    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        return

    model_bundle = joblib.load(model_path)
    model_name = "bawer_63_XGBoost.joblib"
    print(f"    Model: {model_name}")
    print(f"    Features: {len(model_bundle['feature_names'])}")
    print(f"    Threshold: {model_bundle.get('optimal_threshold', 0.5)}")
    print()

    all_results = []

    if BENIGN_DIR.exists():
        print(f"[+] Processing benign emails from: {BENIGN_DIR}")
        all_results.extend(process_email_directory(
            BENIGN_DIR, "benign", model_bundle, model_name))

    if MALICIOUS_DIR.exists():
        print(f"\n[+] Processing malicious emails from: {MALICIOUS_DIR}")
        all_results.extend(process_email_directory(
            MALICIOUS_DIR, "malicious", model_bundle, model_name))

    if not all_results:
        print("\n[!] No results.")
        return

    results_df = pd.DataFrame(all_results)
    results_file = RESULTS_DIR / "email_pipeline_results.csv"
    results_df.to_csv(results_file, index=False)

    classified = results_df[results_df["predicted_label"].isin(["benign", "malicious"])]

    print(f"\n{'='*60}")
    print("BAWER EMAIL PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"\n  Total emails processed: {len(results_df)}")
    print(f"  Successfully classified: {len(classified)}")
    print(f"  No QR detected: {len(results_df[results_df['predicted_label'] == 'no_qr_detected'])}")
    print(f"  No image found: {len(results_df[results_df['predicted_label'] == 'no_image'])}")
    print(f"  Errors: {len(results_df[results_df['predicted_label'] == 'error'])}")

    if len(classified) > 0:
        correct = (classified["true_label"] == classified["predicted_label"]).sum()
        accuracy = correct / len(classified)

        tp = ((classified["true_label"] == "malicious") & (classified["predicted_label"] == "malicious")).sum()
        fp = ((classified["true_label"] == "benign") & (classified["predicted_label"] == "malicious")).sum()
        tn = ((classified["true_label"] == "benign") & (classified["predicted_label"] == "benign")).sum()
        fn = ((classified["true_label"] == "malicious") & (classified["predicted_label"] == "benign")).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\n  Classification Metrics (on {len(classified)} emails):")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    FPR:       {fpr:.4f}")

        print(f"\n  Confusion Matrix:")
        print(f"    TP (malicious→malicious): {tp}")
        print(f"    TN (benign→benign):       {tn}")
        print(f"    FP (benign→malicious):    {fp}")
        print(f"    FN (malicious→benign):    {fn}")

        # Compare with Rivas email pipeline
        rivas_results_path = PIPELINE_DIR / "results" / "email_pipeline_results.csv"
        if rivas_results_path.exists():
            rivas_df = pd.read_csv(rivas_results_path)
            rivas_classified = rivas_df[rivas_df["predicted_label"].isin(["benign", "malicious"])]

            if len(rivas_classified) > 0:
                r_correct = (rivas_classified["true_label"] == rivas_classified["predicted_label"]).sum()
                r_accuracy = r_correct / len(rivas_classified)

                r_tp = ((rivas_classified["true_label"] == "malicious") & (rivas_classified["predicted_label"] == "malicious")).sum()
                r_fp = ((rivas_classified["true_label"] == "benign") & (rivas_classified["predicted_label"] == "malicious")).sum()
                r_tn = ((rivas_classified["true_label"] == "benign") & (rivas_classified["predicted_label"] == "benign")).sum()
                r_fn = ((rivas_classified["true_label"] == "malicious") & (rivas_classified["predicted_label"] == "benign")).sum()

                r_precision = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0
                r_recall = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0
                r_f1 = 2 * r_precision * r_recall / (r_precision + r_recall) if (r_precision + r_recall) > 0 else 0
                r_fpr = r_fp / (r_fp + r_tn) if (r_fp + r_tn) > 0 else 0

                print(f"\n{'='*60}")
                print("COMPARISON: Rivas vs Bawer (Email Pipeline)")
                print(f"{'='*60}")
                print(f"  {'Metric':<12} {'Rivas':>10} {'Bawer':>10} {'Diff':>12}")
                print(f"  {'-'*46}")
                for name, rv, bv in [
                    ("Accuracy", r_accuracy, accuracy),
                    ("Precision", r_precision, precision),
                    ("Recall", r_recall, recall),
                    ("F1", r_f1, f1),
                    ("FPR", r_fpr, fpr),
                ]:
                    diff = bv - rv
                    print(f"  {name:<12} {rv:>10.4f} {bv:>10.4f} {diff:>+12.4f}")

                print(f"\n  Rivas: TP={r_tp} FP={r_fp} TN={r_tn} FN={r_fn}")
                print(f"  Bawer: TP={tp} FP={fp} TN={tn} FN={fn}")

        # Save summary
        summary_file = RESULTS_DIR / "email_pipeline_summary.txt"
        with open(summary_file, "w") as f:
            f.write("BAWER MODEL — EMAIL QR PIPELINE RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Features: {len(model_bundle['feature_names'])}\n")
            f.write(f"Threshold: {model_bundle.get('optimal_threshold', 0.5)}\n\n")
            f.write(f"Total emails: {len(results_df)}\n")
            f.write(f"Classified: {len(classified)}\n\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"FPR:       {fpr:.4f}\n\n")
            f.write(f"TP: {tp}  FP: {fp}\n")
            f.write(f"FN: {fn}  TN: {tn}\n")

        print(f"\n[+] Results: {results_file}")
        print(f"[+] Summary: {summary_file}")

    print(f"\n    Next step: Run reputation enrichment with Bawer model")


if __name__ == "__main__":
    main()
