"""
Step 7: Email QR Code Detection Pipeline

Parses .eml email files, extracts embedded images, detects QR codes,
decodes URLs, and classifies them using our trained models.

This is the core email-aware pipeline that differentiates our research:
  Email → Extract Images → Detect QR → Decode URL → Extract Features → Classify

Input:  pipeline/data/test_emails/benign/*.eml
        pipeline/data/test_emails/malicious/*.eml
        pipeline/models/*.joblib
Output: pipeline/results/email_pipeline_results.csv
        pipeline/results/email_pipeline_summary.txt
"""

import sys
import io
import os
import csv
import email
import joblib
import pandas as pd
import numpy as np
from email import policy
from pathlib import Path
from PIL import Image

# Add project root to path for imports
PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch.qr_decode import decode_qr_bytes
from src.fetch.url_features import url_features_enhanced as extract_url_features

DATA_DIR = PIPELINE_DIR / "data"
MODELS_DIR = PIPELINE_DIR / "models"
RESULTS_DIR = PIPELINE_DIR / "results"
EMAIL_DIR = DATA_DIR / "test_emails"
BENIGN_DIR = EMAIL_DIR / "benign"
MALICIOUS_DIR = EMAIL_DIR / "malicious"
MANIFEST_FILE = EMAIL_DIR / "email_manifest.csv"

RESULTS_DIR.mkdir(exist_ok=True)


def extract_images_from_eml(eml_path):
    """Extract all image attachments/inline images from an .eml file."""
    images = []

    with open(eml_path, "r", encoding="utf-8", errors="replace") as f:
        msg = email.message_from_file(f, policy=policy.default)

    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type.startswith("image/"):
            payload = part.get_payload(decode=True)
            if payload:
                images.append({
                    "content_type": content_type,
                    "filename": part.get_filename() or "unknown",
                    "data": payload,
                })

    return images


def decode_qr_from_image_bytes(image_bytes):
    """Attempt to decode a QR code from raw image bytes."""
    try:
        decoded_url, engine = decode_qr_bytes(image_bytes)
        return decoded_url, engine
    except Exception:
        return None, "error"


def classify_url(url, model_bundle):
    """Classify a URL using a trained model bundle."""
    try:
        # Extract features
        features = extract_url_features(str(url))

        # Create DataFrame with correct feature alignment
        feature_names = model_bundle["feature_names"]
        X = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)

        # Scale
        scaler = model_bundle["scaler"]
        X_scaled = scaler.transform(X.values)

        # Predict using optimal threshold from training
        model = model_bundle["model"]
        threshold = model_bundle.get("optimal_threshold", 0.5)

        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            prob = float(probs[1]) if len(probs) == 2 else float(max(probs))
            # Apply optimized threshold instead of default 0.5
            prediction = 1 if prob >= threshold else 0
        else:
            prediction = model.predict(X_scaled)[0]

        # Decode label
        label_encoder = model_bundle["label_encoder"]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        return predicted_label, prob

    except Exception as e:
        return "error", None


def load_best_model():
    """Load the best performing model (63-feature XGBoost by default)."""
    # Try our 63-feature model first (expected best), then fall back
    preferred_models = [
        "our_63_XGBoost.joblib",
    ]

    for model_file in preferred_models:
        model_path = MODELS_DIR / model_file
        if model_path.exists():
            print(f"  Loading model: {model_file}")
            return joblib.load(model_path), model_file

    # Fall back to any available model
    model_files = list(MODELS_DIR.glob("*.joblib"))
    if model_files:
        model_file = model_files[0]
        print(f"  Loading model: {model_file.name}")
        return joblib.load(model_file), model_file.name

    return None, None


def process_email_directory(email_dir, true_label, model_bundle, model_name):
    """Process all .eml files in a directory."""
    results = []
    eml_files = sorted(email_dir.glob("*.eml"))

    if not eml_files:
        print(f"    No .eml files found in {email_dir}")
        return results

    success = 0
    no_qr = 0
    errors = 0

    for eml_path in eml_files:
        try:
            # Step 1: Extract images from email
            images = extract_images_from_eml(eml_path)

            if not images:
                no_qr += 1
                results.append({
                    "email_file": eml_path.name,
                    "true_label": true_label,
                    "predicted_label": "no_image",
                    "probability": None,
                    "decoded_url": None,
                    "qr_engine": None,
                    "model_used": model_name,
                    "num_images": 0,
                })
                continue

            # Step 2: Try to decode QR from each image
            decoded_url = None
            qr_engine = None

            for img_data in images:
                url, engine = decode_qr_from_image_bytes(img_data["data"])
                if url:
                    decoded_url = url
                    qr_engine = engine
                    break

            if not decoded_url:
                no_qr += 1
                results.append({
                    "email_file": eml_path.name,
                    "true_label": true_label,
                    "predicted_label": "no_qr_detected",
                    "probability": None,
                    "decoded_url": None,
                    "qr_engine": None,
                    "model_used": model_name,
                    "num_images": len(images),
                })
                continue

            # Step 3: Classify the decoded URL
            predicted_label, prob = classify_url(decoded_url, model_bundle)

            success += 1
            results.append({
                "email_file": eml_path.name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "probability": prob,
                "decoded_url": decoded_url,
                "qr_engine": qr_engine,
                "model_used": model_name,
                "num_images": len(images),
            })

        except Exception as e:
            errors += 1
            results.append({
                "email_file": eml_path.name,
                "true_label": true_label,
                "predicted_label": "error",
                "probability": None,
                "decoded_url": None,
                "qr_engine": None,
                "model_used": model_name,
                "num_images": 0,
            })

    print(f"    Processed: {len(eml_files)} | Decoded: {success} | No QR: {no_qr} | Errors: {errors}")
    return results


def main():
    print("=" * 60)
    print("STEP 7: EMAIL QR CODE DETECTION PIPELINE")
    print("=" * 60)
    print()

    # Check for test emails
    if not EMAIL_DIR.exists():
        print(f"[!] Email directory not found: {EMAIL_DIR}")
        print("    Run '06_generate_email_qr_codes.py' first.")
        return

    # Load model
    print("[+] Loading trained model...")
    model_bundle, model_name = load_best_model()
    if model_bundle is None:
        print("[!] No trained models found in:", MODELS_DIR)
        print("    Run '04_train_models.py' first.")
        return

    print(f"    Model: {model_name}")
    print(f"    Features: {len(model_bundle['feature_names'])}")
    print()

    all_results = []

    # Process benign emails
    if BENIGN_DIR.exists():
        print(f"[+] Processing benign emails from: {BENIGN_DIR}")
        benign_results = process_email_directory(
            BENIGN_DIR, "benign", model_bundle, model_name
        )
        all_results.extend(benign_results)
    else:
        print(f"[!] Benign email directory not found: {BENIGN_DIR}")

    # Process malicious emails
    if MALICIOUS_DIR.exists():
        print(f"\n[+] Processing malicious emails from: {MALICIOUS_DIR}")
        malicious_results = process_email_directory(
            MALICIOUS_DIR, "malicious", model_bundle, model_name
        )
        all_results.extend(malicious_results)
    else:
        print(f"[!] Malicious email directory not found: {MALICIOUS_DIR}")

    if not all_results:
        print("\n[!] No results to report.")
        return

    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = RESULTS_DIR / "email_pipeline_results.csv"
    results_df.to_csv(results_file, index=False)

    # Calculate metrics for successfully classified emails
    classified = results_df[results_df["predicted_label"].isin(["benign", "malicious"])]

    print(f"\n{'='*60}")
    print("EMAIL PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"\n  Total emails processed: {len(results_df)}")
    print(f"  Successfully classified: {len(classified)}")
    print(f"  No QR detected: {len(results_df[results_df['predicted_label'] == 'no_qr_detected'])}")
    print(f"  No image found: {len(results_df[results_df['predicted_label'] == 'no_image'])}")
    print(f"  Errors: {len(results_df[results_df['predicted_label'] == 'error'])}")

    if len(classified) > 0:
        # Accuracy
        correct = (classified["true_label"] == classified["predicted_label"]).sum()
        accuracy = correct / len(classified)

        # Per-class metrics
        true_pos = ((classified["true_label"] == "malicious") & (classified["predicted_label"] == "malicious")).sum()
        false_pos = ((classified["true_label"] == "benign") & (classified["predicted_label"] == "malicious")).sum()
        true_neg = ((classified["true_label"] == "benign") & (classified["predicted_label"] == "benign")).sum()
        false_neg = ((classified["true_label"] == "malicious") & (classified["predicted_label"] == "benign")).sum()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0

        print(f"\n  Classification Metrics (on {len(classified)} emails):")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    FPR:       {fpr:.4f}")

        print(f"\n  Confusion Matrix:")
        print(f"    True Positives  (malicious -> malicious): {true_pos}")
        print(f"    True Negatives  (benign -> benign):       {true_neg}")
        print(f"    False Positives (benign -> malicious):    {false_pos}")
        print(f"    False Negatives (malicious -> benign):    {false_neg}")

        # Save summary
        summary_file = RESULTS_DIR / "email_pipeline_summary.txt"
        with open(summary_file, "w") as f:
            f.write("EMAIL QR CODE DETECTION PIPELINE — RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Features: {len(model_bundle['feature_names'])}\n\n")
            f.write(f"Total emails: {len(results_df)}\n")
            f.write(f"Classified: {len(classified)}\n\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"FPR:       {fpr:.4f}\n\n")
            f.write(f"TP: {true_pos}  FP: {false_pos}\n")
            f.write(f"FN: {false_neg}  TN: {true_neg}\n")

        print(f"\n[+] Results saved to: {results_file}")
        print(f"[+] Summary saved to: {summary_file}")

    print("\n    Next step: Run '08_reputation_enrichment.py'")


if __name__ == "__main__":
    main()
