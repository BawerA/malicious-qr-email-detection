"""
Quick capability test: does the Bawer model distinguish a legit ASCII URL
from its Cyrillic-homoglyph twin? (Limitation 2 verification)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd
from src.fetch.url_features import url_features_enhanced as extract

import os
MODEL = Path(os.environ.get("HG_MODEL",
    str(ROOT / "pipeline" / "Bawer_Dataset" / "models" / "bawer_63_XGBoost.joblib")))
print(f"MODEL: {MODEL.name}")
bundle = joblib.load(MODEL)
feature_names = bundle["feature_names"]
scaler = bundle["scaler"]
model = bundle["model"]
threshold = bundle.get("optimal_threshold", 0.5)

print(f"Model threshold: {threshold}")
print(f"unicode_homoglyph_score in model features? "
      f"{'unicode_homoglyph_score' in feature_names}")
print(f"homoglyph_score in model features? "
      f"{'homoglyph_score' in feature_names}")
print("=" * 70)

# (legit ASCII, Cyrillic-homoglyph twin)
pairs = [
    ("https://apple.com",  "https://аpple.com"),   # Cyrillic а
    ("https://paypal.com", "https://раypal.com"),  # Cyrillic р, а
    ("https://google.com", "https://gооgle.com"),  # Cyrillic о, о
    ("https://amazon.com", "https://аmazon.com"),   # Cyrillic а
]

def classify(url):
    feats = extract(str(url))
    X = pd.DataFrame([feats]).reindex(columns=feature_names, fill_value=0)
    Xs = scaler.transform(X.values)
    prob = float(model.predict_proba(Xs)[0][1])
    label = "malicious" if prob >= threshold else "benign"
    return prob, label, feats

for legit, fake in pairs:
    p1, l1, f1 = classify(legit)
    p2, l2, f2 = classify(fake)
    print(f"\nLegit : {legit}")
    print(f"        prob={p1:.4f}  -> {l1}")
    print(f"Cyrillic: {fake}")
    print(f"        prob={p2:.4f}  -> {l2}")
    print(f"        score change: {p2 - p1:+.4f}")
    # show any feature that changed notably
    diffs = []
    for k in f1:
        if isinstance(f1[k], (int, float)) and isinstance(f2.get(k), (int, float)):
            if abs(f2[k] - f1[k]) > 1e-9:
                diffs.append((k, f1[k], f2[k]))
    if diffs:
        print("        features that differ:")
        for k, a, b in diffs:
            print(f"           {k}: {a} -> {b}")
    else:
        print("        NO features differ between the two URLs")
