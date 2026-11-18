"""
Bawer_Dataset — Step 3: Train XGBoost (Same Architecture as Rivas)

Trains XGBoost on the Bawer augmented dataset using the SAME architecture,
hyperparameter grid, and feature set as the Rivas model. The only variable
changed is the training data (added bare-domain URLs).

Same as Rivas:
  - 63 lexical features
  - XGBoost with 72-combo grid search
  - Early stopping (1000 trees, patience 50)
  - scale_pos_weight for class imbalance
  - Threshold optimization on validation set

Input:  pipeline/Bawer_Dataset/data/bawer_train.csv
        pipeline/Bawer_Dataset/data/bawer_test.csv
        pipeline/Bawer_Dataset/data/bawer_val.csv
Output: pipeline/Bawer_Dataset/models/bawer_63_XGBoost.joblib
        pipeline/Bawer_Dataset/results/training_results.csv
"""

import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Input files
TRAIN_FILE = DATA_DIR / "bawer_train.csv"
TEST_FILE = DATA_DIR / "bawer_test.csv"
VAL_FILE = DATA_DIR / "bawer_val.csv"

# Columns to exclude from features
NON_FEATURE_COLS = ["url", "label", "label_binary", "label_bin", "url_type",
                    "source", "file", "decoded_url"]

SEED = 42

# SAME hyperparameter grid as Rivas model (72 combinations)
PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}


def get_feature_columns(df):
    """Get only the feature columns from a dataframe."""
    return [col for col in df.columns if col not in NON_FEATURE_COLS]


def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate a model and return metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
    }

    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["tn"] = int(tn)
        metrics["fn"] = int(fn)
    else:
        metrics["fpr"] = None

    try:
        metrics["auc"] = roc_auc_score(y, y_prob)
    except Exception:
        metrics["auc"] = None

    return metrics


def find_optimal_threshold(model, X_val, y_val):
    """Find the probability threshold that maximizes F1 on the validation set."""
    y_prob = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = 0
    all_results = []

    for threshold in np.arange(0.30, 0.70, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        all_results.append({"threshold": round(threshold, 2), "precision": prec, "recall": rec, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Print results near optimal threshold (+/- 2 steps)
    best_t = round(best_threshold, 2)
    print(f"\n    Threshold sweep (near optimal {best_t}):")
    print(f"    {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"    {'-'*42}")
    for r in all_results:
        if abs(r["threshold"] - best_t) <= 0.025:
            marker = " <-- BEST" if r["threshold"] == best_t else ""
            print(f"    {r['threshold']:>10.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}{marker}")

    return best_threshold, best_f1


def main():
    print("=" * 60)
    print("BAWER_DATASET — STEP 3: TRAIN XGBOOST")
    print("=" * 60)
    print(f"\n  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Same architecture as Rivas model — only dataset changed")

    # Check inputs
    for f in [TRAIN_FILE, TEST_FILE, VAL_FILE]:
        if not f.exists():
            print(f"\n[!] Missing: {f}")
            print("    Run '02_split_and_extract_features.py' first.")
            return

    # Load data
    print(f"\n[+] Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    val_df = pd.read_csv(VAL_FILE)

    feature_cols = get_feature_columns(train_df)
    print(f"    Features: {len(feature_cols)}")
    print(f"    Train: {len(train_df):,} | Test: {len(test_df):,} | Val: {len(val_df):,}")

    # Prepare X and y (binary classification)
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values

    y_train = train_df["label_binary"].values
    y_test = test_df["label_binary"].values
    y_val = val_df["label_binary"].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_val_enc = le.transform(y_val)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Class imbalance ratio
    n_benign = (y_train_enc == 0).sum()
    n_malicious = (y_train_enc == 1).sum()
    if n_malicious > 0 and n_benign > n_malicious:
        spw = n_benign / n_malicious
    elif n_benign > 0 and n_malicious > n_benign:
        spw = n_malicious / n_benign
    else:
        spw = 1.0

    print(f"    Benign: {n_benign:,} | Malicious: {n_malicious:,}")
    print(f"    scale_pos_weight: {spw:.2f}")

    # ── Hyperparameter Grid Search (72 combinations) ──
    grid_combos = list(ParameterGrid(PARAM_GRID))
    print(f"\n[+] Grid search ({len(grid_combos)} combinations)...")
    start_time = time.time()

    best_val_f1 = 0
    best_params = None
    best_model = None

    for params in tqdm(grid_combos, desc="  Grid search", unit="combo", ncols=80):
        model = XGBClassifier(
            n_estimators=1000,
            random_state=SEED,
            eval_metric="logloss",
            early_stopping_rounds=50,
            scale_pos_weight=spw,
            n_jobs=-1,
            verbosity=0,
            **params,
        )

        model.fit(
            X_train_scaled, y_train_enc,
            eval_set=[(X_val_scaled, y_val_enc)],
            verbose=False,
        )

        val_metrics = evaluate_model(model, X_val_scaled, y_val_enc)
        val_f1 = val_metrics["f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
            best_model = model

    grid_time = time.time() - start_time
    print(f"\n[+] Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"    Trees used: {best_model.best_iteration + 1} (of 1000 max)")
    print(f"    scale_pos_weight: {spw:.2f}")
    print(f"    Grid search time: {grid_time/60:.1f} minutes")

    # ── Threshold Optimization ──
    print(f"\n[+] Optimizing classification threshold...")
    optimal_threshold, threshold_f1 = find_optimal_threshold(
        best_model, X_val_scaled, y_val_enc
    )
    print(f"    Optimal threshold: {optimal_threshold:.2f}")
    print(f"    Val F1 at optimal: {threshold_f1:.4f}")

    # ── Test Set Results ──
    print(f"\n[+] Test set results (threshold={optimal_threshold:.2f}):")
    test_metrics = evaluate_model(
        best_model, X_test_scaled, y_test_enc, threshold=optimal_threshold
    )

    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    F1:        {test_metrics['f1']:.4f}")
    if test_metrics["fpr"] is not None:
        print(f"    FPR:       {test_metrics['fpr']:.4f}")
    if test_metrics["auc"] is not None:
        print(f"    AUC:       {test_metrics['auc']:.4f}")

    if test_metrics.get("tp") is not None:
        print(f"\n    Confusion Matrix:")
        print(f"      TP={test_metrics['tp']}  FP={test_metrics['fp']}")
        print(f"      FN={test_metrics['fn']}  TN={test_metrics['tn']}")

    # Default threshold comparison
    default_metrics = evaluate_model(best_model, X_test_scaled, y_test_enc, threshold=0.5)
    print(f"\n    (Default 0.50: Acc={default_metrics['accuracy']:.4f} "
          f"P={default_metrics['precision']:.4f} "
          f"R={default_metrics['recall']:.4f} "
          f"F1={default_metrics['f1']:.4f})")

    # ── Feature Importance ──
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print(f"\n[+] Top 15 features:")
    for fname, imp in feat_imp[:15]:
        print(f"    {fname:<35} {imp:.4f}")

    zero_feats = [f for f, i in feat_imp if i == 0.0]
    if zero_feats:
        print(f"\n[!] {len(zero_feats)} features with zero importance: {zero_feats}")

    # ── H1 Check ──
    print(f"\n{'='*60}")
    print("H1 CHECK (Bawer Model)")
    print(f"{'='*60}")
    print(f"  Precision: {test_metrics['precision']:.4f} {'>=90% PASS' if test_metrics['precision'] >= 0.90 else '<90% FAIL'}")
    print(f"  Recall:    {test_metrics['recall']:.4f} {'>=90% PASS' if test_metrics['recall'] >= 0.90 else '<90% FAIL'}")
    h1 = test_metrics['precision'] >= 0.90 and test_metrics['recall'] >= 0.90
    print(f"  >>> H1 {'SUPPORTED' if h1 else 'NOT SUPPORTED'} <<<")

    # ── Compare with Rivas Model ──
    rivas_model_path = Path(__file__).parent.parent / "models" / "our_63_XGBoost.joblib"
    if rivas_model_path.exists():
        rivas_bundle = joblib.load(rivas_model_path)
        rivas_metrics = rivas_bundle.get("metrics", {})

        print(f"\n{'='*60}")
        print("COMPARISON: Rivas Model vs Bawer Model (on their respective test sets)")
        print(f"{'='*60}")
        print(f"  {'Metric':<12} {'Rivas':>10} {'Bawer':>10} {'Diff':>12}")
        print(f"  {'-'*46}")
        for metric in ["accuracy", "precision", "recall", "f1", "fpr", "auc"]:
            rivas_val = rivas_metrics.get(metric)
            bawer_val = test_metrics.get(metric)
            if rivas_val is not None and bawer_val is not None:
                diff = bawer_val - rivas_val
                arrow = "+" if diff >= 0 else ""
                print(f"  {metric:<12} {rivas_val:>10.4f} {bawer_val:>10.4f} {arrow}{diff:>+10.4f}")

    # ── Save Model Bundle ──
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_cols,
        "feature_set": "bawer_63",
        "metrics": test_metrics,
        "optimal_threshold": optimal_threshold,
        "best_params": best_params,
    }

    model_filename = "bawer_63_XGBoost.joblib"
    joblib.dump(bundle, MODELS_DIR / model_filename)
    print(f"\n[+] Model saved: {MODELS_DIR / model_filename}")

    # ── Save Results CSV ──
    results_data = {
        "model": "XGBoost",
        "dataset": "Bawer (augmented)",
        "features": len(feature_cols),
        "train_size": len(train_df),
        "test_size": len(test_df),
        **test_metrics,
        "threshold": optimal_threshold,
        "best_params": str(best_params),
    }
    pd.DataFrame([results_data]).to_csv(RESULTS_DIR / "training_results.csv", index=False)
    print(f"[+] Results saved: {RESULTS_DIR / 'training_results.csv'}")

    total_time = time.time() - start_time
    print(f"\n    Total time: {total_time/60:.1f} minutes")
    print(f"\n    Next step: Run the email pipeline with the Bawer model")


if __name__ == "__main__":
    main()
