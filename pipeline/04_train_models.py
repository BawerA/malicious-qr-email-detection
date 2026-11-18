"""
Step 4: Train XGBoost (Optimized) — 1 Model x 2 Feature Sets

Trains XGBoost on both feature sets for direct comparison with Rivas:
  - Run A: Rivas's 24 features (baseline — replicate Rivas's XGBoost)
  - Run B: Our 63 features (improvement)

Optimizations:
  1. scale_pos_weight = class ratio x 1.5 recall boost (see RECALL_BOOST)
  2. Early stopping with validation set (1000 max trees, stops when no improvement)
  3. Hyperparameter grid search (72 combinations)
  4. Threshold optimization on validation set (maximize F1)
  5. Feature importance analysis

Uses pre-split train/test/validation sets from Step 1 (75/15/10).

Input:  pipeline/data/rivas_24_train.csv, rivas_24_test.csv, rivas_24_val.csv
        pipeline/data/our_63_train.csv, our_63_test.csv, our_63_val.csv
Output: pipeline/models/*.joblib
        pipeline/results/training_results.csv
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

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
MODELS_DIR = PIPELINE_DIR / "models"
RESULTS_DIR = PIPELINE_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Feature set files (train / test / val)
FEATURE_SETS = {
    "rivas_24": {
        "train": DATA_DIR / "rivas_24_train.csv",
        "test":  DATA_DIR / "rivas_24_test.csv",
        "val":   DATA_DIR / "rivas_24_val.csv",
    },
    "our_63": {
        "train": DATA_DIR / "our_63_train.csv",
        "test":  DATA_DIR / "our_63_test.csv",
        "val":   DATA_DIR / "our_63_val.csv",
    },
}

# Columns to exclude from features
NON_FEATURE_COLS = ["url", "label", "label_binary", "source", "file", "decoded_url"]

SEED = 42

# Hyperparameter grid for tuning (72 combinations)
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
    print(f"\n      Threshold sweep (near optimal {best_t}):")
    print(f"      {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"      {'-'*42}")
    for r in all_results:
        if abs(r["threshold"] - best_t) <= 0.025:
            marker = " <-- BEST" if r["threshold"] == best_t else ""
            print(f"      {r['threshold']:>10.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}{marker}")

    return best_threshold, best_f1


def run_experiment(feature_set_name, files, use_binary=True):
    """Run optimized XGBoost training pipeline."""

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {feature_set_name}")
    print(f"{'='*60}")

    # Load pre-split data
    train_df = pd.read_csv(files["train"])
    test_df = pd.read_csv(files["test"])
    val_df = pd.read_csv(files["val"])

    feature_cols = get_feature_columns(train_df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | Val: {len(val_df):,}")

    # Prepare X and y
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values

    if use_binary:
        y_train = train_df["label_binary"].values
        y_test = test_df["label_binary"].values
        y_val = val_df["label_binary"].values
        label_name = "binary"
    else:
        y_train = train_df["label"].values
        y_test = test_df["label"].values
        y_val = val_df["label"].values
        label_name = "multiclass"

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

    # Calculate class imbalance ratio, then boost for recall
    n_benign = (y_train_enc == 0).sum()
    n_malicious = (y_train_enc == 1).sum()
    if n_malicious > 0 and n_benign > n_malicious:
        base_spw = n_benign / n_malicious
    elif n_benign > 0 and n_malicious > n_benign:
        base_spw = n_malicious / n_benign
    else:
        base_spw = 1.0

    # Recall boost: multiplier above base class imbalance ratio
    # Higher = more recall, lower precision (security-focused trade-off)
    RECALL_BOOST = 1.5
    spw = base_spw * RECALL_BOOST

    print(f"  Classification: {label_name}")
    print(f"  Base scale_pos_weight: {base_spw:.2f}")
    print(f"  Boosted scale_pos_weight: {spw:.2f} (boost={RECALL_BOOST}x)")

    # ── Hyperparameter Grid Search ──
    grid_combos = list(ParameterGrid(PARAM_GRID))
    print(f"\n  [+] Hyperparameter grid search ({len(grid_combos)} combinations)...")

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

    print(f"\n  [+] Best hyperparameters:")
    for k, v in best_params.items():
        print(f"      {k}: {v}")
    print(f"      n_estimators used: {best_model.best_iteration + 1} (of 1000 max)")
    print(f"      scale_pos_weight: {spw:.2f}")
    print(f"      Best Val F1: {best_val_f1:.4f}")

    # ── Threshold optimization ──
    print(f"\n  [+] Optimizing classification threshold...")
    optimal_threshold, threshold_f1 = find_optimal_threshold(best_model, X_val_scaled, y_val_enc)
    print(f"      Optimal threshold: {optimal_threshold:.2f} (default 0.50)")
    print(f"      Val F1 at optimal: {threshold_f1:.4f}")

    # ── Evaluate on test set ──
    print(f"\n  [+] Test set results (threshold={optimal_threshold:.2f}):")
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test_enc, threshold=optimal_threshold)
    test_metrics["model"] = "XGBoost"

    print(f"      Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"      Precision: {test_metrics['precision']:.4f}")
    print(f"      Recall:    {test_metrics['recall']:.4f}")
    print(f"      F1:        {test_metrics['f1']:.4f}")
    if test_metrics["fpr"] is not None:
        print(f"      FPR:       {test_metrics['fpr']:.4f}")
    if test_metrics["auc"] is not None:
        print(f"      AUC:       {test_metrics['auc']:.4f}")

    # Default threshold comparison
    default_metrics = evaluate_model(best_model, X_test_scaled, y_test_enc, threshold=0.5)
    print(f"\n      (Default 0.50 threshold: Acc={default_metrics['accuracy']:.4f} P={default_metrics['precision']:.4f} R={default_metrics['recall']:.4f} F1={default_metrics['f1']:.4f})")

    # Validation metrics
    val_final = evaluate_model(best_model, X_val_scaled, y_val_enc, threshold=optimal_threshold)
    print(f"      Val Accuracy: {val_final['accuracy']:.4f} | Val F1: {val_final['f1']:.4f}")

    # ── Feature importance ──
    if len(feature_cols) > 24:
        importances = best_model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

        print(f"\n  [+] Top 15 most important features:")
        for fname, imp in feat_imp[:15]:
            print(f"      {fname:<35} {imp:.4f}")

        print(f"\n  [+] Bottom 5 least important features:")
        for fname, imp in feat_imp[-5:]:
            print(f"      {fname:<35} {imp:.4f}")

        zero_feats = [f for f, i in feat_imp if i == 0.0]
        if zero_feats:
            print(f"\n  [!] {len(zero_feats)} features with zero importance: {zero_feats}")

    # Build results
    test_metrics["feature_set"] = feature_set_name
    test_metrics["num_features"] = len(feature_cols)
    test_metrics["classification"] = label_name
    test_metrics["val_accuracy"] = val_final["accuracy"]
    test_metrics["val_f1"] = val_final["f1"]
    test_metrics["threshold"] = optimal_threshold
    test_metrics["best_params"] = str(best_params)

    # Save model bundle
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_cols,
        "feature_set": feature_set_name,
        "metrics": test_metrics,
        "optimal_threshold": optimal_threshold,
        "best_params": best_params,
    }
    model_filename = f"{feature_set_name}_XGBoost.joblib"
    joblib.dump(bundle, MODELS_DIR / model_filename)
    print(f"\n      Saved: {model_filename}")

    return [test_metrics]


def main():
    print("=" * 60)
    print("STEP 4: TRAIN XGBOOST (Optimized) — 2 Feature Sets")
    print("=" * 60)

    grid_count = len(list(ParameterGrid(PARAM_GRID)))
    print(f"  Grid search: {grid_count} combinations per feature set ({grid_count * 2} total)")

    # Check inputs
    for fs_name, files in FEATURE_SETS.items():
        for split, path in files.items():
            if not path.exists():
                print(f"[!] Missing: {path}")
                if "24" in fs_name:
                    print("    Run '02_extract_features_24.py' first.")
                else:
                    print("    Run '03_extract_features_63.py' first.")
                return

    start_time = time.time()
    all_results = []

    # Run A: Rivas's 24 features (binary classification)
    results_24 = run_experiment("rivas_24", FEATURE_SETS["rivas_24"], use_binary=True)
    all_results.extend(results_24)

    # Run B: Our 63 features (binary classification)
    results_63 = run_experiment("our_63", FEATURE_SETS["our_63"], use_binary=True)
    all_results.extend(results_63)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "training_results.csv", index=False)

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Rivas 24 Features vs Our 63 Features")
    print("=" * 90)
    print(f"\n{'Model':<12} {'Features':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10} {'AUC':>10} {'Threshold':>10}")
    print("-" * 96)

    for _, row in results_df.iterrows():
        fpr_str = f"{row['fpr']:.4f}" if row['fpr'] is not None else "N/A"
        auc_str = f"{row['auc']:.4f}" if row.get('auc') is not None else "N/A"
        thr_str = f"{row.get('threshold', 0.5):.2f}"
        print(f"{row['model']:<12} {row['feature_set']:<12} {row['accuracy']:>10.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f} {fpr_str:>10} {auc_str:>10} {thr_str:>10}")

    print(f"\n  Total time: {minutes}m {seconds}s")
    print(f"[+] Results saved to: {RESULTS_DIR / 'training_results.csv'}")
    print("\n    Next step: Run '05_evaluate_models.py'")


if __name__ == "__main__":
    main()
