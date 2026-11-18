"""
Bawer_Dataset — Step 2: Split and Extract 63 Features

Takes the augmented Bawer URL dataset, splits into train/test/val,
then extracts the same 63 lexical features used in the Rivas model.

Input:  pipeline/Bawer_Dataset/data/bawer_urls_raw.csv
Output: pipeline/Bawer_Dataset/data/bawer_train.csv     (75%)
        pipeline/Bawer_Dataset/data/bawer_test.csv      (15%)
        pipeline/Bawer_Dataset/data/bawer_val.csv        (10%)
"""

import sys
import time
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch.url_features import url_features_enhanced as extract_url_features

DATA_DIR = SCRIPT_DIR / "data"
SEED = 42

# Input
RAW_FILE = DATA_DIR / "bawer_urls_raw.csv"

# Output
TRAIN_FILE = DATA_DIR / "bawer_train.csv"
TEST_FILE = DATA_DIR / "bawer_test.csv"
VAL_FILE = DATA_DIR / "bawer_val.csv"

# Features with zero importance (same as Rivas model — keep identical for comparison)
DROP_FEATURES = [
    "has_punycode",
    "domain_has_double_slash",
    "has_unicode_encoding",
    "has_mixed_case_domain",
    "unicode_homoglyph_score",
]


def split_dataset(df):
    """Split dataset into train/test/val (75/15/10) with stratification."""
    print(f"\n[+] Splitting dataset (75/15/10, stratified)...")

    # Binary label for stratification
    df["label_binary"] = (df["label_bin"] == "malicious").astype(int)

    # First split: 75% train, 25% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df["label_binary"]
    )

    # Second split: 60% of temp = test (15% total), 40% of temp = val (10% total)
    test_df, val_df = train_test_split(
        temp_df, test_size=0.40, random_state=SEED, stratify=temp_df["label_binary"]
    )

    print(f"    Train: {len(train_df):,} ({len(train_df)/len(df)*100:.0f}%)")
    print(f"    Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.0f}%)")
    print(f"    Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.0f}%)")

    # Check balance per split
    for name, split_df in [("Train", train_df), ("Test", test_df), ("Val", val_df)]:
        mal_pct = split_df["label_binary"].mean() * 100
        bare_benign = ((split_df["label_bin"] == "benign") &
                       (split_df["url_type"] == "bare_domain")).sum()
        total_benign = (split_df["label_bin"] == "benign").sum()
        bare_pct = bare_benign / total_benign * 100 if total_benign > 0 else 0
        print(f"    {name}: {mal_pct:.1f}% malicious, "
              f"{bare_pct:.1f}% bare-domain benign")

    return train_df, test_df, val_df


def extract_features(df, output_file, split_name):
    """Extract 63 lexical features for a data split."""
    print(f"\n[+] Extracting features for {split_name} ({len(df):,} URLs)...")

    features_list = []
    errors = 0

    for url in tqdm(df["url"], desc=split_name):
        try:
            features = extract_url_features(str(url))
            features_list.append(features)
        except Exception:
            features_list.append({})
            errors += 1

    features_df = pd.DataFrame(features_list).fillna(0)
    raw_count = len(features_df.columns)

    # Drop zero-importance features (same as Rivas model)
    dropped = [f for f in DROP_FEATURES if f in features_df.columns]
    features_df = features_df.drop(columns=dropped, errors="ignore")

    print(f"    {raw_count} raw → dropped {len(dropped)} → {len(features_df.columns)} features")
    if errors > 0:
        print(f"    Errors: {errors:,} URLs failed")

    # Combine metadata + features
    meta_cols = ["url", "label_bin", "label_binary", "url_type", "source"]
    meta = df[meta_cols].reset_index(drop=True)

    # Rename label_bin to label for compatibility with training script
    meta = meta.rename(columns={"label_bin": "label"})

    result = pd.concat([meta, features_df.reset_index(drop=True)], axis=1)
    result.to_csv(output_file, index=False)

    print(f"    Saved: {output_file.name} ({len(result):,} rows x {len(features_df.columns)} features)")
    return len(features_df.columns)


def main():
    print("=" * 60)
    print("BAWER_DATASET — STEP 2: SPLIT & EXTRACT 63 FEATURES")
    print("=" * 60)
    print(f"\n  Date: {time.strftime('%Y-%m-%d %H:%M')}")

    # Load raw dataset
    if not RAW_FILE.exists():
        print(f"\n[!] Raw dataset not found: {RAW_FILE}")
        print("    Run '01_collect_urls.py' first.")
        return

    print(f"\n[+] Loading: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    print(f"    {len(df):,} URLs loaded")
    print(f"    Benign: {(df['label_bin']=='benign').sum():,}  "
          f"Malicious: {(df['label_bin']=='malicious').sum():,}")

    # Split
    train_df, test_df, val_df = split_dataset(df)

    # Extract features for each split
    start = time.time()

    splits = [
        (train_df, TRAIN_FILE, "train"),
        (test_df,  TEST_FILE,  "test"),
        (val_df,   VAL_FILE,   "val"),
    ]

    for split_df, output_file, split_name in splits:
        n_features = extract_features(split_df, output_file, split_name)

    elapsed = time.time() - start
    rate = len(df) / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed/60:.1f} minutes ({rate:.0f} URLs/sec)")
    print(f"  Features per URL: {n_features}")
    print(f"  Files saved:")
    print(f"    {TRAIN_FILE.name}")
    print(f"    {TEST_FILE.name}")
    print(f"    {VAL_FILE.name}")
    print(f"\n    Next step: Run '03_train_model.py'")


if __name__ == "__main__":
    main()
