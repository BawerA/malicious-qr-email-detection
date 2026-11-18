"""
Step 3: Extract Our 63 Lexical Features

Extracts our 63-feature set on the same Rivas dataset,
so we can show the improvement from more features.

Uses the existing url_features.py from the project (extracts 68 raw features),
then drops 5 zero-importance features identified during model training:
  - has_punycode, domain_has_double_slash, has_unicode_encoding,
    has_mixed_case_domain, unicode_homoglyph_score

Extracts features from train, test, and validation sets separately.

Input:  pipeline/data/train_set.csv
        pipeline/data/test_set.csv
        pipeline/data/validation_set.csv
Output: pipeline/data/our_63_train.csv
        pipeline/data/our_63_test.csv
        pipeline/data/our_63_val.csv
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path so we can import url_features
PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch.url_features import url_features_enhanced as extract_url_features

DATA_DIR = PIPELINE_DIR / "data"

SPLITS = {
    "train": (DATA_DIR / "train_set.csv", DATA_DIR / "our_63_train.csv"),
    "test":  (DATA_DIR / "test_set.csv",  DATA_DIR / "our_63_test.csv"),
    "val":   (DATA_DIR / "validation_set.csv", DATA_DIR / "our_63_val.csv"),
}

# Features with zero importance — identified during XGBoost training
# These contribute nothing to classification and are removed for a cleaner model
DROP_FEATURES = [
    "has_punycode",
    "domain_has_double_slash",
    "has_unicode_encoding",
    "has_mixed_case_domain",
    "unicode_homoglyph_score",
]


def process_split(input_file, output_file, split_name):
    """Extract features for one data split."""

    if not input_file.exists():
        print(f"[!] {split_name} file not found: {input_file}")
        print("    Run '01_create_dataset.py' first.")
        return False

    print(f"\n[+] Processing {split_name} split: {input_file}")
    df = pd.read_csv(input_file)
    print(f"    {len(df):,} URLs loaded")

    print(f"    Extracting lexical features...")
    features_list = []
    errors = 0

    for url in tqdm(df["url"], desc=f"{split_name}"):
        try:
            features = extract_url_features(str(url))
            features_list.append(features)
        except Exception as e:
            features_list.append({})
            errors += 1

    features_df = pd.DataFrame(features_list).fillna(0)

    raw_count = len(features_df.columns)

    # Drop zero-importance features
    dropped = [f for f in DROP_FEATURES if f in features_df.columns]
    features_df = features_df.drop(columns=dropped, errors="ignore")

    print(f"    Extracted {raw_count} raw features, dropped {len(dropped)} zero-importance → {len(features_df.columns)} features")
    if errors > 0:
        print(f"    Errors: {errors:,} URLs failed feature extraction")

    result = pd.concat([
        df[["url", "label", "label_binary", "source"]].reset_index(drop=True),
        features_df.reset_index(drop=True),
    ], axis=1)

    result.to_csv(output_file, index=False)
    print(f"    Saved: {output_file}")
    print(f"    {len(result):,} rows x {len(features_df.columns)} features")
    return True


def main():
    print("=" * 60)
    print("STEP 3: EXTRACT OUR 63 LEXICAL FEATURES")
    print("=" * 60)
    print(f"  Dropping {len(DROP_FEATURES)} zero-importance features: {DROP_FEATURES}")

    success = True
    for split_name, (input_file, output_file) in SPLITS.items():
        if not process_split(input_file, output_file, split_name):
            success = False

    if success:
        print(f"\n[+] All splits processed.")
        print(f"    Features: {list(pd.read_csv(list(SPLITS.values())[0][1]).columns[4:])}")
        print("\n    Next step: Run '04_train_models.py'")


if __name__ == "__main__":
    main()
