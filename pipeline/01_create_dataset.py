"""
Step 1: Create Dataset — Replicating Rivas et al. (2025)

Downloads and merges URL datasets from the same 5 sources Rivas used,
creating a unified dataset for training ML models.

Rivas's 5 sources (from references [34]-[39]):
1. [37] Kaggle "Malicious URLs Dataset" (sid321axn) — ~651K URLs
2. [38] Kaggle "Phishing Site URLs" (taruntiwarihp) — phishing URLs
3. [39] Kaggle "Malicious and Benign URLs" (siddharthkumar25) — supplementary
4. [34] PhishStorm (Marchal et al., 2014) — phishing URLs
5. [35] ISCX-URL-2016 (Mamun et al., 2016) — benign/phishing/malware/defacement

All sources merged, deduplicated, labels standardized to 4 classes:
  benign, defacement, phishing, malware

Split: 70% train / 20% test / 10% validation (stratified)

Output: pipeline/data/rivas_dataset.csv        (full merged dataset)
        pipeline/data/train_set.csv             (70%)
        pipeline/data/test_set.csv              (20%)
        pipeline/data/validation_set.csv        (10%)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_downloads"

OUTPUT_FILE = DATA_DIR / "rivas_dataset.csv"
TRAIN_FILE = DATA_DIR / "train_set.csv"
TEST_FILE = DATA_DIR / "test_set.csv"
VAL_FILE = DATA_DIR / "validation_set.csv"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

SEED = 42


# ═══════════════════════════════════════════════════════════════════
# SOURCE 1: Kaggle "Malicious URLs Dataset" (sid321axn) [37]
# Columns: url, type
# ═══════════════════════════════════════════════════════════════════
def load_source1_kaggle_malicious_urls():
    """Rivas ref [37]: ~651K URLs: benign, defacement, phishing, malware."""
    filepath = RAW_DIR / "Kaggle_Malicious_URLs_Dataset.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "malicious_phish.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "malicious_urls.csv"

    if not filepath.exists():
        print("[!] SOURCE 1 NOT FOUND — Kaggle Malicious URLs (sid321axn)")
        print("    *** THIS IS THE PRIMARY DATASET — REQUIRED ***")
        return None

    print("[+] SOURCE 1: Kaggle Malicious URLs (sid321axn) [ref 37]")
    df = pd.read_csv(filepath)

    # Columns: url, type
    df = df.rename(columns={"type": "label"})
    df["source"] = "kaggle_sid321axn"

    print(f"    Loaded {len(df):,} URLs")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")

    return df[["url", "label", "source"]]


# ═══════════════════════════════════════════════════════════════════
# SOURCE 2: Kaggle "Phishing Site URLs" (taruntiwarihp) [38]
# Columns: URL, Label (values: "good", "bad")
# ═══════════════════════════════════════════════════════════════════
def load_source2_kaggle_phishing_sites():
    """Rivas ref [38]: Phishing and legitimate URLs."""
    filepath = RAW_DIR / "Kaggle_Phishing_Site_URLs.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "phishing_site_urls.csv"

    if not filepath.exists():
        print("[!] SOURCE 2 NOT FOUND — Kaggle Phishing Site URLs (taruntiwarihp)")
        return None

    print("[+] SOURCE 2: Kaggle Phishing Site URLs (taruntiwarihp) [ref 38]")
    df = pd.read_csv(filepath)

    # Columns: URL, Label (good/bad)
    df = df.rename(columns={"URL": "url", "Label": "label"})

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].replace({
        "good": "benign",
        "bad": "phishing",
    })

    df["source"] = "kaggle_taruntiwarihp"

    print(f"    Loaded {len(df):,} URLs")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")

    return df[["url", "label", "source"]]


# ═══════════════════════════════════════════════════════════════════
# SOURCE 3: Kaggle "Malicious and Benign URLs" (siddharthkumar25) [39]
# Columns: (index), url, label, result
# ═══════════════════════════════════════════════════════════════════
def load_source3_kaggle_malicious_benign():
    """Rivas ref [39]: Supplementary benign and malicious URLs."""
    filepath = RAW_DIR / "Kaggle_Malicious_and_Benign_URLs.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "malicious_and_benign_urls.csv"

    if not filepath.exists():
        print("[!] SOURCE 3 NOT FOUND — Kaggle Malicious and Benign URLs (siddharthkumar25)")
        return None

    print("[+] SOURCE 3: Kaggle Malicious & Benign URLs (siddharthkumar25) [ref 39]")
    df = pd.read_csv(filepath)

    # Columns: (unnamed index), url, label, result
    # label values: "benign", "malicious", etc.
    if "url" not in df.columns:
        # Try to find url column
        for col in df.columns:
            if "url" in col.lower():
                df = df.rename(columns={col: "url"})
                break

    if "label" not in df.columns:
        for col in df.columns:
            if col.lower() in ["label", "type", "class", "target"]:
                df = df.rename(columns={col: "label"})
                break

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].replace({
        "malicious": "malware",
    })

    df["source"] = "kaggle_siddharthkumar25"

    print(f"    Loaded {len(df):,} URLs")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")

    return df[["url", "label", "source"]]


# ═══════════════════════════════════════════════════════════════════
# SOURCE 4: PhishStorm (Marchal et al., 2014) [34]
# Columns: domain, ranking, ..., label (values: 1.0 = phishing, 0.0 = legit)
# ═══════════════════════════════════════════════════════════════════
def load_source4_phishstorm():
    """Rivas ref [34]: 96K URLs — 48K phishing + 48K legitimate."""
    filepath = RAW_DIR / "PhishStorm.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "phishstorm.csv"

    if not filepath.exists():
        print("[!] SOURCE 4 NOT FOUND — PhishStorm (Marchal et al., 2014)")
        print("    [OPTIONAL] — Pipeline works without this source.")
        return None

    print("[+] SOURCE 4: PhishStorm (Marchal et al., 2014) [ref 34]")
    df = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")

    # URL is in the "domain" column
    if "domain" in df.columns:
        df = df.rename(columns={"domain": "url"})

    # Label column: 1.0 = phishing, 0.0 = legitimate
    if "label" in df.columns:
        df["label"] = df["label"].apply(
            lambda x: "phishing" if float(x) >= 1.0 else "benign"
        )
    else:
        # Try to find label-like column
        for col in df.columns:
            if col.lower() in ["label", "type", "class"]:
                df = df.rename(columns={col: "label"})
                break

    df["source"] = "phishstorm"

    print(f"    Loaded {len(df):,} URLs")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")

    return df[["url", "label", "source"]]


# ═══════════════════════════════════════════════════════════════════
# SOURCE 5: ISCX-URL-2016 (Mamun et al., 2016) [35]
# Columns: 79 features + URL_Type_obf_Type (label)
# No explicit URL column — features are pre-extracted
# ═══════════════════════════════════════════════════════════════════
def load_source5_iscx_url_2016():
    """Rivas ref [35]: 57K URLs — benign, spam, phishing, malware, defacement."""
    filepath = RAW_DIR / "ISCX_URL_2016.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "iscx_url_2016.csv"
    if not filepath.exists():
        filepath = RAW_DIR / "All.csv"

    if not filepath.exists():
        print("[!] SOURCE 5 NOT FOUND — ISCX-URL-2016 (Mamun et al., 2016)")
        print("    [OPTIONAL] — Pipeline works without this source.")
        return None

    print("[+] SOURCE 5: ISCX-URL-2016 (Mamun et al., 2016) [ref 35]")
    df = pd.read_csv(filepath)

    # This dataset has pre-extracted features but NO raw URL column.
    # The label is in "URL_Type_obf_Type" (e.g., "benign", "Defacement", etc.)
    label_col = None
    for col in df.columns:
        if "type" in col.lower():
            label_col = col
            break

    if label_col is None:
        print("    [!] Could not find label column. Skipping.")
        return None

    df = df.rename(columns={label_col: "label"})
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].replace({
        "spam": "phishing",  # Rivas used 4 classes, merge spam → phishing
    })

    # ISCX-URL-2016 has no raw URL column — it only has pre-extracted features.
    # We cannot use it for our feature extraction pipeline (Steps 2-3 need raw URLs).
    # However, Source 1 (sid321axn) was BUILT from this dataset, so the URLs are
    # already included. We'll skip this source but report it.
    print(f"    Loaded {len(df):,} rows")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")
    print("    NOTE: This dataset contains pre-extracted features, not raw URLs.")
    print("          Source 1 (sid321axn) was built from ISCX-URL-2016,")
    print("          so these URLs are already included. Skipping to avoid")
    print("          adding rows without raw URLs for feature extraction.")

    return None


# ═══════════════════════════════════════════════════════════════════
# MERGE, CLEAN, AND SPLIT
# ═══════════════════════════════════════════════════════════════════
def merge_and_clean(dataframes):
    """Merge all datasets, deduplicate, and clean."""

    valid_dfs = [df for df in dataframes if df is not None]

    if not valid_dfs:
        print("[!] No datasets loaded. Cannot create merged dataset.")
        return None

    print(f"\n{'='*60}")
    print("MERGING ALL SOURCES")
    print(f"{'='*60}")

    merged = pd.concat(valid_dfs, ignore_index=True)
    print(f"  Total before cleaning: {len(merged):,}")

    # Show per-source counts
    print(f"\n  Per-source breakdown:")
    for source, count in merged["source"].value_counts().items():
        print(f"    {source}: {count:,}")

    # Remove duplicates (keep first occurrence)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["url"], keep="first")
    print(f"\n  Removed {before - len(merged):,} duplicate URLs")

    # Remove rows with missing/empty URLs
    merged = merged.dropna(subset=["url"])
    merged = merged[merged["url"].astype(str).str.strip().str.len() > 0]

    # Standardize labels to Rivas's 4 classes
    label_map = {
        "benign": "benign",
        "legitimate": "benign",
        "safe": "benign",
        "good": "benign",
        "defacement": "defacement",
        "phishing": "phishing",
        "phish": "phishing",
        "malware": "malware",
        "malicious": "malware",
        "bad": "malware",
    }
    merged["label"] = merged["label"].astype(str).str.lower().str.strip().map(label_map)
    merged = merged.dropna(subset=["label"])

    # Binary label for our model
    merged["label_binary"] = merged["label"].apply(
        lambda x: "benign" if x == "benign" else "malicious"
    )

    print(f"\n  Final dataset: {len(merged):,} URLs")
    print(f"\n  Label distribution (4-class, Rivas-style):")
    for label, count in merged["label"].value_counts().items():
        pct = count / len(merged) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    print(f"\n  Label distribution (binary, our model):")
    for label, count in merged["label_binary"].value_counts().items():
        pct = count / len(merged) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    print(f"\n  Per-source contribution (after dedup):")
    for source, count in merged["source"].value_counts().items():
        pct = count / len(merged) * 100
        print(f"    {source}: {count:,} ({pct:.1f}%)")

    # ── Downsample to ~650K to match Rivas's dataset size ──
    TARGET_SIZE = 650_000
    if len(merged) > TARGET_SIZE:
        print(f"\n  Downsampling to ~{TARGET_SIZE:,} URLs (matching Rivas's ~600K+)...")
        merged = merged.sample(n=TARGET_SIZE, random_state=SEED, ignore_index=True)

        print(f"  Downsampled dataset: {len(merged):,} URLs")
        print(f"\n  Label distribution after downsample (4-class):")
        for label, count in merged["label"].value_counts().items():
            pct = count / len(merged) * 100
            print(f"    {label}: {count:,} ({pct:.1f}%)")

        print(f"\n  Label distribution after downsample (binary):")
        for label, count in merged["label_binary"].value_counts().items():
            pct = count / len(merged) * 100
            print(f"    {label}: {count:,} ({pct:.1f}%)")

    return merged


def split_dataset(df):
    """Split into 75% train / 15% test / 10% validation (stratified)."""

    print(f"\n{'='*60}")
    print("SPLITTING DATASET: 75% Train / 15% Test / 10% Validation")
    print(f"{'='*60}")

    # First split: 75% train, 25% temp (which becomes 15% test + 10% val)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.25,
        random_state=SEED,
        stratify=df["label_binary"],
    )

    # Second split: split the 25% into 15% test + 10% val (i.e., 3/5 and 2/5)
    test_df, val_df = train_test_split(
        temp_df,
        test_size=2/5,  # 10% of total = 2/5 of the 25%
        random_state=SEED,
        stratify=temp_df["label_binary"],
    )

    print(f"\n  Train set:      {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test set:       {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")

    # Show class balance per split
    for name, split_df in [("Train", train_df), ("Test", test_df), ("Validation", val_df)]:
        benign = (split_df["label_binary"] == "benign").sum()
        malicious = (split_df["label_binary"] == "malicious").sum()
        print(f"\n  {name} — benign: {benign:,} ({benign/len(split_df)*100:.1f}%) | malicious: {malicious:,} ({malicious/len(split_df)*100:.1f}%)")

    return train_df, test_df, val_df


def main():
    print("=" * 60)
    print("STEP 1: CREATE DATASET — Replicating Rivas et al. (2025)")
    print("=" * 60)
    print()

    # Load all 5 sources (matching Rivas's references)
    dfs = [
        load_source1_kaggle_malicious_urls(),   # [37] Primary — REQUIRED
        load_source2_kaggle_phishing_sites(),    # [38] Phishing URLs
        load_source3_kaggle_malicious_benign(),  # [39] Supplementary
        load_source4_phishstorm(),               # [34] PhishStorm
        load_source5_iscx_url_2016(),            # [35] ISCX-URL-2016
    ]

    # Check if we have at least the primary dataset
    if dfs[0] is None:
        print("\n[!] Pipeline cannot continue without Source 1 (Kaggle sid321axn).")
        print("    This is the primary dataset. Please download it and re-run.")
        return

    # Merge and clean
    merged = merge_and_clean(dfs)

    if merged is None:
        return

    # Split 70/20/10
    train_df, test_df, val_df = split_dataset(merged)

    # Save all files
    merged.to_csv(OUTPUT_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)

    print(f"\n{'='*60}")
    print("FILES SAVED")
    print(f"{'='*60}")
    print(f"  Full dataset:   {OUTPUT_FILE}")
    print(f"  Train set:      {TRAIN_FILE}")
    print(f"  Test set:       {TEST_FILE}")
    print(f"  Validation set: {VAL_FILE}")
    print(f"\n  Total URLs: {len(merged):,}")
    print("\n    Next step: Run '02_extract_features_24.py'")


if __name__ == "__main__":
    main()
