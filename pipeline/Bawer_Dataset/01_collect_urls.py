"""
Bawer_Dataset — Step 1: Augment Rivas Data with Bare Domains

The Rivas-trained model achieved 95.6% F1 on in-distribution URLs but failed
on bare-domain URLs (46.5% accuracy, 99.6% FPR). Root cause: the Rivas dataset
contains predominantly full-path URLs, so the model never learned that bare
domains (e.g., https://google.com) can be benign.

This script augments the Rivas dataset with ONE change: adding bare-domain URLs
to both classes. This isolates the variable (URL format diversity) so we can
clearly attribute any improvement to the dataset composition.

Augmentation:
  Benign:    Keep ~195K Rivas full-path URLs + add ~130K Tranco bare domains
  Malicious: Keep all ~216K Rivas URLs + add fresh URLhaus URLs to reach 325K

Target: 325,000 benign + 325,000 malicious = 650,000 total (same as Rivas)

Output: pipeline/Bawer_Dataset/data/bawer_urls_raw.csv
"""

import csv
import io
import json
import time
import zipfile
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Rivas dataset location
PIPELINE_DIR = SCRIPT_DIR.parent
RIVAS_CSV = PIPELINE_DIR / "data" / "rivas_dataset.csv"

# Targets (same total as Rivas for controlled comparison)
TARGET_BENIGN = 325_000
TARGET_MALICIOUS = 325_000
BARE_DOMAIN_RATIO = 0.40  # 40% bare domains in benign class

# Output / cache files
RAW_OUTPUT = DATA_DIR / "bawer_urls_raw.csv"
TRANCO_CACHE = DATA_DIR / "tranco_list.csv"
URLHAUS_CACHE = DATA_DIR / "urlhaus_urls_cache.csv"


# ── Helper ─────────────────────────────────────────────────────────────

def classify_url_type(url):
    """Classify a URL as bare_domain or full_path."""
    try:
        parsed = urlparse(str(url))
        if not parsed.path or parsed.path == "/" :
            return "bare_domain"
        return "full_path"
    except Exception:
        return "full_path"


# ── Load Rivas Dataset ─────────────────────────────────────────────────

def load_rivas_dataset():
    """Load the original Rivas dataset (650K URLs)."""
    print(f"\n[+] Loading Rivas dataset: {RIVAS_CSV}")

    if not RIVAS_CSV.exists():
        print(f"    [!] File not found: {RIVAS_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(RIVAS_CSV)
    print(f"    Loaded: {len(df):,} URLs")

    # Normalize labels to binary
    df["label_bin"] = df["label"].apply(
        lambda x: "benign" if str(x).lower() in ("benign", "safe", "legitimate") else "malicious"
    )

    # Classify URL type
    df["url_type"] = df["url"].apply(classify_url_type)

    # Stats
    benign = (df["label_bin"] == "benign").sum()
    malicious = (df["label_bin"] == "malicious").sum()
    bare = (df["url_type"] == "bare_domain").sum()
    full = (df["url_type"] == "full_path").sum()

    print(f"    Labels:  benign={benign:,}  malicious={malicious:,}")
    print(f"    Types:   bare_domain={bare:,}  full_path={full:,}")

    return df


# ── Tranco Bare Domains ────────────────────────────────────────────────

def download_tranco_bare_domains(n_domains):
    """Download Tranco top sites as bare-domain URLs."""
    print(f"\n[+] Downloading Tranco top {n_domains:,} domains...")

    if TRANCO_CACHE.exists():
        df = pd.read_csv(TRANCO_CACHE, header=None, names=["rank", "domain"])
        domains = df["domain"].tolist()[:n_domains]
        print(f"    Loaded from cache: {len(domains):,} domains")
    else:
        url = "https://tranco-list.eu/top-1m.csv.zip"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()

            # Extract CSV from zip
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = z.namelist()[0]
            csv_text = z.read(csv_name).decode("utf-8")

            lines = csv_text.strip().split("\n")
            domains = []
            with open(TRANCO_CACHE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        writer.writerow(parts)
                        domains.append(parts[1])

            domains = domains[:n_domains]
            print(f"    Downloaded: {len(domains):,} domains")

        except Exception as e:
            print(f"    [!] Error: {e}")
            return pd.DataFrame()

    # Convert to bare-domain URLs
    bare_urls = []
    for domain in domains:
        bare_urls.append({
            "url": f"https://{domain}",
            "label_bin": "benign",
            "url_type": "bare_domain",
            "source": "tranco",
        })

    df = pd.DataFrame(bare_urls)
    print(f"    Created: {len(df):,} bare-domain benign URLs")
    return df


# ── URLhaus Fresh Malicious URLs ───────────────────────────────────────

def download_urlhaus_urls():
    """Download fresh malicious URLs from URLhaus (full database)."""
    print(f"\n[+] Downloading URLhaus malicious URLs...")

    if URLHAUS_CACHE.exists():
        df = pd.read_csv(URLHAUS_CACHE)
        print(f"    Loaded from cache: {len(df):,} URLs")
        return df

    # Use the full CSV dump (all URLs, not just currently online)
    download_url = "https://urlhaus.abuse.ch/downloads/csv/"

    try:
        print(f"    Downloading full database (may take a minute)...")
        resp = requests.get(download_url, timeout=300)
        resp.raise_for_status()

        # Full dump is a ZIP file containing csv.txt
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_filename = z.namelist()[0]  # typically "csv.txt"
        print(f"    Extracted: {csv_filename} from ZIP")
        content = z.read(csv_filename).decode("utf-8", errors="replace")

        # Parse line by line to handle malformed rows gracefully
        # Format: "id","dateadded","url","url_status","last_online","threat","tags","urlhaus_link","reporter"
        urls = set()
        skipped = 0

        for line in content.split("\n"):
            # Skip comments and empty lines
            if not line.strip() or line.startswith("#"):
                continue

            # Extract URL field (3rd column, index 2) using CSV reader
            try:
                row = next(csv.reader(StringIO(line)))
                if len(row) >= 3:
                    url_val = row[2].strip()
                    if url_val and url_val.startswith(("http://", "https://")):
                        urls.add(url_val)
            except Exception:
                skipped += 1
                continue

        print(f"    Parsed: {len(urls):,} unique URLs (skipped {skipped:,} bad lines)")

    except Exception as e:
        print(f"    [!] Error downloading full dump: {e}")
        print(f"    Trying online-only dump...")

        # Fallback to online-only CSV
        download_url = "https://urlhaus.abuse.ch/downloads/csv_online/"
        try:
            resp = requests.get(download_url, timeout=120)
            resp.raise_for_status()

            content = resp.text
            urls = set()
            skipped = 0
            for line in content.split("\n"):
                if not line.strip() or line.startswith("#"):
                    continue
                try:
                    row = next(csv.reader(StringIO(line)))
                    if len(row) >= 3:
                        url_val = row[2].strip()
                        if url_val and url_val.startswith(("http://", "https://")):
                            urls.add(url_val)
                except Exception:
                    skipped += 1

            print(f"    Downloaded (online only): {len(urls):,} URLs")

        except Exception as e2:
            print(f"    [!] Error: {e2}")
            return pd.DataFrame()

    # Build DataFrame
    records = []
    for u in urls:
        records.append({
            "url": u,
            "label_bin": "malicious",
            "url_type": classify_url_type(u),
            "source": "urlhaus",
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["url"])

    # Cache
    df.to_csv(URLHAUS_CACHE, index=False)
    print(f"    Total (deduplicated): {len(df):,} URLs")

    return df


# ── Combine & Balance ──────────────────────────────────────────────────

def augment_and_balance(rivas_df, tranco_df, urlhaus_df):
    """Augment Rivas dataset with bare domains and balance to 650K."""
    print(f"\n[+] Augmenting Rivas dataset...")

    # ── Benign class ──
    rivas_benign = rivas_df[rivas_df["label_bin"] == "benign"][["url", "label_bin", "url_type"]]
    rivas_benign["source"] = "rivas"

    # How many bare domains to add?
    bare_target = int(TARGET_BENIGN * BARE_DOMAIN_RATIO)  # ~130,000
    full_target = TARGET_BENIGN - bare_target               # ~195,000

    # Sample Rivas benign (full-path URLs) down to full_target
    rivas_benign_full = rivas_benign[rivas_benign["url_type"] == "full_path"]
    rivas_benign_bare = rivas_benign[rivas_benign["url_type"] == "bare_domain"]

    print(f"\n  Benign composition:")
    print(f"    Rivas full-path available: {len(rivas_benign_full):,}")
    print(f"    Rivas bare-domain available: {len(rivas_benign_bare):,}")
    print(f"    Tranco bare-domain available: {len(tranco_df):,}")

    # Take Rivas full-path benign URLs
    if len(rivas_benign_full) > full_target:
        benign_full = rivas_benign_full.sample(n=full_target, random_state=42)
    else:
        benign_full = rivas_benign_full
        print(f"    [!] Only {len(rivas_benign_full):,} Rivas full-path benign "
              f"(target: {full_target:,})")

    # Take Tranco bare domains (+ any Rivas bare domains)
    available_bare = pd.concat([tranco_df[["url", "label_bin", "url_type", "source"]],
                                rivas_benign_bare], ignore_index=True)
    available_bare = available_bare.drop_duplicates(subset=["url"])

    if len(available_bare) > bare_target:
        benign_bare = available_bare.sample(n=bare_target, random_state=42)
    else:
        benign_bare = available_bare
        print(f"    [!] Only {len(available_bare):,} bare-domain benign "
              f"(target: {bare_target:,})")

    benign_combined = pd.concat([benign_full, benign_bare], ignore_index=True)
    print(f"    Final benign: {len(benign_combined):,} "
          f"({len(benign_bare):,} bare + {len(benign_full):,} full)")

    # ── Malicious class ──
    rivas_malicious = rivas_df[rivas_df["label_bin"] == "malicious"][["url", "label_bin", "url_type"]]
    rivas_malicious["source"] = "rivas"

    print(f"\n  Malicious composition:")
    print(f"    Rivas malicious available: {len(rivas_malicious):,}")
    print(f"    URLhaus fresh available: {len(urlhaus_df):,}")

    # Keep all Rivas malicious, add URLhaus to reach target
    needed_from_urlhaus = max(0, TARGET_MALICIOUS - len(rivas_malicious))

    # Remove duplicates between Rivas and URLhaus
    rivas_mal_urls = set(rivas_malicious["url"].tolist())
    urlhaus_new = urlhaus_df[~urlhaus_df["url"].isin(rivas_mal_urls)]

    if len(urlhaus_new) > needed_from_urlhaus:
        urlhaus_sample = urlhaus_new.sample(n=needed_from_urlhaus, random_state=42)
    else:
        urlhaus_sample = urlhaus_new
        print(f"    [!] Only {len(urlhaus_new):,} new URLhaus URLs "
              f"(target: {needed_from_urlhaus:,})")

    malicious_combined = pd.concat([rivas_malicious, urlhaus_sample], ignore_index=True)
    print(f"    Final malicious: {len(malicious_combined):,} "
          f"({len(rivas_malicious):,} Rivas + {len(urlhaus_sample):,} URLhaus)")

    # ── Combine ──
    combined = pd.concat([benign_combined, malicious_combined], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"])
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined


def print_dataset_stats(df):
    """Print detailed dataset statistics."""
    print(f"\n{'='*60}")
    print("BAWER_DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"\n  Total URLs: {len(df):,}")

    print(f"\n  By label:")
    for label, count in df["label_bin"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:12s}: {count:>8,} ({pct:.1f}%)")

    print(f"\n  By URL type:")
    for url_type, count in df["url_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {url_type:12s}: {count:>8,} ({pct:.1f}%)")

    print(f"\n  By source:")
    for source, count in df["source"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {source:16s}: {count:>8,} ({pct:.1f}%)")

    print(f"\n  Cross-tabulation (label x url_type):")
    ct = pd.crosstab(df["label_bin"], df["url_type"])
    for label in ct.index:
        parts = "  ".join([f"{col}: {ct.loc[label, col]:,}" for col in ct.columns])
        print(f"    {label:12s}: {parts}")

    print(f"\n  Cross-tabulation (label x source):")
    ct2 = pd.crosstab(df["label_bin"], df["source"])
    for label in ct2.index:
        parts = "  ".join([f"{col}: {ct2.loc[label, col]:,}" for col in ct2.columns])
        print(f"    {label:12s}: {parts}")

    # URL length stats
    df_temp = df.copy()
    df_temp["url_len"] = df_temp["url"].str.len()
    print(f"\n  URL length stats:")
    for label in ["benign", "malicious"]:
        subset = df_temp[df_temp["label_bin"] == label]
        if len(subset) > 0:
            print(f"    {label:12s}: mean={subset['url_len'].mean():.0f}, "
                  f"median={subset['url_len'].median():.0f}, "
                  f"min={subset['url_len'].min()}, max={subset['url_len'].max()}")

    # Compare with Rivas
    bare_benign = ((df["label_bin"] == "benign") & (df["url_type"] == "bare_domain")).sum()
    total_benign = (df["label_bin"] == "benign").sum()
    bare_pct = bare_benign / total_benign * 100 if total_benign > 0 else 0
    print(f"\n  Key metric — bare-domain benign ratio: {bare_pct:.1f}%")
    print(f"    (Rivas had ~0% → caused 99.6% FPR on bare domains)")
    print(f"    (Bawer target: ~40%)")


def main():
    print("=" * 60)
    print("BAWER_DATASET — STEP 1: AUGMENT RIVAS WITH BARE DOMAINS")
    print("=" * 60)
    print(f"\n  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Strategy: Augment Rivas dataset with bare-domain URLs")
    print(f"  Variable changed: URL format diversity (bare domains added)")
    print(f"  Target: {TARGET_BENIGN:,} benign + {TARGET_MALICIOUS:,} malicious "
          f"= {TARGET_BENIGN + TARGET_MALICIOUS:,} total")
    print(f"  Bare domain ratio (benign): {BARE_DOMAIN_RATIO:.0%}")

    # Step 1: Load Rivas data
    rivas_df = load_rivas_dataset()
    if rivas_df.empty:
        return

    # Step 2: Download Tranco bare domains
    bare_needed = int(TARGET_BENIGN * BARE_DOMAIN_RATIO)
    tranco_df = download_tranco_bare_domains(n_domains=bare_needed + 10000)  # extra buffer
    if tranco_df.empty:
        return

    # Step 3: Download fresh URLhaus URLs
    urlhaus_df = download_urlhaus_urls()
    if urlhaus_df.empty:
        print("\n[!] Could not download URLhaus. Using Rivas malicious URLs only.")
        urlhaus_df = pd.DataFrame(columns=["url", "label_bin", "url_type", "source"])

    # Step 4: Augment and balance
    combined = augment_and_balance(rivas_df, tranco_df, urlhaus_df)

    # Step 5: Print stats
    print_dataset_stats(combined)

    # Step 6: Save
    combined.to_csv(RAW_OUTPUT, index=False)
    print(f"\n[+] Dataset saved to: {RAW_OUTPUT}")
    print(f"    Total URLs: {len(combined):,}")

    print(f"\n    Next step: Extract 63 features using the pipeline's feature extractor")
    print(f"    Command: python pipeline/Bawer_Dataset/02_split_and_extract_features.py")


if __name__ == "__main__":
    main()
