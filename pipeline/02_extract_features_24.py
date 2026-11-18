"""
Step 2: Extract Rivas's 24 Features (Baseline Comparison)

Extracts the exact 24 lexical URL features that Rivas et al. (2025) used,
so we can do a 1-to-1 comparison with their results.

Rivas's 24 features (from Table I):
1. having_ip_address     13. count_question_mark
2. abnormal_url          14. count_hyphen
3. google_index          15. count_equal
4. count_dot             16. url_length
5. count_www             17. hostname_length
6. count_atrate          18. suspicious_words
7. count_dir             19. digit_count
8. count_embed_domain    20. letter_count
9. shortening_service    21. fd_length
10. count_https          22. tld_length
11. count_http           23. entropy
12. count_percent        24. has_port

Extracts features from train, test, and validation sets separately.

Input:  pipeline/data/train_set.csv
        pipeline/data/test_set.csv
        pipeline/data/validation_set.csv
Output: pipeline/data/rivas_24_train.csv
        pipeline/data/rivas_24_test.csv
        pipeline/data/rivas_24_val.csv
"""

import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"

SPLITS = {
    "train": (DATA_DIR / "train_set.csv", DATA_DIR / "rivas_24_train.csv"),
    "test":  (DATA_DIR / "test_set.csv",  DATA_DIR / "rivas_24_test.csv"),
    "val":   (DATA_DIR / "validation_set.csv", DATA_DIR / "rivas_24_val.csv"),
}

# Known URL shortening services
SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
    "v.gd", "clck.ru", "rebrand.ly", "bl.ink", "short.io",
]

# Suspicious words commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure",
    "banking", "confirm", "password", "credential", "suspend",
    "unusual", "alert", "notification", "paypal", "apple", "microsoft",
    "google", "amazon", "netflix", "facebook", "instagram",
]


def calc_entropy(text):
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_rivas_24_features(url):
    """Extract the exact 24 features from Rivas et al. (2025) Table I."""

    features = {}

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        scheme = parsed.scheme or ""

        # 1. having_ip_address: Checks if URL contains an IP address
        ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        features["having_ip_address"] = 1 if re.search(ip_pattern, url) else 0

        # 2. abnormal_url: Determines if URL lacks a hostname
        features["abnormal_url"] = 1 if not hostname else 0

        # 3. google_index: Checks if URL is indexed by Google
        # NOTE: Cannot check in real-time without API. Set to 0 (unknown).
        # Rivas likely used a Google search API or cached results.
        features["google_index"] = 0

        # 4. count_dot: Occurrences of '.' in URL
        features["count_dot"] = url.count(".")

        # 5. count_www: Occurrences of 'www' in URL
        features["count_www"] = url.lower().count("www")

        # 6. count_atrate: Number of '@' characters
        features["count_atrate"] = url.count("@")

        # 7. count_dir: Number of directory paths in URL
        features["count_dir"] = path.count("/") - 1 if path else 0
        features["count_dir"] = max(0, features["count_dir"])

        # 8. count_embed_domain: Occurrences of '//' in the path
        features["count_embed_domain"] = path.count("//")

        # 9. shortening_service: Use of known URL shortening service
        features["shortening_service"] = 1 if any(
            s in url.lower() for s in SHORTENING_SERVICES
        ) else 0

        # 10. count_https: Number of times 'https' appears
        features["count_https"] = url.lower().count("https")

        # 11. count_http: Number of times 'http' appears
        features["count_http"] = url.lower().count("http") - url.lower().count("https")

        # 12. count_percent: Number of '%' characters
        features["count_percent"] = url.count("%")

        # 13. count_question_mark: Number of '?' characters
        features["count_question_mark"] = url.count("?")

        # 14. count_hyphen: Number of '-' characters
        features["count_hyphen"] = url.count("-")

        # 15. count_equal: Number of '=' characters
        features["count_equal"] = url.count("=")

        # 16. url_length: Total length of URL
        features["url_length"] = len(url)

        # 17. hostname_length: Length of hostname
        features["hostname_length"] = len(hostname)

        # 18. suspicious_words: Presence of suspicious words
        features["suspicious_words"] = 1 if any(
            w in url.lower() for w in SUSPICIOUS_WORDS
        ) else 0

        # 19. digit_count: Number of digits
        features["digit_count"] = sum(c.isdigit() for c in url)

        # 20. letter_count: Number of alphabetic characters
        features["letter_count"] = sum(c.isalpha() for c in url)

        # 21. fd_length: Length of first directory in URL path
        dirs = [d for d in path.split("/") if d]
        features["fd_length"] = len(dirs[0]) if dirs else 0

        # 22. tld_length: Length of top-level domain
        tld = hostname.split(".")[-1] if hostname and "." in hostname else ""
        features["tld_length"] = len(tld)

        # 23. entropy: Shannon entropy of URL
        features["entropy"] = calc_entropy(url)

        # 24. has_port: Checks if URL has a port number
        features["has_port"] = 1 if parsed.port else 0

    except Exception:
        # Return zeros for all features on error
        feature_names = [
            "having_ip_address", "abnormal_url", "google_index",
            "count_dot", "count_www", "count_atrate", "count_dir",
            "count_embed_domain", "shortening_service", "count_https",
            "count_http", "count_percent", "count_question_mark",
            "count_hyphen", "count_equal", "url_length", "hostname_length",
            "suspicious_words", "digit_count", "letter_count",
            "fd_length", "tld_length", "entropy", "has_port",
        ]
        features = {name: 0 for name in feature_names}

    return features


def process_split(input_file, output_file, split_name):
    """Extract 24 features for one data split."""

    if not input_file.exists():
        print(f"[!] {split_name} file not found: {input_file}")
        print("    Run '01_create_dataset.py' first.")
        return False

    print(f"\n[+] Processing {split_name} split: {input_file}")
    df = pd.read_csv(input_file)
    print(f"    {len(df):,} URLs loaded")

    print(f"    Extracting 24 Rivas features...")
    features_list = []
    for url in tqdm(df["url"], desc=f"{split_name}"):
        features_list.append(extract_rivas_24_features(str(url)))

    features_df = pd.DataFrame(features_list)

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
    print("STEP 2: EXTRACT RIVAS'S 24 FEATURES (Baseline)")
    print("=" * 60)

    success = True
    for split_name, (input_file, output_file) in SPLITS.items():
        if not process_split(input_file, output_file, split_name):
            success = False

    if success:
        print(f"\n[+] All splits processed.")
        print(f"    Features: {list(pd.read_csv(list(SPLITS.values())[0][1]).columns[4:])}")
        print("\n    Next step: Run '03_extract_features_63.py'")


if __name__ == "__main__":
    main()
