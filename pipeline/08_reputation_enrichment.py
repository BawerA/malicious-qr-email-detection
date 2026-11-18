"""
Step 8: Reputation Enrichment (Stage 2)

Augments lexical ML predictions with external reputation scores
from VirusTotal and Google Safe Browsing APIs.

This tests Hypothesis H2: Adding multi-source reputation scores
will increase precision by at least 3 percentage points over
the lexical model alone.

Input:  pipeline/results/email_pipeline_results.csv
        API keys (set as environment variables or entered at prompt)
Output: pipeline/results/enriched_results.csv
        pipeline/results/enrichment_comparison.csv

API Keys required:
  $env:VIRUSTOTAL_API_KEY = "your_key"
  $env:GOOGLE_SAFE_BROWSING_API_KEY = "your_key"
"""

import os
import sys
import time
import json
import base64
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PIPELINE_DIR = Path(__file__).parent
RESULTS_DIR = PIPELINE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "reputation_cache"
CONFIG_DIR = PIPELINE_DIR / "config"
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"

INPUT_FILE = RESULTS_DIR / "email_pipeline_results.csv"
ENRICHED_FILE = RESULTS_DIR / "enriched_results.csv"
COMPARISON_FILE = RESULTS_DIR / "enrichment_comparison.csv"

# ── API Configuration (Free Tier Limits) ──
VT_REQUESTS_PER_MINUTE = 4
VT_DAILY_LIMIT = 500       # VirusTotal free tier: 500 requests/day
VT_DELAY = 15.5             # seconds between VT requests (safe margin)
GSB_DAILY_LIMIT = 10000     # Google Safe Browsing free tier: 10,000 requests/day

# ── Hybrid Scoring Constants — Probability Combination Formula ──
# (Adopted 2026-04-29)
#
# Reputation signals are inherently one-sided: VT and GSB only emit
# malicious evidence (or silence). The Probability Combination Formula combines the
# ML probability with calibrated reputation contributions using the
# probabilistic OR operator:
#
#     P(malicious) = 1 - (1 - ML) * (1 - P_VT) * (1 - P_GSB)
#
# This is the standard probabilistic combination for independent
# positive-evidence sources. It:
#   1. Never deescalates — adding positive evidence always raises score
#   2. Rewards corroboration — when ML and reputation agree, score
#      boosts above either source alone (a plain maximum does not do this)
#   3. Treats silence as no information — multiplier of 1, neutral
#   4. Uses calibrated VT/GSB contributions — strong consensus contributes
#      more, weak consensus contributes less

# VT tier thresholds (vt_score = fraction of antivirus engines flagging)
VT_TIER_HIGH = 0.30      # >= 30% engines (e.g. >=21/70)  → very strong consensus
VT_TIER_MID  = 0.10      # >= 10% engines (e.g. >= 7/70)  → strong consensus
VT_TIER_LOW  = 0.05      # >=  5% engines (e.g. >= 4/70)  → weak positive signal

# Calibrated reputation contributions used in the Probability Combination Formula
P_GSB_HIT    = 0.95  # GSB blocklist hit  → near-certain malicious
P_VT_HIGH    = 0.90  # VT >= 0.30         → very strong malicious
P_VT_MID     = 0.75  # VT >= 0.10         → likely malicious
P_VT_LOW     = 0.45  # VT >= 0.05         → weak signal (suspicious tier)

# Hybrid classification thresholds
THRESHOLD_MALICIOUS = 0.50
THRESHOLD_SUSPICIOUS = 0.30


# ═══════════════════════════════════════════════════════════
# API KEY LOADING & USAGE TRACKING
# ═══════════════════════════════════════════════════════════

def load_api_keys():
    """Load API keys from config/api_keys.json, fall back to env vars."""
    vt_key = ""
    gsb_key = ""

    # Try config file first
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, "r") as f:
                keys = json.load(f)
            vt_key = keys.get("VIRUSTOTAL_API_KEY", "")
            gsb_key = keys.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
        except Exception as e:
            print(f"  [!] Error reading {API_KEYS_FILE}: {e}")

    # Fall back to environment variables
    if not vt_key or vt_key == "PASTE_YOUR_KEY_HERE":
        vt_key = os.environ.get("VIRUSTOTAL_API_KEY", "")
    if not gsb_key or gsb_key == "PASTE_YOUR_KEY_HERE":
        gsb_key = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")

    return vt_key, gsb_key


def load_usage_tracker():
    """Load daily API usage tracker."""
    tracker_file = CACHE_DIR / "daily_usage.json"
    today = datetime.now().strftime("%Y-%m-%d")

    if tracker_file.exists():
        with open(tracker_file, "r") as f:
            tracker = json.load(f)
        # Reset if new day
        if tracker.get("date") != today:
            tracker = {"date": today, "vt_calls": 0, "gsb_calls": 0}
    else:
        tracker = {"date": today, "vt_calls": 0, "gsb_calls": 0}

    return tracker


def save_usage_tracker(tracker):
    """Save daily API usage tracker."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tracker_file = CACHE_DIR / "daily_usage.json"
    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)


# ═══════════════════════════════════════════════════════════
# GOOGLE SAFE BROWSING (batch API — fast)
# ═══════════════════════════════════════════════════════════

def check_google_safe_browsing_batch(urls, api_key):
    """Query Google Safe Browsing for up to 500 URLs at once.

    Returns dict: {url: 1.0 if flagged, 0.0 if safe}
    """
    results = {}
    if not api_key:
        return results

    # GSB allows 500 URLs per request
    batch_size = 500
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]

        payload = {
            "client": {
                "clientId": "qr-code-detector",
                "clientVersion": "1.0",
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": u} for u in batch],
            },
        }

        try:
            endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
            response = requests.post(endpoint, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                flagged_urls = set()
                for match in data.get("matches", []):
                    threat_url = match.get("threat", {}).get("url", "")
                    if threat_url:
                        flagged_urls.add(threat_url)

                for url in batch:
                    results[url] = 1.0 if url in flagged_urls else 0.0
            else:
                print(f"    [!] GSB batch error: HTTP {response.status_code}")
                for url in batch:
                    results[url] = None

        except Exception as e:
            print(f"    [!] GSB batch error: {e}")
            for url in batch:
                results[url] = None

    return results


# ═══════════════════════════════════════════════════════════
# VIRUSTOTAL (cached, rate-limited)
# ═══════════════════════════════════════════════════════════

def load_vt_cache():
    """Load cached VirusTotal results."""
    cache_file = CACHE_DIR / "vt_cache.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_vt_cache(cache):
    """Save VirusTotal cache (called after each query for resume support)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "vt_cache.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def check_virustotal_single(url, api_key):
    """Query VirusTotal for a single URL using the URL report endpoint.

    Uses GET (existing report lookup) — 1 API call instead of POST+GET.
    Returns a score between 0.0 (clean) and 1.0 (malicious), or None on error.
    """
    try:
        # Base64-encode the URL for the API endpoint
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")

        headers = {"x-apikey": api_key}
        response = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            attrs = response.json().get("data", {}).get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            harmless = stats.get("harmless", 0)
            undetected = stats.get("undetected", 0)
            total = malicious + suspicious + harmless + undetected

            if total > 0:
                score = (malicious + suspicious * 0.5) / total
                return round(min(1.0, score), 4)

        elif response.status_code == 404:
            # URL not in VT database — treat as unknown
            return None

        return None

    except Exception as e:
        return None


def check_virustotal_all(urls, api_key, tracker):
    """Query VirusTotal for all URLs with rate limiting, caching, and daily limit.

    Resumes from cache if interrupted. Saves after each query.
    Stops when daily limit (500 requests) is reached.
    """
    cache = load_vt_cache()
    results = {}

    # Separate cached vs uncached
    uncached = [u for u in urls if u not in cache]
    for u in urls:
        if u in cache:
            results[u] = cache[u]

    if not uncached:
        print(f"    All {len(urls)} URLs found in cache")
        return results, tracker

    cached_count = len(urls) - len(uncached)
    if cached_count > 0:
        print(f"    {cached_count} URLs from cache, {len(uncached)} to query")

    # Check daily limit
    remaining_budget = VT_DAILY_LIMIT - tracker["vt_calls"]
    if remaining_budget <= 0:
        print(f"    [!] Daily VT limit reached ({VT_DAILY_LIMIT}). Skipping VT queries.")
        print(f"        Run again tomorrow to continue.")
        return results, tracker

    # Only query what we can afford today
    to_query = uncached[:remaining_budget]
    if len(to_query) < len(uncached):
        print(f"    [!] Daily budget: {remaining_budget} remaining of {VT_DAILY_LIMIT}")
        print(f"        Will query {len(to_query)}/{len(uncached)} uncached URLs today")
        print(f"        Run again tomorrow for the remaining {len(uncached) - len(to_query)}")

    est_minutes = len(to_query) * VT_DELAY / 60
    print(f"    Estimated time: {est_minutes:.0f} minutes ({len(to_query)} URLs × {VT_DELAY}s)")
    print(f"    (Progress saved after each query — safe to Ctrl+C and resume)\n")

    start_time = time.time()
    for i, url in enumerate(to_query):
        # Progress
        elapsed = time.time() - start_time
        if i > 0:
            per_url = elapsed / i
            remaining = per_url * (len(to_query) - i)
            remaining_min = remaining / 60
            done_total = i + cached_count
            progress_pct = done_total / len(urls) * 100
            print(
                f"    [{progress_pct:5.1f}%] {done_total}/{len(urls)} "
                f"| ETA: {remaining_min:.0f}m | {url[:60]}",
                end="\r",
            )
        else:
            print(f"    [  0.0%] Querying first URL...", end="\r")

        score = check_virustotal_single(url, api_key)

        cache[url] = score
        results[url] = score
        save_vt_cache(cache)

        tracker["vt_calls"] += 1
        save_usage_tracker(tracker)

        # Rate limiting (skip delay on last URL)
        if i < len(to_query) - 1:
            time.sleep(VT_DELAY)

    queried = len(to_query)
    skipped = len(uncached) - len(to_query)
    print(f"\n    VirusTotal: {queried} queried, {cached_count} cached, {skipped} skipped (daily limit)")
    print(f"    Daily usage: {tracker['vt_calls']}/{VT_DAILY_LIMIT}")
    return results, tracker


# ═══════════════════════════════════════════════════════════
# HYBRID SCORING — Probability Combination Formula (adopted 2026-04-29)
# ═══════════════════════════════════════════════════════════

def compute_hybrid_score(ml_prob, vt_score, gsb_score):
    """Compute hybrid risk score using the Probability Combination Formula.

    The score is computed as the probabilistic OR of three independent
    positive-evidence sources (ML, VT, GSB), where VT and GSB
    contributions are calibrated to consensus tiers rather than used
    raw:

        P(malicious) = 1 - (1 - ML) * (1 - P_VT) * (1 - P_GSB)

    Properties:
      1. Combines genuinely (rewards corroboration when sources agree)
      2. Never deescalates — adding positive evidence always raises score
      3. Treats silence as no information — multiplier of 1 (neutral)
      4. Uses calibrated VT/GSB contributions matched to consensus tiers

    Calibrated reputation contributions:
      GSB hit (gsb_score >= 1.0)  → P_GSB = 0.95   near-certain malicious
      VT score >= 0.30            → P_VT  = 0.90   very strong consensus
      VT score >= 0.10            → P_VT  = 0.75   likely malicious
      VT score >= 0.05            → P_VT  = 0.45   weak signal (suspicious tier)
      Otherwise (silent)          → P     = 0      no contribution

    Examples:
      ML=0.95, VT=0.0,  GSB=0  →  1 - 0.05·1·1     = 0.9500  (ML alone)
      ML=0.95, VT=0.30, GSB=0  →  1 - 0.05·0.10·1  = 0.9950  (corroboration boost)
      ML=0.40, VT=0.30, GSB=0  →  1 - 0.60·0.10·1  = 0.9400  (VT escalates → malicious)
      ML=0.40, VT=0.0,  GSB=1  →  1 - 0.60·1·0.05  = 0.9700  (GSB rescues → malicious)
      ML=0.20, VT=0.06, GSB=0  →  1 - 0.80·0.55·1  = 0.5600  (weak VT + low ML → malicious)
    """
    ml = ml_prob if (ml_prob is not None and pd.notna(ml_prob)) else 0.5

    # Calibrated GSB contribution
    p_gsb = 0.0
    if gsb_score is not None and pd.notna(gsb_score) and gsb_score >= 1.0:
        p_gsb = P_GSB_HIT

    # Calibrated VT contribution (highest applicable tier wins)
    p_vt = 0.0
    if vt_score is not None and pd.notna(vt_score):
        if vt_score >= VT_TIER_HIGH:
            p_vt = P_VT_HIGH
        elif vt_score >= VT_TIER_MID:
            p_vt = P_VT_MID
        elif vt_score >= VT_TIER_LOW:
            p_vt = P_VT_LOW

    # Probabilistic OR: P(at least one source flags) = 1 - product of (1 - P_i)
    score = 1.0 - (1.0 - ml) * (1.0 - p_vt) * (1.0 - p_gsb)
    return round(min(1.0, max(0.0, score)), 4)


def classify_hybrid(hybrid_score):
    """Classify based on hybrid score thresholds."""
    if hybrid_score >= THRESHOLD_MALICIOUS:
        return "malicious"
    elif hybrid_score >= THRESHOLD_SUSPICIOUS:
        return "suspicious"
    else:
        return "benign"


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(df, label_col):
    """Compute classification metrics for a given label column."""
    # Only count benign/malicious (exclude suspicious, error, no_qr, etc.)
    classified = df[df[label_col].isin(["benign", "malicious"])].copy()
    if len(classified) == 0:
        return None

    tp = ((classified["true_label"] == "malicious") & (classified[label_col] == "malicious")).sum()
    fp = ((classified["true_label"] == "benign") & (classified[label_col] == "malicious")).sum()
    tn = ((classified["true_label"] == "benign") & (classified[label_col] == "benign")).sum()
    fn = ((classified["true_label"] == "malicious") & (classified[label_col] == "benign")).sum()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "classified": len(classified),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STEP 8: REPUTATION ENRICHMENT (Stage 2)")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Check input
    if not INPUT_FILE.exists():
        print(f"[!] Input not found: {INPUT_FILE}")
        print("    Run '07_email_qr_pipeline.py' first.")
        return

    # ── API Keys ──
    vt_key, gsb_key = load_api_keys()

    print(f"  API keys loaded from: {API_KEYS_FILE if API_KEYS_FILE.exists() else 'environment variables'}")
    print(f"  VirusTotal API key:           {'SET' if vt_key else 'NOT SET'}")
    print(f"  Google Safe Browsing API key: {'SET' if gsb_key else 'NOT SET'}")

    if not vt_key and not gsb_key:
        print(f"\n[!] No API keys found.")
        print(f"    Edit: {API_KEYS_FILE}")
        print(f"    Replace PASTE_YOUR_KEY_HERE with your actual API keys.")
        return

    # ── Daily Usage Tracker ──
    tracker = load_usage_tracker()
    print(f"\n  Daily usage ({tracker['date']}):")
    print(f"    VirusTotal:          {tracker['vt_calls']}/{VT_DAILY_LIMIT}")
    print(f"    Google Safe Browsing: {tracker['gsb_calls']}/{GSB_DAILY_LIMIT}")

    print(f"\n  Formula: Probability Combination Formula")
    print(f"    Score = 1 - (1 - ML) * (1 - P_VT) * (1 - P_GSB)")
    print(f"    P_GSB = {P_GSB_HIT:.2f} if GSB hit, else 0")
    print(f"    P_VT  = {P_VT_HIGH:.2f} if VT >= {VT_TIER_HIGH:.2f}")
    print(f"    P_VT  = {P_VT_MID:.2f} if VT >= {VT_TIER_MID:.2f}")
    print(f"    P_VT  = {P_VT_LOW:.2f} if VT >= {VT_TIER_LOW:.2f}")
    print(f"    P_VT  = 0    otherwise")
    print(f"  Threshold: malicious >= {THRESHOLD_MALICIOUS}, suspicious >= {THRESHOLD_SUSPICIOUS}")

    # ── Load Stage 1 results ──
    df = pd.read_csv(INPUT_FILE)
    print(f"\n[+] Loaded {len(df)} email results from Stage 1")

    # Get unique URLs that need reputation lookup
    has_url = df[df["decoded_url"].notna() & (df["decoded_url"] != "")].copy()
    unique_urls = has_url["decoded_url"].unique().tolist()
    print(f"    Emails with decoded URLs: {len(has_url)}")
    print(f"    Unique URLs to query:     {len(unique_urls)}")

    # ── Google Safe Browsing (batch — fast) ──
    gsb_results = {}
    if gsb_key:
        # GSB batches 500 per request, so 982 URLs = 2 API calls
        num_gsb_calls = (len(unique_urls) + 499) // 500
        if tracker["gsb_calls"] + num_gsb_calls > GSB_DAILY_LIMIT:
            print(f"\n[!] GSB daily limit would be exceeded. Skipping.")
        else:
            print(f"\n[+] Querying Google Safe Browsing ({len(unique_urls)} URLs, {num_gsb_calls} batch requests)...")
            start = time.time()
            gsb_results = check_google_safe_browsing_batch(unique_urls, gsb_key)
            elapsed = time.time() - start
            tracker["gsb_calls"] += num_gsb_calls
            save_usage_tracker(tracker)
            flagged = sum(1 for v in gsb_results.values() if v == 1.0)
            safe = sum(1 for v in gsb_results.values() if v == 0.0)
            errors = sum(1 for v in gsb_results.values() if v is None)
            print(f"    Done in {elapsed:.1f}s — Flagged: {flagged}, Safe: {safe}, Errors: {errors}")

    # ── VirusTotal (rate-limited, cached, daily limit enforced) ──
    vt_results = {}
    if vt_key:
        print(f"\n[+] Querying VirusTotal ({len(unique_urls)} URLs, rate-limited)...")
        vt_results, tracker = check_virustotal_all(unique_urls, vt_key, tracker)
        vt_found = sum(1 for v in vt_results.values() if v is not None)
        vt_flagged = sum(1 for v in vt_results.values() if v is not None and v > 0.1)
        print(f"    Results: {vt_found} found, {vt_flagged} flagged (score > 0.1)")

    # ── Compute Hybrid Scores ──
    print(f"\n[+] Computing hybrid scores...")

    vt_scores = []
    gsb_scores_col = []
    hybrid_scores = []
    hybrid_labels = []

    for _, row in df.iterrows():
        url = row.get("decoded_url", "")

        if pd.isna(url) or url == "":
            vt_scores.append(None)
            gsb_scores_col.append(None)
            hybrid_scores.append(None)
            hybrid_labels.append(row.get("predicted_label", "error"))
            continue

        vt_score = vt_results.get(url)
        gsb_score = gsb_results.get(url)
        ml_prob = row.get("probability")
        if pd.isna(ml_prob):
            ml_prob = None

        vt_scores.append(vt_score)
        gsb_scores_col.append(gsb_score)

        hybrid = compute_hybrid_score(ml_prob, vt_score, gsb_score)
        hybrid_scores.append(hybrid)
        hybrid_labels.append(classify_hybrid(hybrid))

    df["vt_score"] = vt_scores
    df["gsb_score"] = gsb_scores_col
    df["hybrid_score"] = hybrid_scores
    df["hybrid_label"] = hybrid_labels

    # Save enriched results
    df.to_csv(ENRICHED_FILE, index=False)

    # ═══════════════════════════════════════════════════════
    # COMPARISON: Stage 1 vs Stage 2
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("COMPARISON: Stage 1 (Lexical Only) vs Stage 2 (Hybrid)")
    print(f"{'=' * 60}")

    stage1_metrics = compute_metrics(df, "predicted_label")
    stage2_metrics = compute_metrics(df, "hybrid_label")

    comparison = []

    for name, metrics in [("Stage 1 (Lexical)", stage1_metrics), ("Stage 2 (Hybrid)", stage2_metrics)]:
        if metrics is None:
            continue

        comparison.append({"stage": name, **metrics})

        print(f"\n  {name} (n={metrics['classified']}):")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
        print(f"    FPR:       {metrics['fpr']:.4f}")
        print(f"    TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    if len(comparison) == 2:
        s1 = comparison[0]
        s2 = comparison[1]

        print(f"\n  {'─' * 50}")
        print(f"  IMPROVEMENT (Stage 2 − Stage 1):")
        for metric in ["accuracy", "precision", "recall", "f1", "fpr"]:
            diff = s2[metric] - s1[metric]
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "─"
            # FPR improvement is negative (lower is better)
            if metric == "fpr":
                direction = "↓" if diff < 0 else "↑" if diff > 0 else "─"
            print(f"    {metric.upper():>10}: {diff:+.4f} ({diff * 100:+.2f}pp) {direction}")

        # H2 check: precision improvement ≥ 3 percentage points
        precision_pp = (s2["precision"] - s1["precision"]) * 100
        print(f"\n  {'═' * 50}")
        print(f"  H2 CHECK: Precision improvement = {precision_pp:+.2f} percentage points")
        if precision_pp >= 3:
            print(f"  >>> H2 SUPPORTED: >= 3pp improvement achieved <<<")
        else:
            print(f"  >>> H2 NOT SUPPORTED: < 3pp improvement <<<")

        # H3 check: FPR ≤ 5%
        fpr_pct = s2["fpr"] * 100
        print(f"\n  H3 CHECK: Stage 2 FPR = {fpr_pct:.2f}%")
        if fpr_pct <= 5:
            print(f"  >>> H3 SUPPORTED: FPR <= 5% <<<")
        else:
            print(f"  >>> H3 NOT SUPPORTED: FPR > 5% <<<")

    # Save comparison
    if comparison:
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(COMPARISON_FILE, index=False)

    # ── Summary ──
    # Count suspicious labels (only exists in Stage 2)
    suspicious_count = (df["hybrid_label"] == "suspicious").sum()

    print(f"\n{'=' * 60}")
    print("FILES SAVED")
    print(f"{'=' * 60}")
    print(f"  Enriched results: {ENRICHED_FILE}")
    print(f"  Comparison:       {COMPARISON_FILE}")
    if CACHE_DIR.exists():
        print(f"  VT cache:         {CACHE_DIR / 'vt_cache.json'}")
    if suspicious_count > 0:
        print(f"\n  Note: {suspicious_count} emails classified as 'suspicious' (between thresholds)")
        print(f"  These are excluded from binary metrics above")

    print(f"\n    Next step: Run '09_final_evaluation.py'")


if __name__ == "__main__":
    main()
