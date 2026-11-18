"""
Step 6: Generate Email QR Codes — Real-World Test Set

Creates realistic .eml email files with QR codes embedded as inline images.
This simulates the actual threat vector: malicious QR codes inside emails.

URLs are sourced from LIVE, external threat intelligence feeds (not the training data):
- Benign: Tranco Top Sites (research-grade top websites list)
- Malicious: PhishTank (verified phishing) + URLhaus (active malware URLs)

This ensures the model is tested on URLs it has NEVER seen during training.

Two types of emails generated:
- Benign: QR codes pointing to top legitimate websites
- Malicious: QR codes pointing to recently verified phishing/malware URLs

Output: pipeline/data/test_emails/benign/*.eml
        pipeline/data/test_emails/malicious/*.eml
        pipeline/data/test_emails/email_manifest.csv
        pipeline/data/test_emails/cached_urls/ (cached downloads for reproducibility)
"""

import os
import io
import csv
import random
import zipfile
import urllib.request
import ssl
import segno
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import generator
from pathlib import Path
from datetime import datetime
import pandas as pd

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
EMAIL_DIR = DATA_DIR / "test_emails"
BENIGN_DIR = EMAIL_DIR / "benign"
MALICIOUS_DIR = EMAIL_DIR / "malicious"
CACHE_DIR = EMAIL_DIR / "cached_urls"
MANIFEST_FILE = EMAIL_DIR / "email_manifest.csv"

# How many test emails to generate per class
NUM_BENIGN = 500
NUM_MALICIOUS = 500

SEED = 42

# Realistic URL paths to append to a small share of benign domains (see PATH_APPEND_RATIO)
# Simulates real QR codes in emails: menus, events, blogs, products
# Uses mild paths that legitimate sites actually use in QR code emails
BENIGN_PATH_TEMPLATES = [
    "/menu/seasonal-specials",
    "/events/spring-2026",
    "/newsletter/march-2026",
    "/blog/2026/03/new-features",
    "/products/catalog",
    "/about/locations",
    "/careers/openings",
    "/store/hours",
    "/rewards/program",
    "/contact-us",
    "/promotions/spring-sale",
    "/gallery/photos",
    "/schedule/classes",
    "/faq",
    "/menu",
]
PATH_APPEND_RATIO = 0.02  # 2% of benign URLs get short paths appended

# ── Download URLs ──
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
URLHAUS_URL = "https://urlhaus.abuse.ch/downloads/csv_online/"

# Email templates for malicious emails (mimicking real quishing attacks)
MALICIOUS_TEMPLATES = [
    {
        "subject": "Action Required: Verify Your Account",
        "from": "security@company-verify.com",
        "body": """
        <html><body>
        <p>Dear User,</p>
        <p>We have detected unusual activity on your account.
        Please scan the QR code below to verify your identity
        and secure your account.</p>
        <p><img src="cid:qrcode"></p>
        <p>If you did not request this, please ignore this email.</p>
        <p>Security Team</p>
        </body></html>
        """,
    },
    {
        "subject": "Your Package is Ready for Pickup",
        "from": "delivery@express-shipping.com",
        "body": """
        <html><body>
        <p>Hello,</p>
        <p>Your package has arrived and is ready for pickup.
        Please scan the QR code below to confirm delivery
        and view tracking details.</p>
        <p><img src="cid:qrcode"></p>
        <p>Thank you for choosing Express Shipping.</p>
        </body></html>
        """,
    },
    {
        "subject": "IT Department: Password Reset Required",
        "from": "it-helpdesk@company-support.com",
        "body": """
        <html><body>
        <p>Dear Employee,</p>
        <p>Your company password will expire in 24 hours.
        Please scan the QR code below to reset your password
        immediately to avoid account lockout.</p>
        <p><img src="cid:qrcode"></p>
        <p>IT Help Desk</p>
        </body></html>
        """,
    },
    {
        "subject": "Payment Confirmation - Invoice #INV-2025-3847",
        "from": "billing@payment-portal.com",
        "body": """
        <html><body>
        <p>Dear Customer,</p>
        <p>A payment of $149.99 has been charged to your account.
        If you did not authorize this transaction, scan the QR code
        below to dispute the charge.</p>
        <p><img src="cid:qrcode"></p>
        <p>Billing Department</p>
        </body></html>
        """,
    },
]

# Email templates for benign emails
BENIGN_TEMPLATES = [
    {
        "subject": "Check Out Our New Menu!",
        "from": "hello@local-restaurant.com",
        "body": """
        <html><body>
        <p>Hi there!</p>
        <p>We've updated our menu for the season.
        Scan the QR code below to view our latest offerings.</p>
        <p><img src="cid:qrcode"></p>
        <p>See you soon!</p>
        </body></html>
        """,
    },
    {
        "subject": "Your Event Ticket",
        "from": "tickets@events-platform.com",
        "body": """
        <html><body>
        <p>Hello,</p>
        <p>Here is your QR code ticket for the upcoming event.
        Please present this at the entrance.</p>
        <p><img src="cid:qrcode"></p>
        <p>Enjoy the event!</p>
        </body></html>
        """,
    },
    {
        "subject": "Wi-Fi Access for Conference Room B",
        "from": "facilities@company.com",
        "body": """
        <html><body>
        <p>Hi,</p>
        <p>Scan the QR code below to connect to the guest Wi-Fi
        network in Conference Room B.</p>
        <p><img src="cid:qrcode"></p>
        <p>Facilities Team</p>
        </body></html>
        """,
    },
]


def download_with_fallback(url, cache_path, description):
    """Download a URL, using cached version if download fails."""
    if cache_path.exists():
        print(f"    [cache] Using cached {description}: {cache_path.name}")
        return cache_path

    print(f"    [download] Fetching {description}...")
    try:
        # Create SSL context that doesn't verify (some academic sites have cert issues)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            data = response.read()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)

        size_mb = len(data) / (1024 * 1024)
        print(f"    [download] Saved {size_mb:.1f} MB to {cache_path.name}")
        return cache_path

    except Exception as e:
        print(f"    [!] Download failed: {e}")
        if cache_path.exists():
            print(f"    [cache] Falling back to cached version")
            return cache_path
        return None


def load_tranco_urls(n=500):
    """Load benign URLs from Tranco list (ranks 5,000-50,000).

    Uses mid-tier domains instead of top 500 (google.com, facebook.com)
    to create a more challenging and realistic test. These are legitimate
    but less well-known sites that better represent real-world QR code URLs.
    """
    print(f"\n[+] Loading benign URLs from Tranco (ranks 5,000-50,000)...")

    cache_path = CACHE_DIR / "tranco_top1m.csv.zip"
    downloaded = download_with_fallback(TRANCO_URL, cache_path, "Tranco Top 1M")

    if downloaded is None:
        print("    [!] Tranco download failed. No cached version available.")
        return []

    try:
        with zipfile.ZipFile(downloaded) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                lines = f.read().decode("utf-8").strip().split("\n")

        # Use ranks 5,000-50,000 for a more challenging test
        # These are legitimate but less obvious domains
        START_RANK = 5000
        END_RANK = 50000
        candidate_lines = lines[START_RANK:END_RANK]

        urls = []
        for line in candidate_lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                domain = parts[1].strip()
                urls.append(f"https://{domain}")

        # Randomly sample n URLs from the range
        random.shuffle(urls)
        selected = urls[:n]

        # Append realistic paths to a share of URLs (PATH_APPEND_RATIO) to simulate real-world QR codes
        # (real emails link to specific pages, not just bare domains)
        n_with_paths = int(len(selected) * PATH_APPEND_RATIO)
        for i in range(n_with_paths):
            path = random.choice(BENIGN_PATH_TEMPLATES)
            # Replace {rand} placeholders with random strings
            path = path.replace("{rand}", f"{random.randint(10000, 99999)}")
            selected[i] = selected[i] + path

        n_bare = len(selected) - n_with_paths
        print(f"    Loaded {len(selected)} benign URLs from Tranco "
              f"(ranks {START_RANK:,}-{END_RANK:,})")
        print(f"    {n_with_paths} with paths ({PATH_APPEND_RATIO:.0%}), "
              f"{n_bare} bare domains")
        return selected

    except Exception as e:
        print(f"    [!] Error parsing Tranco: {e}")
        return []


def load_phishtank_urls(n=300):
    """Load verified phishing URLs from PhishTank."""
    print(f"\n[+] Loading phishing URLs from PhishTank...")

    cache_path = CACHE_DIR / "phishtank_online_valid.csv"
    downloaded = download_with_fallback(PHISHTANK_URL, cache_path, "PhishTank verified phishing")

    if downloaded is None:
        print("    [!] PhishTank download failed. No cached version available.")
        return []

    try:
        # PhishTank CSV has headers: phish_id,url,phish_detail_url,submission_time,verified,verified_time,online,target
        df = pd.read_csv(downloaded, on_bad_lines="skip")

        if "url" in df.columns:
            urls = df["url"].dropna().tolist()
        elif "URL" in df.columns:
            urls = df["URL"].dropna().tolist()
        else:
            # Try first column
            urls = df.iloc[:, 1].dropna().tolist()

        # Filter to only http/https URLs
        urls = [u for u in urls if str(u).startswith(("http://", "https://"))]

        print(f"    Loaded {len(urls)} phishing URLs from PhishTank")
        random.shuffle(urls)
        return urls[:n]

    except Exception as e:
        print(f"    [!] Error parsing PhishTank: {e}")
        return []


def load_urlhaus_urls(n=300):
    """Load active malware URLs from URLhaus."""
    print(f"\n[+] Loading malware URLs from URLhaus...")

    cache_path = CACHE_DIR / "urlhaus_online.csv"
    downloaded = download_with_fallback(URLHAUS_URL, cache_path, "URLhaus online malware URLs")

    if downloaded is None:
        print("    [!] URLhaus download failed. No cached version available.")
        return []

    try:
        # URLhaus CSV starts with comment lines (# ...), then headers:
        # id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter
        lines = []
        with open(downloaded, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    lines.append(line)

        if not lines:
            print("    [!] No data rows found in URLhaus CSV")
            return []

        # Parse the non-comment lines as CSV
        from io import StringIO
        csv_data = StringIO("".join(lines))
        df = pd.read_csv(csv_data, on_bad_lines="skip")

        if "url" in df.columns:
            urls = df["url"].dropna().tolist()
        else:
            # Try third column (index 2)
            urls = df.iloc[:, 2].dropna().tolist()

        # Filter to only http/https URLs
        urls = [u for u in urls if str(u).startswith(("http://", "https://"))]

        print(f"    Loaded {len(urls)} malware URLs from URLhaus")
        random.shuffle(urls)
        return urls[:n]

    except Exception as e:
        print(f"    [!] Error parsing URLhaus: {e}")
        return []


def generate_qr_png(url):
    """Generate a QR code PNG image in memory."""
    qr = segno.make(url, error="m", micro=False)
    buffer = io.BytesIO()
    qr.save(buffer, kind="png", scale=10, border=4, dark="black", light="white")
    buffer.seek(0)
    return buffer.read()


def create_eml(template, url, qr_image_bytes, to_addr="victim@example.com"):
    """Create a .eml email with an embedded QR code image."""

    msg = MIMEMultipart("related")
    msg["Subject"] = template["subject"]
    msg["From"] = template["from"]
    msg["To"] = to_addr
    msg["MIME-Version"] = "1.0"

    # HTML body with inline image reference
    html_part = MIMEText(template["body"], "html")
    msg.attach(html_part)

    # Attach QR code as inline image
    img_part = MIMEImage(qr_image_bytes, _subtype="png")
    img_part.add_header("Content-ID", "<qrcode>")
    img_part.add_header("Content-Disposition", "inline", filename="qrcode.png")
    msg.attach(img_part)

    return msg


def main():
    print("=" * 60)
    print("STEP 6: GENERATE EMAIL QR CODES (Real-World URLs)")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Target: {NUM_BENIGN} benign + {NUM_MALICIOUS} malicious emails")
    print()
    print("  Sources:")
    print("    Benign:    Tranco Top Sites (tranco-list.eu)")
    print("    Malicious: PhishTank (phishtank.org) + URLhaus (urlhaus.abuse.ch)")

    random.seed(SEED)

    # Create directories
    BENIGN_DIR.mkdir(parents=True, exist_ok=True)
    MALICIOUS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load URLs from external sources ──
    benign_urls = load_tranco_urls(n=NUM_BENIGN)

    phishtank_urls = load_phishtank_urls(n=NUM_MALICIOUS)
    urlhaus_urls = load_urlhaus_urls(n=NUM_MALICIOUS)

    # Combine malicious URLs: PhishTank + URLhaus, deduplicate
    malicious_urls = list(set(phishtank_urls + urlhaus_urls))
    random.shuffle(malicious_urls)
    malicious_urls = malicious_urls[:NUM_MALICIOUS]

    print(f"\n  URL Summary:")
    print(f"    Benign (Tranco):         {len(benign_urls)}")
    print(f"    Malicious (PhishTank):   {len(phishtank_urls)}")
    print(f"    Malicious (URLhaus):     {len(urlhaus_urls)}")
    print(f"    Malicious (combined):    {len(malicious_urls)}")

    if len(benign_urls) == 0 or len(malicious_urls) == 0:
        print("\n[!] Not enough URLs. Check internet connection or cached files.")
        return

    manifest = []

    # Generate benign emails
    print(f"\n[+] Generating {len(benign_urls)} benign emails...")
    benign_success = 0
    for i, url in enumerate(benign_urls):
        template = random.choice(BENIGN_TEMPLATES)
        try:
            qr_bytes = generate_qr_png(url)
            eml = create_eml(template, url, qr_bytes)

            filename = f"benign_{i:05d}.eml"
            filepath = BENIGN_DIR / filename
            with open(filepath, "w") as f:
                gen = generator.Generator(f)
                gen.flatten(eml)

            manifest.append({
                "filename": filename,
                "label": "benign",
                "url": url,
                "source": "tranco",
                "subject": template["subject"],
                "from": template["from"],
            })
            benign_success += 1
        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"    [!] Error {i}: {e}")

    # Generate malicious emails
    print(f"[+] Generating {len(malicious_urls)} malicious emails...")
    malicious_success = 0
    for i, url in enumerate(malicious_urls):
        template = random.choice(MALICIOUS_TEMPLATES)
        try:
            qr_bytes = generate_qr_png(url)
            eml = create_eml(template, url, qr_bytes)

            filename = f"malicious_{i:05d}.eml"
            filepath = MALICIOUS_DIR / filename
            with open(filepath, "w") as f:
                gen = generator.Generator(f)
                gen.flatten(eml)

            manifest.append({
                "filename": filename,
                "label": "malicious",
                "url": url,
                "source": "phishtank+urlhaus",
                "subject": template["subject"],
                "from": template["from"],
            })
            malicious_success += 1
        except Exception as e:
            if i < 5:
                print(f"    [!] Error {i}: {e}")

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(MANIFEST_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Benign emails generated:    {benign_success}")
    print(f"  Malicious emails generated: {malicious_success}")
    print(f"  Total:                      {benign_success + malicious_success}")
    print(f"\n  Emails saved to: {EMAIL_DIR}")
    print(f"  Manifest saved to: {MANIFEST_FILE}")
    print(f"  Cached URLs saved to: {CACHE_DIR}")
    print(f"\n  Cached URL files (for reproducibility):")
    for f in CACHE_DIR.iterdir():
        size = f.stat().st_size / 1024
        print(f"    {f.name} ({size:.0f} KB)")
    print("\n    Next step: Run '07_email_qr_pipeline.py'")


if __name__ == "__main__":
    main()
