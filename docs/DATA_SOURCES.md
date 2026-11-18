# Data sources

No dataset is redistributed in this repository. Download each source yourself and check its terms of use.

## Rivas Replication Dataset

The five sources used by Rivas et al. (2025). Place the files in `pipeline/data/raw_downloads/`.

| Source | File expected | Where to get it |
|---|---|---|
| Kaggle "Malicious URLs Dataset" (sid321axn) | `malicious_phish.csv` | https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset |
| Kaggle "Phishing Site URLs" (taruntiwarihp) | `phishing_site_urls.csv` | https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls |
| Kaggle "Malicious and Benign URLs" (siddharthkumar25) | `malicious_and_benign_urls.csv` | https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls |
| PhishStorm (Marchal et al., 2014) | `phishstorm.csv` | https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset/ |
| ISCX-URL-2016 (Mamun et al., 2016) | `iscx_url_2016.csv` | https://www.unb.ca/cic/datasets/url-2016.html |

The Kaggle sources need a free Kaggle account.

## Bawer Dataset

The Rivas sources above, plus two feeds that `pipeline/Bawer_Dataset/01_collect_urls.py` downloads:

| Source | Used for | Where it comes from |
|---|---|---|
| Tranco top sites list | Bare-domain benign URLs | https://tranco-list.eu |
| URLhaus (abuse.ch) | Recent malicious URLs | https://urlhaus.abuse.ch |

## Email QR Code Test Set 1

Downloaded by `pipeline/06_generate_email_qr_codes.py`.

| Source | Used for |
|---|---|
| Tranco top sites list, ranks 5,000 to 50,000 | Benign URLs |
| PhishTank verified phishing feed | Malicious URLs |
| URLhaus online feed | Malicious URLs |

Tranco and URLhaus are also sources for the Bawer Dataset, so this test set is not independent of it. See the README.

## Email QR Code Test Set 2

Built from sources used nowhere else in this praxis.

| Source | Used for |
|---|---|
| Majestic Million | 225 bare-domain benign URLs and 25 real page URLs from Majestic-ranked sites |
| OpenPhish | Malicious URLs |
| TweetFeed | Malicious URLs |

## Reputation services (Stage 2)

VirusTotal and Google Safe Browsing. Both need your own API key.
