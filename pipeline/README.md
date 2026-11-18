# Pipeline steps

Run every script from the repository root, for example `python pipeline/01_create_dataset.py`. Each script reads and writes inside `pipeline/`, so no paths need editing.

## Rivas Replication pipeline

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `01_create_dataset.py` | Merges the five source datasets, removes duplicates, downsamples to 650,000 URLs, splits 75 / 15 / 10 | `data/rivas_dataset.csv` and the three splits |
| 2 | `02_extract_features_24.py` | Extracts the 24 lexical features from Rivas et al. (2025) | `data/rivas_24_*.csv` |
| 3 | `03_extract_features_63.py` | Extracts 68 features and drops 5 with zero importance, leaving 63 | `data/our_63_*.csv` |
| 4 | `04_train_models.py` | Trains XGBoost on both feature sets with a 72-combination grid search, early stopping, and threshold selection on the validation split | `models/*.joblib`, `results/training_results.csv` |
| 5 | `05_evaluate_models.py` | Compares the 24-feature and 63-feature models with the published baseline | `results/` |
| 6 | `06_generate_email_qr_codes.py` | Builds Email QR Code Test Set 1: 1,000 `.eml` files with an inline QR code | `data/test_emails/` |
| 7 | `07_email_qr_pipeline.py` | Parses each email, decodes the QR code, classifies the URL | `results/email_pipeline_results.csv` |
| 8 | `08_reputation_enrichment.py` | Stage 2: VirusTotal and Google Safe Browsing, combined by the Probability Combination Formula | `results/enriched_results.csv` |
| 9 | `09_final_evaluation.py` | Evaluates the three hypotheses | `results/final_evaluation.txt` |

## Bawer pipeline

Run these after step 1 above, because the Bawer Dataset starts from `data/rivas_dataset.csv`.

| Step | Script | What it does |
|---|---|---|
| 1 | `Bawer_Dataset/01_collect_urls.py` | Adds Tranco bare domains (benign) and recent URLhaus URLs (malicious) to the Rivas sources |
| 2 | `Bawer_Dataset/02_split_and_extract_features.py` | Splits 75 / 15 / 10 and extracts the same 63 features |
| 3 | `Bawer_Dataset/03_train_model.py` | Trains XGBoost with the same grid search and seed |
| 4 | `Bawer_Dataset/04_email_qr_pipeline.py` | Runs the Bawer Model on the same test emails from step 6 |
| 5 | `Bawer_Dataset/05_reputation_enrichment.py` | Stage 2 for the Bawer Model. Reuses the reputation cache from step 8 |

## Notes

- Steps 6 and 8 need network access. Step 6 downloads live threat feeds, so a new run produces a different test set than the one reported.
- Windows Defender can quarantine CSV files that contain malicious URLs. Add an exclusion for `pipeline/` before running.
- The pipeline is deterministic: the same URLs and the same trained bundle give the same result on any machine.
