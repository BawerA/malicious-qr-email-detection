# Detection of Malicious QR Codes Embedded in Emails

Code for the Doctor of Engineering praxis **"Detection of Malicious QR Codes Embedded in Emails through URL Analysis and Machine-Learning Threat Classification"** by Bawer Alissa, The George Washington University, August 2026.

Praxis text: [ProQuest Dissertations & Theses](https://www.proquest.com/openview/6219deffb932ebd1ab468db8dd1cbfd9/1?pq-origsite=gscholar&cbl=18750&diss=y)

Quishing is phishing delivered through a QR code. The malicious link sits inside an image, so email filters that scan text never see it. This praxis builds a two-stage pipeline that opens an email, decodes the QR code, and classifies the URL it contains.

## What this praxis found

1. **Training data coverage matters more than model architecture.** Two models share the same features, grid search, and seed. The Rivas Replication Model scores about 95.6% on its own held-out split and 48.78% on emails with embedded QR codes. Its training data has almost no plain bare-domain benign URLs (no `www` prefix, no trailing slash), so it flags nearly every one as malicious.
2. **The Bawer Model holds up on independent data.** Trained on a dataset that adds bare-domain benign URLs and recent malicious URLs, it reaches 93.88% accuracy on an independent email test set that shares no URL with its training data.
3. **Reputation services raise recall, not precision.** Adding VirusTotal and Google Safe Browsing lifted recall to 99.18% on the independent set. Precision did not improve, so the hypothesis that it would (H2) is not supported.

## How the pipeline works

```
.eml email
  -> extract inline images
  -> decode QR code (ZXing if installed, then OpenCV)
  -> URL
  -> Stage 1: XGBoost on 63 lexical URL features (offline, no network calls)
  -> Stage 2: VirusTotal + Google Safe Browsing, combined with the Stage 1 score
  -> verdict: benign (< 0.30), suspicious (0.30 to 0.50), malicious (>= 0.50)
```

Stage 2 combines the scores with the **Probability Combination Formula**:

```
Score = 1 - (1 - ML) x (1 - P_VT) x (1 - P_GSB)
```

| Reputation signal | Contribution |
|---|---|
| Google Safe Browsing hit | P_GSB = 0.95 |
| VirusTotal score >= 0.30 | P_VT = 0.90 |
| VirusTotal score >= 0.10 | P_VT = 0.75 |
| VirusTotal score >= 0.05 | P_VT = 0.45 |
| No signal | 0 |

This is a probabilistic OR of independent evidence sources. A silent source contributes nothing, the score can only rise, and agreement between sources compounds.

## The two models

| | Rivas Replication Model | Bawer Model |
|---|---|---|
| Training data | Rivas Replication Dataset, about 650,000 URLs from the five public sources used by Rivas et al. (2025) | Bawer Dataset, 611,246 URLs: the Rivas sources plus Tranco bare domains and recent URLhaus URLs |
| Benign URL format | 88% full-path, 12% bare-domain | 60% full-path, 40% bare-domain |
| Features | 63 lexical | 63 lexical |
| Grid search | 72 combinations | 72 combinations |
| Split and seed | 75 / 15 / 10 stratified, seed 42 | same |
| Selected threshold | 0.69 | 0.50 |
| `scale_pos_weight` | 3.03 | 1.14 |

One training setting is not identical between the two scripts. `pipeline/04_train_models.py` multiplies the class ratio by a 1.5 recall boost when it sets `scale_pos_weight` (2.02 x 1.5 = 3.03). `pipeline/Bawer_Dataset/03_train_model.py` uses the plain class ratio (1.14).

## Results

All email test set metrics treat malicious as the positive class. QR codes that could not be decoded are excluded from the metrics.

### Email QR Code Test Set 2 (independent)

500 URLs from sources not used anywhere else in this praxis: Majestic Million (benign), OpenPhish and TweetFeed (malicious). No URL appears in either model's training data. 490 decoded and classified.

| Model | Stage | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| Bawer | 1 (lexical) | 93.88% | 92.16% | 95.92% | 94.00% | 8.16% |
| Bawer | 2 (with reputation) | 95.09% | 91.67% | 99.18% | 95.28% | 8.98% |
| Rivas Replication | 1 (lexical) | 52.65% | 51.39% | 98.37% | 67.51% | 93.06% |
| Rivas Replication | 2 (with reputation) | 51.75% | 51.05% | 99.18% | 67.41% | 96.27% |

Stage 1 confusion matrices: Bawer TP 235, FP 20, TN 225, FN 10. Rivas Replication TP 241, FP 228, TN 17, FN 4.

All 20 Bawer false positives are full-path benign URLs. None of the 220 bare-domain benign URLs were misclassified.

### Email QR Code Test Set 1

1,000 emails with URLs from Tranco (benign) and PhishTank and URLhaus (malicious). 982 decoded and classified.

| Model | Stage | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| Bawer | 1 (lexical) | 97.86% | 97.94% | 97.74% | 97.84% | 2.02% |
| Bawer | 2 (with reputation) | 98.88% | 97.98% | 99.79% | 98.88% | 2.03% |
| Rivas Replication | 1 (lexical) | 48.78% | 49.15% | 94.46% | 64.65% | 96.16% |
| Rivas Replication | 2 (with reputation) | 49.64% | 49.69% | 98.76% | 66.11% | 98.98% |

Stage 1 confusion matrices: Bawer TP 476, FP 10, TN 485, FN 11. Rivas Replication TP 460, FP 476, TN 19, FN 27.

**Read these numbers with the overlap in mind.** Test Set 1 and the Bawer Dataset both draw on Tranco and URLhaus. Comparing the saved files shows that 280 of the 500 benign URLs and 173 of the 500 malicious URLs in Test Set 1 also appear in the Bawer training split, and 46 and 22 more appear in its validation split. None appear in the Rivas Replication Dataset. Test Set 1 therefore overstates how well the Bawer Model generalizes. Test Set 2 is the independent estimate.

### Held-out test split

On the held-out 15% of its own dataset, the Rivas Replication Model reaches about 95.6% accuracy (AUC 0.991) and the Bawer Model about 96.5% (AUC 0.995). See `pipeline/results/training_results.csv` and `pipeline/Bawer_Dataset/results/training_results.csv`. Precision, recall, and F1 in those two files are weighted averages across both classes.

### Hypotheses

Evaluated for the Bawer Model on Email QR Code Test Set 1, as defined in the praxis.

| Hypothesis | Result |
|---|---|
| H1: precision and recall of at least 90% | Supported |
| H2: reputation enrichment raises precision by at least 3 percentage points | Not supported |
| H3: false-positive rate of 5% or lower | Supported on Test Set 1 (2.02%). Not met on Test Set 2 (8.16%) |

## Repository layout

```
src/fetch/
  url_features.py        lexical feature extractor (68 raw features, 63 used)
  qr_decode.py           QR decoding and URL cleanup
pipeline/
  01_create_dataset.py   to   09_final_evaluation.py     Rivas Replication pipeline
  test_homoglyph_capability.py                           Cyrillic lookalike check
  config/api_keys.example.json                           template for Stage 2 keys
  results/                                               aggregate results, Test Set 1
  Bawer_Dataset/
    01_collect_urls.py   to   05_reputation_enrichment.py   Bawer pipeline
    models/bawer_63_XGBoost.joblib                          trained Bawer Model bundle
    results/                                                aggregate results, Test Set 1
  Email_Test_Set_2/                                      aggregate results, Test Set 2
docs/DATA_SOURCES.md     where each dataset comes from
```

## Running it

```
pip install -r requirements.txt
```

1. Download the source datasets listed in `docs/DATA_SOURCES.md` into `pipeline/data/raw_downloads/`.
2. Run the numbered scripts in `pipeline/` in order. `pipeline/README.md` describes each step.
3. Run the numbered scripts in `pipeline/Bawer_Dataset/` in order.
4. For Stage 2, copy `pipeline/config/api_keys.example.json` to `pipeline/config/api_keys.json` and add your own VirusTotal and Google Safe Browsing keys, or set them as environment variables. The free VirusTotal tier allows 500 lookups a day.

Classifying one URL with the trained bundle:

```python
import joblib, pandas as pd
from src.fetch.url_features import url_features_enhanced

bundle = joblib.load("pipeline/Bawer_Dataset/models/bawer_63_XGBoost.joblib")
features = url_features_enhanced("https://example.com/menu")
X = pd.DataFrame([features]).reindex(columns=bundle["feature_names"], fill_value=0)
prob = float(bundle["model"].predict_proba(bundle["scaler"].transform(X.values))[0][1])
print("malicious" if prob >= bundle["optimal_threshold"] else "benign", prob)
```

Only load `.joblib` files from a source you trust. They are Python pickles.

## What is not in this repository

- The training datasets. They belong to third parties. Sources are in `docs/DATA_SOURCES.md`.
- The generated `.eml` files, per-URL result files, and reputation caches. They contain live malicious URLs.
- API keys.

## Limitations

1. **Test Set 1 overlaps the Bawer training data.** See the note under its results table.
2. **Full-path benign URLs are the weak point.** The false-positive rate rises with the share of full-path benign URLs. An archived composition of Test Set 2 with 100 full-path benign URLs out of 250 gave the Bawer Model 84.05% accuracy and a 27.87% false-positive rate (`pipeline/Email_Test_Set_2/metrics.json`).
3. **Split QR codes are not handled.** A URL split across several codes with Structured Append is not reassembled.
4. **Unicode homoglyphs are not detected.** The Bawer Model does not flag Cyrillic lookalike domains. `pipeline/test_homoglyph_capability.py` shows this.
5. **No page content is inspected.** A clean-looking malicious URL that no reputation service knows can pass both stages.
6. **One QR decoder was used in the reported runs.** The decoder tries ZXing first when it is installed. The reported runs used OpenCV only. 18 of 1,000 codes in Test Set 1 and 10 of 500 in Test Set 2 could not be decoded.
7. **The Stage 2 formula was chosen using Test Set 1.** Test Set 2 is the independent check of Stage 2.
8. **The test sets are generated.** No public corpus of real quishing emails exists, so both sets were built for this praxis.

## Citation

Alissa, B. (2026). *Detection of malicious QR codes embedded in emails through URL analysis and machine-learning threat classification* (Doctor of Engineering praxis). The George Washington University.

Baseline replicated: Rivas et al. (2025), as cited in the praxis.
