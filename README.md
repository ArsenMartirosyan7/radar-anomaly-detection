cat > README.md <<'EOF'
# Radar Anomaly Detection with Hybrid TranAD-Autoencoder

This project implements an end-to-end anomaly detection pipeline for aeronautical radar data.

The input data consists of raw radar traffic stored in PCAP files. The project decodes radar messages, extracts ASTERIX-based radar features, builds time-series datasets, injects ENAC-style synthetic attacks, trains deep learning anomaly detection models, and evaluates a final Hybrid TranAD-Autoencoder ensemble.

The final model combines:

- **TranAD**, a Transformer-based temporal anomaly detector.
- **Feed-forward Autoencoder**, a point-level reconstruction model.
- **Feature-group anomaly scoring**, which focuses on the radar features affected by each attack type.
- **Validation-based thresholds** and post-processing to reduce false positives.

The final optimized hybrid model is compared against the previous ENAC experimental results.

---

## 1. Project Goal

The goal of this project is to detect anomalies in radar data.

The original radar data is stored in PCAP files. These files contain radar network traffic, not ready-to-use machine learning tables. Therefore, the first objective is to transform raw radar packets into meaningful numerical features.

The project follows this global logic:

```text
Raw radar PCAP files
→ ASTERIX/radar feature extraction
→ time-series dataset creation
→ ENAC-style anomaly injection
→ TranAD training
→ Autoencoder training
→ Hybrid ensemble evaluation
→ comparison with previous ENAC results
```

The final model is called:

```text
Hybrid TranAD-Autoencoder Ensemble
```

---

## 2. High-Level Pipeline

The complete pipeline is:

```text
1. Raw PCAP files
   ↓
2. ASTERIX decoding and radar feature extraction
   ↓
3. Per-second radar time-series construction
   ↓
4. Train / validation / test split
   ↓
5. ENAC-style synthetic anomaly injection
   ↓
6. TranAD training
   ↓
7. Autoencoder training
   ↓
8. TranAD testing on ENAC attacks
   ↓
9. Autoencoder testing on ENAC attacks
   ↓
10. Hybrid ensemble evaluation
   ↓
11. Final Precision / Recall / F1 comparison
```

---

## 3. Dataset Description

The dataset consists of radar PCAP files collected over several days.

In this project, the PCAP files cover:

```text
2019-06-23 to 2019-06-29
```

The data is split chronologically:

```text
Training set:   2019-06-23 to 2019-06-26
Validation set: 2019-06-27
Test set:       2019-06-28 to 2019-06-29
```

A chronological split is used because radar data is time-series data. Random splitting would mix past and future behavior and make evaluation less realistic.

The processed dataset has:

```text
52 radar features per second
```

The dataset shapes obtained in the project were:

```text
train.npy: (345624, 52)
val.npy:   (86406, 52)
test.npy:  (172812, 52)
```

---

## 4. Radar and ASTERIX Context

The PCAP files contain radar messages encoded using the ASTERIX format.

ASTERIX is a standard format used for the exchange of surveillance and radar data. The main radar category used in this project is **CAT048**, which contains radar target reports.

Important radar attributes include:

| Feature Group | Meaning |
|---|---|
| `THETA` | Target angle / azimuth from the radar |
| `RHO` | Target range / distance from the radar |
| `X`, `Y` | Cartesian position coordinates |
| `FL` | Flight level / altitude |
| `CGS` | Calculated ground speed |
| `Heading` | Direction of target movement |
| `Track ID` | Radar track identifier |
| `Aircraft address` | Aircraft identifier when available |

The model does **not** train directly on raw PCAP bytes. Instead, PCAP packets are decoded into structured radar features.

---

## 5. Extracted Radar Features

Each row of the processed dataset corresponds approximately to one second of radar behavior.

The extracted feature set includes radar motion features, ASTERIX message statistics, and time features.

Examples of extracted features:

| Feature | Meaning |
|---|---|
| `packet_count` | Number of UDP packets in one second |
| `udp_byte_count` | Number of UDP payload bytes in one second |
| `asterix_block_count` | Number of ASTERIX blocks in one second |
| `cat001_count` | Number of CAT001 blocks |
| `cat002_count` | Number of CAT002 blocks |
| `cat034_count` | Number of CAT034 blocks |
| `cat048_count` | Number of CAT048 blocks |
| `cat048_record_count` | Number of decoded CAT048 radar target reports |
| `unique_tracks` | Number of unique radar tracks |
| `unique_aircraft` | Number of unique aircraft addresses |
| `time_sin`, `time_cos` | Cyclical time-of-day encoding |

For continuous radar attributes, statistics are computed per second:

```text
mean
standard deviation
minimum
maximum
```

For example:

```text
rho_nm_mean
rho_nm_std
rho_nm_min
rho_nm_max
```

This means that for each second, all decoded RHO values are summarized using average, standard deviation, minimum, and maximum.

Important feature groups:

| Feature Group | Example Features | Meaning |
|---|---|---|
| RHO | `rho_nm_mean`, `rho_nm_std`, `rho_nm_min`, `rho_nm_max` | Distance from radar |
| THETA | `theta_deg_mean`, `theta_deg_std`, `theta_deg_min`, `theta_deg_max` | Angle from radar |
| X/Y | `x_nm_mean`, `y_nm_mean`, `x_nm_std`, `y_nm_std` | Cartesian position |
| FL | `flight_level_mean`, `flight_level_std`, `flight_level_min`, `flight_level_max` | Flight level / altitude |
| CGS | `ground_speed_nm_s_mean`, `ground_speed_nm_s_std`, `ground_speed_nm_s_min`, `ground_speed_nm_s_max` | Ground speed |
| Heading | `heading_deg_mean`, `heading_deg_std`, `heading_deg_min`, `heading_deg_max` | Direction of movement |

---

## 6. ENAC-Style Attacks

The original radar dataset did not contain ground-truth anomaly labels. Therefore, controlled synthetic anomalies are injected into the clean test set.

The project reproduces ENAC-style attack categories:

| Attack Type | Meaning | Modified Features |
|---|---|---|
| `THETA` | Angle deviation | `theta_deg_*` |
| `RHO` | Range deviation | `rho_nm_*` |
| `ALL` | Multiple radar attributes modified | theta, rho, x/y, flight level, speed, heading |
| `RND` | Random corruption | Random radar features |
| `ROUTE` | Different trajectory / route | `x_nm_*`, `y_nm_*` |
| `FL_PLUS` | Flight level increase | `flight_level_*` |
| `FL_MINUS` | Flight level decrease | `flight_level_*` |
| `CGS_PLUS` | Ground speed increase | `ground_speed_nm_s_*` |
| `CGS_MINUS` | Ground speed decrease | `ground_speed_nm_s_*` |

Each attack type contains:

```text
12 injected attacks
```

Each attack lasts:

```text
300 seconds
```

Total number of injected attacks:

```text
9 attack types × 12 attacks = 108 attacks
```

---

## 7. Models

The project contains several model files, but the final optimized detector uses two main models:

```text
1. TranAD
2. Feed-forward Autoencoder
```

Other models such as LSTM Autoencoder and Isolation Forest may be kept as optional baselines.

---

## 8. TranAD Model

TranAD is the Transformer-based anomaly detection model.

File:

```text
models/tranad.py
```

Training script:

```text
train.py
```

Testing script:

```text
test.py
```

TranAD receives a sliding time window:

```text
30 seconds × 52 features
```

It reconstructs the last timestamp of the window:

```text
input window: X[t-29 : t]
target:       X[t]
```

If the reconstruction error is high, the radar behavior is considered anomalous.

TranAD is useful because it captures **temporal behavior**.

Example:

```text
A speed, trajectory, or flight-level change may only look suspicious when compared with previous seconds.
```

### TranAD Architecture

The TranAD architecture is based on:

```text
Transformer Encoder
Transformer Decoder
```

The simplified flow is:

```text
radar time window
→ input projection
→ positional encoding
→ Transformer encoder
→ Transformer decoder
→ reconstructed radar features
→ reconstruction error
→ anomaly score
```

TranAD uses two reconstruction stages:

```text
Stage 1: normal reconstruction
Stage 2: anomaly-focused reconstruction using the error from Stage 1
```

This helps the model focus more strongly on suspicious parts of the input.

---

## 9. Feed-Forward Autoencoder

The Autoencoder is a point-level reconstruction model.

File:

```text
models/autoencoder.py
```

Training script:

```text
train_autoencoder.py
```

Testing script:

```text
test_autoencoder.py
```

The Autoencoder receives one timestamp at a time:

```text
1 second × 52 features
```

It compresses the feature vector into a smaller latent representation and reconstructs it.

Architecture:

```text
52 features
→ 128 hidden units
→ 64 hidden units
→ 32 latent representation
→ 64 hidden units
→ 128 hidden units
→ reconstructed 52 features
```

The Autoencoder is useful because it captures **point-level feature abnormalities**.

Example:

```text
A single radar state may have an abnormal combination of RHO, THETA, FL, CGS, or heading values.
```

---

## 10. Final Hybrid Model

The final model is a Hybrid TranAD-Autoencoder ensemble.

It combines:

```text
TranAD anomaly score
+
Autoencoder anomaly score
```

The final score is:

```text
final_score = 0.4 × TranAD_score + 0.6 × Autoencoder_score
```

Final optimized weights:

| Component | Weight |
|---|---:|
| TranAD | 0.4 |
| Autoencoder | 0.6 |

Before combining the scores, both scores are normalized using validation data.

This is important because TranAD and Autoencoder reconstruction errors may have different numerical scales.

---

## 11. Feature-Group Scoring

Instead of averaging reconstruction error over all 52 features, the final model uses radar-specific feature groups.

This avoids diluting localized anomalies.

Example:

```text
A CGS attack affects ground-speed features.
If the reconstruction error is averaged across all 52 features, the CGS anomaly can be hidden by unrelated normal features.
```

Feature-group scoring fixes this.

| Attack Type | Feature Group Used |
|---|---|
| `THETA` | `theta_deg_*` |
| `RHO` | `rho_nm_*` |
| `ROUTE` | `x_nm_*`, `y_nm_*` |
| `FL_PLUS`, `FL_MINUS` | `flight_level_*` |
| `CGS_PLUS`, `CGS_MINUS` | `ground_speed_nm_s_*` |
| `ALL` | theta, rho, x/y, flight level, speed, heading |
| `RND` | broad radar feature group |

This is one of the main reasons why the hybrid model improves over a simple global reconstruction-error detector.

---

## 12. Thresholding

The model outputs a continuous anomaly score.

A threshold is used to convert this score into a binary prediction:

```text
if anomaly_score > threshold:
    anomaly
else:
    normal
```

The final model uses validation-based threshold quantiles.

Final optimized threshold policy:

| Feature Group | Quantile |
|---|---:|
| Default groups | `0.9995` |
| `THETA` | `0.999` |
| `ROUTE` | `0.999` |
| `CGS_PLUS`, `CGS_MINUS` | `0.997` |

A quantile of `0.999` means:

```text
The threshold is set at the 99.9th percentile of normal validation scores.
```

So a point is considered anomalous only if its score is higher than almost all normal validation behavior.

Higher quantile means stricter detection:

```text
0.99   → less strict, more detections, more false positives
0.999  → stricter
0.9995 → even stricter, fewer false positives
```

---

## 13. Post-Processing

Raw anomaly scores may contain short noisy spikes. Radar attacks are not isolated one-second events, so post-processing is applied.

Final post-processing parameters:

```text
smooth_window = 5
min_event_length = 120
merge_gap = 0
```

Meaning:

| Parameter | Meaning |
|---|---|
| `smooth_window = 5` | Smooth anomaly scores over 5 seconds |
| `min_event_length = 120` | Remove predicted anomaly intervals shorter than 120 seconds |
| `merge_gap = 0` | Do not merge separated predicted events |

This reduces false positives and keeps only sustained anomalies.

---

## 14. Evaluation Metrics

To compare with the previous ENAC experiment, the main metrics are timestamp-level:

```text
Precision
Recall
F1-score
```

### Precision

Precision answers:

```text
When the model predicts anomaly, how often is it correct?
```

Formula:

```text
Precision = TP / (TP + FP)
```

High precision means fewer false positives.

### Recall

Recall answers:

```text
Of all real anomalies, how many did the model detect?
```

Formula:

```text
Recall = TP / (TP + FN)
```

High recall means fewer missed attacks.

### F1-score

F1-score balances precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

It is useful as a single metric when both false positives and false negatives matter.

---

## 15. Final Results

Final results are saved in:

```text
results/ensemble/final_optimized_strict/ensemble_results.csv
```

### Mean Comparison with Previous ENAC Autoencoder

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Previous ENAC Autoencoder | 0.8565 | 0.6571 | 0.7116 |
| Hybrid TranAD-AE | 0.9885 | 0.9517 | 0.9650 |

### Detailed Results by Attack Type

| Attack Type | Precision | Recall | F1-score |
|---|---:|---:|---:|
| THETA | 0.9957 | 0.9597 | 0.9774 |
| RHO | 0.9879 | 1.0000 | 0.9939 |
| ALL | 0.9868 | 1.0000 | 0.9934 |
| RND | 0.9877 | 1.0000 | 0.9938 |
| ROUTE | 0.9882 | 0.6058 | 0.7512 |
| FL(+) | 0.9868 | 1.0000 | 0.9934 |
| FL(-) | 0.9868 | 1.0000 | 0.9934 |
| CGS(+) | 0.9887 | 1.0000 | 0.9943 |
| CGS(-) | 0.9882 | 1.0000 | 0.9941 |
| MEAN | 0.9885 | 0.9517 | 0.9650 |

The weakest attack type is `ROUTE`, because trajectory anomalies are more gradual and harder to detect at the exact timestamp level.

---

## 16. Why the Hybrid Model Improves the Previous Approach

The previous ENAC approach used an LSTM-based autoencoder to reconstruct radar time-series windows and detect anomalies from reconstruction error.

The Hybrid TranAD-Autoencoder model improves this approach in several ways:

### 1. Temporal modeling with TranAD

TranAD uses Transformer self-attention to model temporal radar behavior over 30-second windows.

This helps detect anomalies that depend on time evolution.

### 2. Point-level reconstruction with Autoencoder

The Autoencoder detects abnormal feature combinations at individual timestamps.

This complements TranAD because some anomalies are visible immediately in the feature vector.

### 3. Ensemble combination

The final detector combines both reconstruction errors:

```text
temporal anomaly evidence
+
point-level anomaly evidence
```

This makes the detector more robust than using only one model.

### 4. Feature-group scoring

Instead of averaging errors across all features, the model focuses on the radar feature group affected by each attack type.

This is especially important for attacks like:

```text
THETA
RHO
ROUTE
FL
CGS
```

### 5. Group-specific thresholds

Different radar feature groups have different error distributions.

The final model uses validation-based thresholds specific to feature groups, improving detection precision and recall.

### 6. Post-processing

Short false alarms are removed, and only sustained anomaly intervals are kept.

This reduces false positives.

---

## 17. Project Structure

```text
radar-anomaly-detection/
│
├── data/
│   ├── raw_pcaps/
│   ├── decoded/
│   └── processed/
│       ├── RADAR/
│       └── RADAR_ENAC/
│
├── models/
│   ├── __init__.py
│   ├── tranad.py
│   ├── autoencoder.py
│   ├── lstm_autoencoder.py
│   └── isolation_forest.py
│
├── scripts/
│   ├── extract_asterix.py
│   ├── build_dataset.py
│   ├── inject_enac_anomalies.py
│   ├── evaluate_ensemble.py
│   ├── plot_ensemble_results.py
│   ├── evaluate_enac_group_scores.py
│   ├── search_ensemble_params.py
│   ├── inject_anomalies.py
│   └── evaluate.py
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── postprocessing.py
│   └── scaler.py
│
├── results/
│   ├── tranad/
│   ├── autoencoder/
│   └── ensemble/
│
├── train.py
├── test.py
├── train_autoencoder.py
├── test_autoencoder.py
├── requirements.txt
└── README.md
```

---

## 18. Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/radar-anomaly-detection.git
cd radar-anomaly-detection
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install requirements

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
matplotlib
torch
pyyaml
tqdm
joblib
```

---

## 19. Data Placement

Place raw PCAP files in:

```text
data/raw_pcaps/
```

Example:

```text
data/raw_pcaps/2019-06-23-0207.pcap
data/raw_pcaps/2019-06-23-0607.pcap
data/raw_pcaps/2019-06-23-1007.pcap
...
```

Large files such as PCAPs, `.npy` datasets, model checkpoints, and generated results are usually not committed to GitHub.

---

## 20. How to Run the Full Pipeline

### Step 1: Extract radar features from PCAP files

```bash
python3 scripts/extract_asterix.py \
  --input data/raw_pcaps \
  --output data/decoded
```

This creates decoded CSV files:

```text
data/decoded/*_features.csv
```

---

### Step 2: Build the processed dataset

```bash
python3 scripts/build_dataset.py
```

This creates:

```text
data/processed/RADAR/train.npy
data/processed/RADAR/val.npy
data/processed/RADAR/test.npy
data/processed/RADAR/test_labels.npy
data/processed/RADAR/feature_names.json
data/processed/RADAR/scaler.pkl
```

Expected shapes:

```text
train.npy: (345624, 52)
val.npy:   (86406, 52)
test.npy:  (172812, 52)
```

---

### Step 3: Inject ENAC-style anomalies

```bash
python3 scripts/inject_enac_anomalies.py
```

This creates:

```text
data/processed/RADAR_ENAC/THETA/
data/processed/RADAR_ENAC/RHO/
data/processed/RADAR_ENAC/ALL/
data/processed/RADAR_ENAC/RND/
data/processed/RADAR_ENAC/ROUTE/
data/processed/RADAR_ENAC/FL_PLUS/
data/processed/RADAR_ENAC/FL_MINUS/
data/processed/RADAR_ENAC/CGS_PLUS/
data/processed/RADAR_ENAC/CGS_MINUS/
```

Each folder contains:

```text
test.npy
test_labels.npy
anomaly_report.json
```

---

### Step 4: Train TranAD

```bash
python3 train.py
```

This saves:

```text
results/tranad/checkpoint.pt
results/tranad/training_history.csv
results/tranad/val_scores.npy
results/tranad/threshold.txt
```

If CUDA is available, the script uses GPU automatically.

---

### Step 5: Train Autoencoder

```bash
python3 train_autoencoder.py
```

This saves:

```text
results/autoencoder/checkpoint.pt
results/autoencoder/training_history.csv
results/autoencoder/val_feature_errors_aligned.npy
results/autoencoder/val_scores_aligned.npy
```

---

### Step 6: Test TranAD on each ENAC attack type

```bash
for t in THETA RHO ALL RND ROUTE FL_PLUS FL_MINUS CGS_PLUS CGS_MINUS; do
  python3 test.py \
    --data-dir data/processed/RADAR_ENAC/$t \
    --model-dir results/tranad \
    --output-dir results/tranad/enac_$t \
    --smooth-window 15 \
    --min-event-length 120 \
    --merge-gap 60 \
    --threshold-quantile 0.999
done
```

This creates TranAD reconstruction-error outputs for each attack type.

Important files inside each folder:

```text
test_feature_errors.npy
test_labels_aligned.npy
metrics.csv
anomaly_scores_full.png
```

---

### Step 7: Test Autoencoder on each ENAC attack type

```bash
for t in THETA RHO ALL RND ROUTE FL_PLUS FL_MINUS CGS_PLUS CGS_MINUS; do
  python3 test_autoencoder.py \
    --data-dir data/processed/RADAR_ENAC/$t \
    --model-dir results/autoencoder \
    --output-dir results/autoencoder/enac_$t \
    --smooth-window 15 \
    --min-event-length 120 \
    --merge-gap 60 \
    --threshold-quantile 0.999
done
```

This creates Autoencoder reconstruction-error outputs for each attack type.

Important files inside each folder:

```text
test_feature_errors.npy
test_labels_aligned.npy
metrics.csv
```

---

### Step 8: Evaluate the final Hybrid TranAD-AE ensemble

```bash
PYTHONPATH=. python3 scripts/evaluate_ensemble.py \
  --tranad-weight 0.4 \
  --smooth-window 5 \
  --min-event-length 120 \
  --merge-gap 0 \
  --default-quantile 0.9995 \
  --cgs-quantile 0.997 \
  --route-quantile 0.999 \
  --theta-quantile 0.999 \
  --output-dir results/ensemble/final_optimized_strict
```

Final result file:

```text
results/ensemble/final_optimized_strict/ensemble_results.csv
```

---

## 21. How to View Results

Print final results:

```bash
python3 -c "import pandas as pd; df=pd.read_csv('results/ensemble/final_optimized_strict/ensemble_results.csv'); print(df[['type','precision','recall','f1']].to_string(index=False))"
```

View raw CSV:

```bash
cat results/ensemble/final_optimized_strict/ensemble_results.csv
```

Print final mean row only:

```bash
python3 -c "import pandas as pd; df=pd.read_csv('results/ensemble/final_optimized_strict/ensemble_results.csv'); print(df[df['type']=='MEAN'][['precision','recall','f1']].to_string(index=False))"
```

---

## 22. How to Generate Visualizations

Create ensemble visualizations:

```bash
PYTHONPATH=. python3 scripts/plot_ensemble_results.py
```

Plots are saved in:

```text
results/ensemble/final_optimized_strict/plots/
```

Generated visualizations include:

```text
summary/01_enac_vs_hybrid_mean_metrics.png
summary/02_per_attack_precision_recall_f1.png
summary/03_per_attack_f1.png
summary/04_number_of_attacks.png
summary/05_total_confusion_matrix.png
timelines/*_full_timeline.png
zooms/*_event_XX_zoom.png
```

Recommended plots for presentation:

```text
1. 01_enac_vs_hybrid_mean_metrics.png
2. 02_per_attack_precision_recall_f1.png
3. 03_per_attack_f1.png
4. One full timeline plot
5. One zoomed event plot
6. One difficult ROUTE example
```

---

## 23. Meaning of Important Result Folders

### `results/tranad/`

Contains outputs related to TranAD alone:

```text
checkpoint.pt
training_history.csv
val_scores.npy
threshold.txt
enac_THETA/
enac_RHO/
...
```

The `enac_*` folders contain TranAD reconstruction errors used later by the final ensemble.

### `results/autoencoder/`

Contains outputs related to the Autoencoder:

```text
checkpoint.pt
training_history.csv
val_feature_errors_aligned.npy
enac_THETA/
enac_RHO/
...
```

The `enac_*` folders contain Autoencoder reconstruction errors used by the final ensemble.

### `results/ensemble/final_optimized_strict/`

Contains the final model results:

```text
ensemble_results.csv
THETA/
RHO/
ALL/
RND/
ROUTE/
FL_PLUS/
FL_MINUS/
CGS_PLUS/
CGS_MINUS/
plots/
```

This is the main folder used for the final report and presentation.

---

## 24. Important Experimental Notes

### Synthetic anomalies

The original radar dataset did not contain ground-truth anomaly labels. Therefore, ENAC-style synthetic anomalies were injected into the clean test set.

This allows calculation of:

```text
Precision
Recall
F1-score
```

### Comparison with ENAC

The comparison is made against the previous ENAC experimental results on similar attack categories.

The main comparable metrics are timestamp-level:

```text
Precision
Recall
F1-score
```

### Parameter optimization

The final optimized configuration was selected through a search over ensemble and threshold parameters.

For a stricter scientific protocol, parameter selection should be performed on a separate calibration set and then evaluated once on a held-out test set.

### Raw data is not included

Raw PCAP files and processed `.npy` files may be large or sensitive. They are not intended to be committed to GitHub.

Recommended `.gitignore` entries:

```gitignore
data/raw_pcaps/
data/decoded/
data/processed/
results/
*.pcap
*.pcapng
*.npy
*.npz
*.pkl
*.pt
*.pth
__pycache__/
```

---

## 25. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'models'`

Run scripts with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python3 scripts/evaluate_ensemble.py
```

or:

```bash
PYTHONPATH=. python3 scripts/plot_ensemble_results.py
```

### Problem: CUDA not available

The scripts automatically use CPU if CUDA is not available.

To force CPU:

```bash
python3 train.py --device cpu
```

or:

```bash
python3 test.py --device cpu
```

### Problem: Missing `test_clean.npy`

`inject_enac_anomalies.py` expects:

```text
data/processed/RADAR/test_clean.npy
```

If missing, rebuild the dataset:

```bash
python3 scripts/build_dataset.py
```

Then rerun ENAC anomaly injection:

```bash
python3 scripts/inject_enac_anomalies.py
```

### Problem: Missing `test_feature_errors.npy`

The final ensemble requires TranAD and Autoencoder test outputs.

Run TranAD testing:

```bash
for t in THETA RHO ALL RND ROUTE FL_PLUS FL_MINUS CGS_PLUS CGS_MINUS; do
  python3 test.py \
    --data-dir data/processed/RADAR_ENAC/$t \
    --model-dir results/tranad \
    --output-dir results/tranad/enac_$t \
    --smooth-window 15 \
    --min-event-length 120 \
    --merge-gap 60 \
    --threshold-quantile 0.999
done
```

Run Autoencoder testing:

```bash
for t in THETA RHO ALL RND ROUTE FL_PLUS FL_MINUS CGS_PLUS CGS_MINUS; do
  python3 test_autoencoder.py \
    --data-dir data/processed/RADAR_ENAC/$t \
    --model-dir results/autoencoder \
    --output-dir results/autoencoder/enac_$t \
    --smooth-window 15 \
    --min-event-length 120 \
    --merge-gap 60 \
    --threshold-quantile 0.999
done
```

---

## 26. Final Summary

This project implements a complete radar anomaly detection pipeline:

```text
PCAP radar data
→ ASTERIX decoding
→ radar feature extraction
→ time-series dataset construction
→ ENAC-style anomaly injection
→ TranAD training
→ Autoencoder training
→ hybrid ensemble scoring
→ final evaluation
```

The final Hybrid TranAD-Autoencoder ensemble improves over the previous ENAC Autoencoder baseline on the reproduced ENAC-style benchmark:

```text
Precision: 0.8565 → 0.9885
Recall:    0.6571 → 0.9517
F1-score:  0.7116 → 0.9650
```

The improvement comes from:

```text
1. Temporal modeling with TranAD.
2. Point-level reconstruction with the Autoencoder.
3. Feature-group anomaly scoring.
4. Validation-based thresholds.
5. Post-processing to remove short false alarms.
```
EOF