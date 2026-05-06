# Radar Anomaly Detection with Hybrid TranAD-Autoencoder

This project detects anomalies in radar data extracted from PCAP files.  
The pipeline converts raw radar PCAP traffic into ASTERIX-based features, builds a time-series dataset, injects ENAC-style synthetic anomalies, trains TranAD and Autoencoder models, and evaluates a final Hybrid TranAD-AE ensemble.

## Pipeline

1. Extract ASTERIX/radar features from PCAP files.
2. Build train/validation/test datasets.
3. Inject ENAC-style anomalies.
4. Train TranAD on normal radar behavior.
5. Train Autoencoder on normal radar behavior.
6. Evaluate TranAD and Autoencoder separately.
7. Combine both models in a Hybrid TranAD-AE ensemble.
8. Compare results with previous ENAC baselines.

## Final Model

The final model is a Hybrid TranAD-Autoencoder ensemble:

- TranAD captures temporal radar behavior using 30-second windows.
- Autoencoder captures point-level abnormal feature combinations.
- Their normalized reconstruction-error scores are combined.
- Feature-group thresholds are used for ENAC-style attacks.

## Final Results

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Previous ENAC Autoencoder | 0.8565 | 0.6571 | 0.7116 |
| Hybrid TranAD-AE | 0.9885 | 0.9517 | 0.9650 |

## Project Structure

```text
models/
  tranad.py
  autoencoder.py
  lstm_autoencoder.py
  isolation_forest.py

scripts/
  extract_asterix.py
  build_dataset.py
  inject_enac_anomalies.py
  evaluate_ensemble.py

utils/
  dataset.py
  metrics.py
  plotting.py
  postprocessing.py

train.py
test.py
train_autoencoder.py
test_autoencoder.py
requirements.txt
