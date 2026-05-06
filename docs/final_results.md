# Final Results

The final model is a Hybrid TranAD-Autoencoder ensemble.

## Final Optimized Configuration

- TranAD weight: 0.4
- Autoencoder weight: 0.6
- Smooth window: 5
- Minimum event length: 120
- Merge gap: 0
- Default threshold quantile: 0.9995
- CGS threshold quantile: 0.997
- ROUTE threshold quantile: 0.999
- THETA threshold quantile: 0.999

## Final Mean Results

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Previous ENAC Autoencoder | 0.8565 | 0.6571 | 0.7116 |
| Hybrid TranAD-AE | 0.9885 | 0.9517 | 0.9650 |

## Note

The results were obtained on a reproduced ENAC-style synthetic anomaly benchmark using injected radar attacks.
