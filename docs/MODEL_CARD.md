# Model Card — Tip20 Classifier (NYC TLC)

## Overview
- **Task**: Binary classification of high tip rate (≥20%) for NYC yellow taxi trips.
- **Intended Use**: Exploration of time-split tabular ML with drift monitoring and cost-sensitive thresholds.
- **Not for**: Individual-level financial decisions; fairness-sensitive deployment.

## Data
- **Source**: NYC TLC Yellow Taxi Trip Records (public).
- **Time Splits**: Train Jan–Jun 2023, Valid Jul 2023, Test Aug 2023 (configurable).
- **Known Limitations**: Missing or noisy fares/tolls; stationarity violations across months; potential holiday/season effects.

## Metrics
- ROC-AUC, PR-AUC, Brier.
- **Cost-sensitive** expected loss with threshold selection on validation.
- **Mean±std** across multiple seeds.

## Drift & Monitoring
- Evidently reports: data quality, covariate drift, target drift (monthly comparisons).
- Rolling backtests to visualize stability over time.

## Ethics & Fairness
- Contains pickup/dropoff location proxies; use caution to avoid discriminatory uses.
- Include calibration, ECE, and uncertainty bands before any decision thresholds.

## Maintenance
- Re-train monthly with last K months as rolling window.
- Add alerting when drift magnitude or cost exceeds control limits.
