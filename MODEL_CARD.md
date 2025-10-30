# Model Card · NYC Taxi Time-Split Benchmark (XGBoost)

## Overview
Binary classification on NYC Yellow Taxi (2023-01…08) with **time-based splits** to study **temporal drift** and **cost-sensitive** decisions.

## Data
- Source: NYC TLC public trip records (2023-01…08).
- Splits: Train 01…06, Valid 07, Test 08.
- Volume: ~25,224,937 rows (full); sampled smoke-set also supported.

## Intended Use
- Research & benchmarking of tabular models under temporal shift.
- Educational use for reproducible pipelines and evaluation.

## Not Intended Use
- Direct business deployment without domain validation.
- Decisions affecting individuals without fairness review.

## Metrics
- ROC-AUC, PR-AUC, Brier, Expected-Cost (validation threshold search).
- Calibration: ECE with 95% CI.

## Model & Pipeline
- XGBoost classifier in sklearn pipeline with imputation + OHE.
- Rolling backtests (expanding origin), calibration, feature-group ablations.
- 3 seeds with mean±std reporting.

## Risks & Limitations
- Temporal shift and covariate drift can degrade performance in later months.
- Label noise/selection bias in public data.
- Thresholds and costs are scenario-dependent; re-fit/recalibrate per period.

## Reproducibility
- Config-driven (`configs/*.yaml`), artifacts in `reports/**`.
- Release: v1.0.0 (see GitHub Releases).
