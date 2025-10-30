# RESULTS SUMMARY

Generated at: **2025-10-30 09:38:57**

## Metrics (XGBoost, Full Run, 3 Seeds)
- ROC-AUC: **0.7729 ± 0.0001**
- PR-AUC: **0.7623 ± 0.0001**
- Brier: **0.1624 ± 0.0000**
- Min Expected Cost: **47240.67 ± 11.47**
- Best Threshold (mean): **0.30**

## Calibration
- ECE: **0.0037** (95% CI **[0.0026, 0.0056]**)
- Figure: `reports/calibration/calibration_seed42_xgb.png`

## Rolling Backtest
- CSV: `reports/rolling/metrics.csv`
- Figure: `reports/rolling/rolling_metrics.png`

## Ablation
- CSV: `reports/ablation/ablation_summary.csv`
- Figure: `reports/ablation/ablation_summary.png`

## Artifacts
- Models: `models/seed_42/xgb.joblib`, `models/seed_43/xgb.joblib`, `models/seed_44/xgb.joblib`
- Summary JSON: `reports/summary/metrics_mean_std.json`

*Project root:* `/Users/jiaoyanguopei/Downloads/mega_tabular_time_drift_project_routeB_tools`
