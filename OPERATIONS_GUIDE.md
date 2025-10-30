# OPERATIONS GUIDE · NYC Taxi Time-Split Benchmark

(Write how to run, reproduce, smoke vs full, figures, troubleshooting, etc.)
- Env: conda create -n mega-tabular python=3.10 …
- Data: scripts/download_data.py …; src/preprocess …
- Train/Eval: src/train / src/evaluate (seeds 42/43/44) → reports/summary/metrics_mean_std.json
- Figures: calibration / rolling / ablation commands and paths
- Smoke run: processed_sample with 200k rows per month (Jan–Mar)
- Troubleshooting: conda env, xgboost install, stall fix, etc.
