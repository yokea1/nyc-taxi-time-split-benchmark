# CV Snippets · NYC Taxi Time-Split Benchmark

## EN · Resume bullets (3–5)
**NYC Taxi Time-Split Benchmark — Drift, Cost & Calibration (Open-Source)**  
GitHub: https://github.com/yokea1/nyc-taxi-time-split-benchmark  
- Built a **1M+ row** tabular benchmark with **time-based splits** (train: 2023-01…06; valid: 07; test: 08) to study **temporal distribution shift** and **cost-sensitive** decisions.  
- Implemented **rolling backtests** (expanding origin), **probability calibration** (ECE with **95% CI**), and **feature-group ablations**; reported **3-seed mean±std** for robustness.  
- Results (test, XGBoost, 3-seed mean±std): **ROC-AUC 0.7729 ± 0.0001**, **PR-AUC 0.7623 ± 0.0001**, **Brier 0.1624 ± 0.0000**, **Min Expected Cost 47,240.67 ± 11.47** @ threshold **0.30**.  
- Stack: Python, pandas, scikit-learn, XGBoost/CatBoost, PyArrow, matplotlib, Evidently; config-driven pipeline, artifacts & figures reproducible.

## 中 · 简历要点
**NYC Taxi 时间切分基准：漂移 × 成本 × 校准（开源）**  
GitHub：同上  
- 基于 **≥100 万行**数据，以**时间划分**（训：01…06；验：07；测：08）评估**时间分布漂移**与**成本敏感**决策。  
- 实现**滚动回测**、**ECE+95%CI 校准**、**特征组消融**；以 **3 个种子**输出 **mean±std**。  
- 测试集（XGBoost，3 种子）：**ROC-AUC 0.7729 ± 0.0001**、**PR-AUC 0.7623 ± 0.0001**、**Brier 0.1624 ± 0.0000**、**最小期望成本 47,240.67 ± 11.47**（阈值 **0.30**）。

## Ultra-brief (80–120 chars)
EN: 1M+ time-split benchmark with rolling backtests, cost-sensitive metrics, calibration (ECE+CI), ablations; test ROC-AUC **0.7729** / PR-AUC **0.7623**.  
中：100万+ 时间切分基准，含滚动回测、成本敏感、校准（ECE+CI）、消融；测试 ROC-AUC **0.7729**、PR-AUC **0.7623**。

## 3-line pitch (email)
I open-sourced a 1M+ row time-split benchmark to study temporal shift and cost-aware decisions. It includes rolling backtests, calibration (ECE+CI), and ablations with 3-seed mean±std. On the 2023-08 test, XGBoost achieves **0.7729 ROC-AUC / 0.7623 PR-AUC**, minimizing expected cost at **0.30**.

## Elevator talk
- 30s（中）：做了 100万+ 的时间切分基准，关注**时间漂移**与**成本敏感**，含滚动回测、ECE+CI 校准与消融；3 种子 mean±std 抗偶然性。测试 ROC-AUC **0.7729**、PR-AUC **0.7623**，阈值 **0.30** 处期望成本最低，代码/图表开源。  
- 60s（EN）：Built and open-sourced a 1M+ time-split benchmark for temporal shift and cost-aware decisions; rolling backtests, ECE with CIs, ablations; 3-seed mean±std. Test **0.7729/0.7623** ROC/PR; min cost at **0.30**; config-driven and fully reproducible.
