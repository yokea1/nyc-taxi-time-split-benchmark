import argparse, os, json, pandas as pd, numpy as np, yaml
from joblib import dump
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from .train import load_splits, infer_feature_types
from .models import make_logreg, make_xgb, make_catboost

def month_sort(months):
    return sorted(months)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--model", type=str, default="xgb", choices=["logreg","xgb","cat"])
    ap.add_argument("--min_train_months", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.cfg,"r") as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg, ccfg = cfg["data"], cfg["models"], cfg["costs"]

    # available months from processed/
    months = []
    for fn in os.listdir(dcfg["processed_dir"]):
        if fn.startswith("table_") and fn.endswith(".parquet"):
            months.append(fn.split("table_")[1].split(".parquet")[0])
    months = month_sort(months)

    # choose model factory
    if args.model == "logreg":
        factory = lambda num, cat: make_logreg(num, cat, C=mcfg["logistic_regression"]["C"], max_iter=mcfg["logistic_regression"]["max_iter"])
    elif args.model == "xgb":
        factory = lambda num, cat: make_xgb(num, cat, mcfg["xgboost"])
    else:
        factory = lambda num, cat: make_catboost(num, cat, mcfg["catboost"])

    rows = []
    for i in range(args.min_train_months, len(months)):
        train_months = months[:i]      # up to i-1
        test_month = months[i]         # month i
        # use last month of train as valid for threshold selection
        valid_month = train_months[-1]
        train_months_wo_valid = train_months[:-1] if len(train_months) > 1 else train_months

        Xtr, ytr = load_splits(dcfg["processed_dir"], train_months_wo_valid, dcfg["label_col"])
        Xva, yva = load_splits(dcfg["processed_dir"], [valid_month], dcfg["label_col"])
        Xte, yte = load_splits(dcfg["processed_dir"], [test_month], dcfg["label_col"])

        num_cols, cat_cols = infer_feature_types(Xtr, dcfg["label_col"], dcfg["time_col"])
        mdl = factory(num_cols, cat_cols)
        if hasattr(mdl[-1], "random_state"):
            mdl[-1].set_params(random_state=args.seed)
        mdl.fit(Xtr, ytr)

        p_va = mdl.predict_proba(Xva)[:,1]
        p_te = mdl.predict_proba(Xte)[:,1]

        # select threshold on valid by expected cost
        best_th, min_cost = None, float("inf")
        for th in cfg["costs"]["thresholds"]:
            ypred = (p_va >= th).astype(int)
            fp = ((ypred==1) & (yva==0)).sum()
            fn = ((ypred==0) & (yva==1)).sum()
            cost = cfg["costs"]["c_fp"]*fp + cfg["costs"]["c_fn"]*fn + cfg["costs"]["c_tp"]*0 + cfg["costs"]["c_tn"]*0
            if cost < min_cost:
                min_cost, best_th = cost, th

        # metrics on test
        roc = roc_auc_score(yte, p_te)
        pr = average_precision_score(yte, p_te)
        brier = brier_score_loss(yte, p_te)
        ypred_te = (p_te >= best_th).astype(int)
        fp = int(((ypred_te==1) & (yte==0)).sum())
        fn = int(((ypred_te==0) & (yte==1)).sum())
        exp_cost = cfg["costs"]["c_fp"]*fp + cfg["costs"]["c_fn"]*fn

        rows.append({"test_month": test_month, "valid_month": valid_month, "roc_auc": float(roc), "pr_auc": float(pr),
                     "brier": float(brier), "best_th": float(best_th), "expected_cost": float(exp_cost),
                     "n_test": int(len(yte))})

    os.makedirs("reports/rolling", exist_ok=True)
    out_csv = "reports/rolling/metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved rolling metrics -> {out_csv}")

if __name__ == "__main__":
    main()
