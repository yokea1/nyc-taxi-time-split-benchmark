import argparse, os, json, yaml, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import dump
from .train import load_splits, infer_feature_types, load_config
from .models import make_logreg, make_xgb, make_catboost

def select_columns(df, groups, keep_groups=None, drop_groups=None):
    cols = df.columns.tolist()
    if keep_groups:
        keep = set()
        for g in keep_groups:
            for key in groups[g]:
                keep.update([c for c in cols if key in c])
        return list(keep)
    elif drop_groups:
        drop = set()
        for g in drop_groups:
            for key in groups[g]:
                drop.update([c for c in cols if key in c])
        return [c for c in cols if c not in drop]
    else:
        return cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="main config path")
    ap.add_argument("--ablation", type=str, required=True, help="ablation config path")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    ap.add_argument("--model", type=str, default="xgb", choices=["logreg","xgb","cat"])
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    with open(args.ablation, "r") as f:
        ablcfg = yaml.safe_load(f)
    groups = ablcfg["feature_groups"]

    dcfg, mcfg = cfg["data"], cfg["models"]
    Xtr_full, ytr = load_splits(dcfg["processed_dir"], dcfg["train_months"], dcfg["label_col"])
    Xva_full, yva = load_splits(dcfg["processed_dir"], dcfg["valid_months"], dcfg["label_col"])

    results = []
    for exp in ablcfg["experiments"]:
        keep_groups = exp.get("keep_groups")
        drop_groups = exp.get("drop_groups")

        tr_cols = select_columns(Xtr_full, groups, keep_groups, drop_groups)
        va_cols = [c for c in tr_cols if c in Xva_full.columns]

        Xtr = Xtr_full[tr_cols]
        Xva = Xva_full[va_cols]

        # infer numeric/categorical
        num_cols, cat_cols = [], []
        for c in tr_cols:
            if c in [dcfg["label_col"], dcfg["time_col"]]: continue
            if str(Xtr[c].dtype).startswith(("float","int")):
                num_cols.append(c)
            else:
                cat_cols.append(c)

        if args.model == "logreg":
            factory = lambda: make_logreg(num_cols, cat_cols, C=cfg["models"]["logistic_regression"]["C"], max_iter=cfg["models"]["logistic_regression"]["max_iter"])
        elif args.model == "xgb":
            factory = lambda: make_xgb(num_cols, cat_cols, mcfg["xgboost"])
        else:
            factory = lambda: make_catboost(num_cols, cat_cols, mcfg["catboost"])

        seed_metrics = []
        for seed in args.seeds:
            mdl = factory()
            if hasattr(mdl[-1], "random_state"):
                mdl[-1].set_params(random_state=seed)
            mdl.fit(Xtr, ytr)
            p_va = mdl.predict_proba(Xva)[:,1]
            m = {
                "roc_auc": float(roc_auc_score(yva, p_va)),
                "pr_auc": float(average_precision_score(yva, p_va)),
                "brier": float(brier_score_loss(yva, p_va))
            }
            seed_metrics.append(m)
        # mean±std
        agg = {k: {"mean": float(np.mean([d[k] for d in seed_metrics])), "std": float(np.std([d[k] for d in seed_metrics]))} for k in seed_metrics[0].keys()}
        results.append({"experiment": exp["name"], **{f"{k}_mean": v["mean"] for k,v in agg.items()}, **{f"{k}_std": v["std"] for k,v in agg.items()}})

    os.makedirs("reports/ablation", exist_ok=True)
    out_csv = "reports/ablation/ablation_summary.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved ablation summary -> {out_csv}")

if __name__ == "__main__":
    main()
