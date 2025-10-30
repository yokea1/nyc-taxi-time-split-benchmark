import os, json, glob, argparse, numpy as np, pandas as pd, yaml
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import load

def load_config(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_splits(processed_dir, months, label_col):
    frames = []
    for m in months:
        fp = os.path.join(processed_dir, f"table_{m}.parquet")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        frames.append(pd.read_parquet(fp))
    df = pd.concat(frames, ignore_index=True)
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int).values
    return X, y

def expected_cost(y_true, y_prob, threshold, c_fp, c_fn, c_tp=0.0, c_tn=0.0):
    y_pred = (y_prob >= threshold).astype(int)
    fp = ((y_pred==1) & (y_true==0)).sum()
    fn = ((y_pred==0) & (y_true==1)).sum()
    tp = ((y_pred==1) & (y_true==1)).sum()
    tn = ((y_pred==0) & (y_true==0)).sum()
    return c_fp*fp + c_fn*fn + c_tp*tp + c_tn*tn

def summarize_mean_std(dicts):
    # dicts: list of {model: {metric: value}}
    models = sorted({m for d in dicts for m in d.keys()})
    out = {}
    for m in models:
        metrics = {}
        all_metrics = sorted({k for d in dicts if m in d for k in d[m].keys()})
        for k in all_metrics:
            vals = [d[m][k] for d in dicts if m in d and k in d[m]]
            metrics[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        out[m] = metrics
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    dcfg, mcfg, ccfg = cfg["data"], cfg["models"], cfg["costs"]

    # Load test split
    Xte, yte = load_splits(dcfg["processed_dir"], dcfg["test_months"], dcfg["label_col"])

    results = []
    for seed in args.seeds:
        seed_dir = f"models/seed_{seed}"
        metrics = {}
        for name in ["xgb"]:
            model_path = os.path.join(seed_dir, f"{name}.joblib")
            mdl = load(model_path)
            p = mdl.predict_proba(Xte)[:,1]

            # base metrics
            roc = roc_auc_score(yte, p)
            pr = average_precision_score(yte, p)
            brier = brier_score_loss(yte, p)

            # cost-sensitive sweep
            costs = []
            for th in ccfg["thresholds"]:
                ec = expected_cost(yte, p, th, ccfg["c_fp"], ccfg["c_fn"], ccfg["c_tp"], ccfg["c_tn"])
                costs.append({"threshold": th, "expected_cost": float(ec)})
            best = min(costs, key=lambda x: x["expected_cost"])

            metrics[name] = {
                "roc_auc": float(roc),
                "pr_auc": float(pr),
                "brier": float(brier),
                "best_threshold": best["threshold"],
                "min_expected_cost": best["expected_cost"]
            }
            # save per-model threshold table
            os.makedirs("reports/metrics", exist_ok=True)
            with open(f"reports/metrics/test_thresholds_seed_{seed}_{name}.json", "w") as f:
                json.dump(costs, f, indent=2)

        with open(f"reports/metrics/test_seed_{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        results.append(metrics)

    # mean±std
    agg = summarize_mean_std(results)
    os.makedirs("reports/summary", exist_ok=True)
    with open("reports/summary/metrics_mean_std.json", "w") as f:
        json.dump(agg, f, indent=2)

    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
