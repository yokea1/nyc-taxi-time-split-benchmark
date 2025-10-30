import argparse, os, json, numpy as np, pandas as pd, yaml
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

def ece_score(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds==b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
        ece += np.abs(acc - conf) * w
    return float(ece)

def bootstrap_ci(y, p, func, n_boot=200, alpha=0.05, **kwargs):
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        vals.append(func(y[idx], p[idx], **kwargs))
    vals = np.sort(vals)
    lo = vals[int((alpha/2)*n_boot)]
    hi = vals[int((1-alpha/2)*n_boot)-1]
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default="xgb", choices=["logreg","xgb","cat"])
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--n_boot", type=int, default=200)
    args = ap.parse_args()

    with open(args.cfg,"r") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    # load test set & model
    from .evaluate import load_splits
    Xte, yte = load_splits(dcfg["processed_dir"], dcfg["test_months"], dcfg["label_col"])
    model_path = f"models/seed_{args.seed}/{args.model}.joblib"
    mdl = load(model_path)
    prob = mdl.predict_proba(Xte)[:,1]

    # reliability diagram data
    frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=args.n_bins, strategy="uniform")

    # ECE + CI
    ece = ece_score(yte, prob, n_bins=args.n_bins)
    lo, hi = bootstrap_ci(yte, prob, lambda y,p: ece_score(y,p,n_bins=args.n_bins), n_boot=args.n_boot)

    os.makedirs("reports/calibration", exist_ok=True)
    out = {
        "n_bins": args.n_bins,
        "bins_frac_pos": frac_pos.tolist(),
        "bins_mean_pred": mean_pred.tolist(),
        "ece": ece,
        "ece_ci": [lo, hi]
    }
    with open("reports/calibration/calibration_seed{}_{}.json".format(args.seed, args.model), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
