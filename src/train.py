import os, json, glob, numpy as np, pandas as pd, yaml, argparse
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import dump
from .models import make_logreg, make_xgb, make_catboost

def load_config(path: str):
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

def infer_feature_types(df: pd.DataFrame, label_col: str, time_col: str):
    cols = [c for c in df.columns if c not in [label_col, time_col]]
    num_cols = [c for c in cols if str(df[c].dtype).startswith(("float","int"))]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    mcfg = cfg["models"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    # load train/valid explicitly by months
    Xtr, ytr = load_splits(dcfg["processed_dir"], dcfg["train_months"], dcfg["label_col"])
    Xva, yva = load_splits(dcfg["processed_dir"], dcfg["valid_months"], dcfg["label_col"])

    num_cols, cat_cols = infer_feature_types(Xtr, dcfg["label_col"], dcfg["time_col"])

    for seed in args.seeds:
        seed_dir = f"models/seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)

        models = {
      "xgb": make_xgb(num_cols, cat_cols, mcfg["xgboost"])
  }

        metrics = {}
        for name, mdl in models.items():
            print(f"[seed {seed}] Fitting {name}...")
            # set random_state if available
            if hasattr(mdl[-1], "random_state"):
                mdl[-1].set_params(random_state=seed)
            mdl.fit(Xtr, ytr)
            p_va = mdl.predict_proba(Xva)[:,1]

            metrics[name] = {
                "roc_auc": float(roc_auc_score(yva, p_va)),
                "pr_auc": float(average_precision_score(yva, p_va)),
                "brier": float(brier_score_loss(yva, p_va))
            }
            dump(mdl, os.path.join(seed_dir, f"{name}.joblib"))

        with open(f"reports/metrics/valid_seed_{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[seed {seed}] Done. Metrics saved.")

if __name__ == "__main__":
    main()
