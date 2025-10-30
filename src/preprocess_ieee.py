import os, argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw/ieee")
    ap.add_argument("--out_dir", type=str, default="data/processed_ieee")
    ap.add_argument("--min_rows", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # The dataset has TransactionDT as a relative time in seconds
    train_trx = os.path.join(args.raw_dir, "train_transaction.csv")
    train_id = os.path.join(args.raw_dir, "train_identity.csv")
    if not (os.path.exists(train_trx) and os.path.exists(train_id)):
        raise SystemExit("IEEE data not found; ensure you've unzipped.")

    trx = pd.read_csv(train_trx)
    ide = pd.read_csv(train_id)
    df = trx.merge(ide, on="TransactionID", how="left")
    if len(df) > args.min_rows:
        df = df.iloc[:args.min_rows].copy()

    # create a pseudo-absolute datetime: choose an origin and add seconds
    origin = pd.Timestamp("2018-01-01")
    df["time"] = origin + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["label_fraud"] = df["isFraud"].astype("int8")
    df["hour"] = df["time"].dt.hour.astype("int8")
    df["dow"] = df["time"].dt.dayofweek.astype("int8")
    df["month"] = df["time"].dt.month.astype("int8")
    df["year"] = df["time"].dt.year.astype("int16")

    # pick a subset of numeric/cat features for demo
    sel = ["label_fraud","time","hour","dow","month","year","TransactionAmt","card1","card2","addr1","addr2"]
    for c in ["TransactionAmt","card1","card2","addr1","addr2"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    feat = df[sel]
    feat["_month"] = feat["time"].dt.to_period("M").astype(str)
    for m, sub in feat.groupby("_month"):
        sub.drop(columns=["_month"]).to_parquet(os.path.join(args.out_dir, f"table_{m}.parquet"), index=False)
        print("Saved", m)

if __name__ == "__main__":
    main()
