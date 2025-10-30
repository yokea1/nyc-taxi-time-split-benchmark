import os, argparse, pandas as pd, numpy as np

def parse_hour(h):
    # Avazu 'hour' like 14102100 (YYMMDDHH)
    h = int(h)
    HH = h % 100
    DD = (h // 100) % 100
    MM = (h // 10000) % 100
    YY = (h // 1000000) % 100
    year = 2000 + YY
    # Construct a pandas datetime
    return pd.Timestamp(year=year, month=MM, day=DD, hour=HH)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw/avazu")
    ap.add_argument("--out_dir", type=str, default="data/processed_avazu")
    ap.add_argument("--min_rows", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Use a subset of train for speed; Avazu is very large
    train_path = os.path.join(args.raw_dir, "train.gz")
    if not os.path.exists(train_path):
        raise SystemExit(f"Not found: {train_path}")

    df = pd.read_csv(train_path, compression="gzip")
    if len(df) > args.min_rows:
        df = df.iloc[:args.min_rows].copy()
    df["dt"] = df["hour"].apply(parse_hour)
    df["label_click"] = df["click"].astype("int8")
    df["hour_"] = df["dt"].dt.hour.astype("int8")
    df["dow"] = df["dt"].dt.dayofweek.astype("int8")
    df["month"] = df["dt"].dt.month.astype("int8")
    df["year"] = df["dt"].dt.year.astype("int16")

    keep = ["label_click","dt","hour_","dow","month","year"] + [c for c in df.columns if c.startswith(("site_","app_","device_","C"))]
    feat = df[keep].rename(columns={"dt":"time"})
    # partition by month
    feat["_month"] = feat["time"].dt.to_period("M").astype(str)
    for m, sub in feat.groupby("_month"):
        sub.drop(columns=["_month"]).to_parquet(os.path.join(args.out_dir, f"table_{m}.parquet"), index=False)
        print("Saved", m)

if __name__ == "__main__":
    main()
