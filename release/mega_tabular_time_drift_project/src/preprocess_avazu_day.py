import os, argparse, pandas as pd, numpy as np

def parse_hour(h):
    h = int(h)
    HH = h % 100
    DD = (h // 100) % 100
    MM = (h // 10000) % 100
    YY = (h // 1000000) % 100
    year = 2000 + YY
    return pd.Timestamp(year=year, month=MM, day=DD, hour=HH)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw/avazu")
    ap.add_argument("--out_dir", type=str, default="data/processed_avazu_day")
    ap.add_argument("--min_rows", type=int, default=2_000_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.raw_dir, "train.gz")
    if not os.path.exists(train_path):
        raise SystemExit(f"Not found: {train_path}")

    df = pd.read_csv(train_path, compression="gzip")
    if len(df) > args.min_rows:
        df = df.iloc[:args.min_rows].copy()

    df["time"] = df["hour"].apply(parse_hour)
    df["date_str"] = df["time"].dt.date.astype(str)
    df["label_click"] = df["click"].astype("int8")
    df["hour_"] = df["time"].dt.hour.astype("int8")
    df["dow"] = df["time"].dt.dayofweek.astype("int8")
    df["month"] = df["time"].dt.month.astype("int8")
    df["year"] = df["time"].dt.year.astype("int16")

    feat_cols = ["label_click","time","hour_","dow","month","year"] + \
                [c for c in df.columns if c.startswith(("site_","app_","device_","C"))]
    feat = df[feat_cols]

    # Partition by "day" strings that our trainer will still accept as table_<id>.parquet
    for day, sub in feat.groupby(df["time"].dt.date.astype(str)):
        out = os.path.join(args.out_dir, f"table_{day}.parquet")
        sub.to_parquet(out, index=False)
        print("Saved", out)

if __name__ == "__main__":
    main()
