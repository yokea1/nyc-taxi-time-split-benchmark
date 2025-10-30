import argparse, os, pandas as pd, numpy as np
from pathlib import Path

def compute_tip_rate(df: pd.DataFrame) -> pd.Series:
    charges = (df.get("fare_amount", 0) + df.get("tolls_amount", 0) + 
               df.get("mta_tax", 0) + df.get("improvement_surcharge", 0))
    tip_rate = df.get("tip_amount", 0) / charges.replace(0, np.nan)
    tip_rate = tip_rate.clip(lower=0, upper=1).fillna(0.0)
    return tip_rate

def engineer_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[time_col])
    df["hour"] = dt.dt.hour.astype("int16")
    df["dow"] = dt.dt.dayofweek.astype("int8")
    df["month"] = dt.dt.month.astype("int8")
    df["year"] = dt.dt.year.astype("int16")

    # Basic sanitization
    for col in ["trip_distance","passenger_count","PULocationID","DOLocationID","payment_type","VendorID","RatecodeID"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
        else:
            df[col] = 0.0

    # Label
    tip_rate = compute_tip_rate(df)
    df["label_tip20"] = (tip_rate >= 0.20).astype("int8")

    # ID
    if "VendorID" in df.columns:
        df["ride_id"] = (dt.view("int64") // 10**9).astype("int64") * 10 + df["VendorID"].astype("int64")
    else:
        df["ride_id"] = (dt.view("int64") // 10**9).astype("int64")

    keep = ["ride_id","label_tip20", time_col, "hour","dow","month","year",
            "trip_distance","passenger_count","PULocationID","DOLocationID","payment_type","VendorID","RatecodeID"]
    return df[keep]

def partition_by_month(df: pd.DataFrame, time_col: str, out_dir: str):
    df["_month"] = pd.to_datetime(df[time_col]).dt.to_period("M").astype(str)
    for m, sub in df.groupby("_month"):
        out_path = os.path.join(out_dir, f"table_{m}.parquet")
        sub.drop(columns=["_month"]).to_parquet(out_path, index=False)
        print("Saved", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--time_col", type=str, required=True)
    ap.add_argument("--min_rows", type=int, default=1_000_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Concatenate raw monthly parquet files
    files = sorted([str(p) for p in Path(args.raw_dir).glob("*.parquet")])
    if not files:
        raise SystemExit(f"No parquet files found in {args.raw_dir}")

    dfs = []
    total = 0
    for fp in files:
        print("Reading", fp)
        df = pd.read_parquet(fp, engine="pyarrow")
        total += len(df)
        dfs.append(df)
        if total >= args.min_rows:
            break

    full = pd.concat(dfs, ignore_index=True)
    print("Loaded rows:", len(full))

    # Feature engineering
    ft = engineer_features(full, args.time_col)

    # Partition by month for time splits
    partition_by_month(ft, args.time_col, args.out_dir)

if __name__ == "__main__":
    main()
