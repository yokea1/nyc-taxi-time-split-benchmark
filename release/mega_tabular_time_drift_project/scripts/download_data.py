import argparse, os, sys, subprocess, pathlib

NYC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

def month_url(year: int, month: int):
    return f"{NYC_BASE}/yellow_tripdata_{year}-{month:02d}.parquet"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--months", type=int, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, default="data/raw")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for m in args.months:
        url = month_url(args.year, m)
        out = os.path.join(args.out_dir, f"yellow_tripdata_{args.year}-{m:02d}.parquet")
        print(f"Downloading {url} -> {out}")
        # Use curl or wget if available, else try python requests if necessary.
        try:
            subprocess.check_call(["curl", "-L", "-o", out, url])
        except Exception:
            try:
                subprocess.check_call(["wget", "-O", out, url])
            except Exception as e:
                print("Failed to download with curl/wget, please download manually:", url)
                raise e

if __name__ == "__main__":
    main()
