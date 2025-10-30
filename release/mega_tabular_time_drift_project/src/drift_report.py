import argparse, os, glob, pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True)
    ap.add_argument("--report_dir", type=str, required=True)
    ap.add_argument("--ref_month", type=str, default=None, help="reference month (e.g., 2023-06). If None, use earliest month.")
    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.processed_dir, "table_*.parquet")))
    if not files:
        raise SystemExit("No processed parquet found.")

    # map month -> df
    mapping = {}
    for fp in files:
        m = os.path.basename(fp).split("table_")[1].split(".parquet")[0]
        mapping[m] = pd.read_parquet(fp)

    months = sorted(mapping.keys())
    ref_m = args.ref_month or months[0]
    ref = mapping[ref_m]

    for m in months:
        if m == ref_m:
            continue
        cur = mapping[m]
        rep = Report(metrics=[DataQualityPreset(), DataDriftPreset(), TargetDriftPreset()])
        rep.run(reference_data=ref, current_data=cur)
        out = os.path.join(args.report_dir, f"drift_{ref_m}_vs_{m}.html")
        rep.save_html(out)
        print("Saved", out)

if __name__ == "__main__":
    main()
