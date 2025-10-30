import os, argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="reports/rolling/metrics.csv")
    ap.add_argument("--out", type=str, default="reports/rolling/rolling_metrics.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plt.figure()
    plt.plot(df["test_month"], df["pr_auc"], marker="o", label="PR-AUC")
    plt.plot(df["test_month"], df["expected_cost"], marker="o", label="Expected Cost")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
