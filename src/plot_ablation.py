import os, argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="reports/ablation/ablation_summary.csv")
    ap.add_argument("--out", type=str, default="reports/ablation/ablation_summary.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metrics = ["pr_auc_mean","roc_auc_mean"]
    plt.figure()
    for m in metrics:
        plt.plot(df["experiment"], df[m], marker="o", label=m)
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
