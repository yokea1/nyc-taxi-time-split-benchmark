import os, json, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    x = data["bins_mean_pred"]
    y = data["bins_frac_pos"]
    ece = data["ece"]
    lo, hi = data["ece_ci"]

    if args.out is None:
        base = os.path.basename(args.json).replace(".json",".png")
        args.out = os.path.join("reports/calibration", base)

    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(x, y, marker="o")
    plt.title(f"Reliability Diagram\nECE={ece:.4f} (95% CI [{lo:.4f},{hi:.4f}])")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
