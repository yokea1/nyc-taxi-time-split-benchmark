import os, subprocess, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/avazu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # Avazu CTR dataset (Kaggle): 'avazu-ctr-prediction'
    cmd = ["kaggle", "competitions", "download", "-c", "avazu-ctr-prediction", "-p", args.out_dir]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print("Unzipping...")
    subprocess.check_call(["unzip", "-o", os.path.join(args.out_dir, "avazu-ctr-prediction.zip"), "-d", args.out_dir])
    print("Done.")
if __name__ == "__main__":
    main()
