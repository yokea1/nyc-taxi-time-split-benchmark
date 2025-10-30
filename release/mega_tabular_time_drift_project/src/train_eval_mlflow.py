import argparse, yaml, json
from .train import main as train_main
from .evaluate import main as eval_main
from .mlflow_utils import try_mlflow_start, mlflow_log_params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    args = ap.parse_args()

    with open(args.cfg,"r") as f:
        cfg = yaml.safe_load(f)
    with try_mlflow_start(run_name="train+eval") as run:
        mlflow_log_params({"seeds": str(args.seeds)})
        # call original scripts
        import sys
        sys.argv = ["train", "--cfg", args.cfg, "--seeds"] + list(map(str,args.seeds))
        train_main()
        sys.argv = ["evaluate", "--cfg", args.cfg, "--seeds"] + list(map(str,args.seeds))
        eval_main()

if __name__ == "__main__":
    main()
