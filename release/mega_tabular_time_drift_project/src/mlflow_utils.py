import os

def try_mlflow_start(run_name: str, params: dict=None):
    try:
        import mlflow
    except Exception:
        print("[mlflow] not installed; skipping logging.")
        class Dummy:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return Dummy()
    mlflow.set_experiment("mega_tabular_experiments")
    return mlflow.start_run(run_name=run_name)

def mlflow_log_metrics(metrics: dict, step: int=None):
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass

def mlflow_log_params(params: dict):
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception:
        pass
