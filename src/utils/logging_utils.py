import yaml
import pandas as pd

from pathlib import Path
from datetime import datetime


def create_run_dir(base, algo, config=None):
    run_dir = Path(base) / algo
    run_dir.mkdir(parents=True, exist_ok=True)

    if config is not None:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)

    return run_dir

def create_results_df(results_dict):
    return (
        pd.DataFrame
        .from_dict(results_dict, orient='index')
        .reset_index()
        .rename(columns={'index':'timestep'})
    )

def log_results_df(run_dir, results_df):
    results_df.to_csv(f"{run_dir}/results.csv")

def log_video(run_dir, viewer_class_object,id=None):
    viewer_class_object.save(f"{run_dir}/result_video{id}.mp4")