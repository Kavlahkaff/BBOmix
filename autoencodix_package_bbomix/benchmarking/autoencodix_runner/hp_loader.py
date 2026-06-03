import pandas as pd
from autoencodix_runner.hyperparams import random_log
import random

import os

def load_results(path: str) -> pd.DataFrame:
    base_path = os.environ.get("AUTOENCODIX_HPO_RESULTS_DIR", "./data/ralf_hpo_results/ae_results_30000_runs/")
    return pd.read_parquet(os.path.join(base_path, path))


def get_top_k_configs(df: pd.DataFrame, k=3, metric="valid_recon_loss", architecture="vanillix"):
    """
    Returns a list of hyperparameter dicts for the k best runs.
    """
    best = df.nsmallest(k, metric)

    configs = []
    for _, row in best.iterrows():
        cfg = {
            "epochs": 300,
            "checkpoint_interval": 100,
            "loss_reduction": "sum",

            "k_filter": row["K_FILTER"],
            "n_layers": row["N_LAYERS"],
            "enc_factor": row["ENC_FACTOR"],
            "latent_dim": row["LATENT_DIM_FIXED"],

            "batch_size": row["BATCH_SIZE"],
            "learning_rate": row["LR_FIXED"],
            "drop_p": row["DROP_P"],

            # if not in parquet, just set a default
            "weight_decay": row.get("WEIGHT_DECAY", 0.001)
        }
        if architecture != "disentanglix":
            cfg["beta"] = row["BETA"]
        else:
            cfg["beta_mi"] = random.choice([1,2,3,4,5,6,7,8,9,10])
            cfg["beta_tc"] = random.choice([1000,2000,3000,4000,5000])
            cfg["beta_dimKL"] = random_log(0.01, 1.0)
        configs.append(cfg)

    return configs
