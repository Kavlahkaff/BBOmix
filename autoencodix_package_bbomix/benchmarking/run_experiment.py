import argparse
import time
import yaml
import logging
import sys
from autoencodix_runner.data import create_data_config
from autoencodix_runner.models import create_model
from autoencodix_runner.evaluation import evaluate
import pathlib
import json
import numpy as np
import pickle
# -----------------------------------------------------------------------------
# Logging setup (flushes immediately, cluster-safe)
# -----------------------------------------------------------------------------
def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(process)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)

    # Ensure line-buffered output (Python 3.7+)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    return logger


logger = setup_logging()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
import os

def load_ontology_paths(dataset, ontology_name):
    config_path = pathlib.Path(__file__).parent / "configs" / "ontologies.yaml"
    cfg = yaml.safe_load(open(config_path))
    if dataset not in cfg:
        raise ValueError(f"No ontology configuration for dataset: {dataset}")
    if ontology_name not in cfg[dataset]:
        raise ValueError(f"No ontology named {ontology_name} for dataset {dataset}")

    paths = cfg[dataset][ontology_name]["paths"]
    data_dir = os.environ.get("AUTOENCODIX_DATA_DIR", "./data")
    return [os.path.join(data_dir, paths["lvl1"]), os.path.join(data_dir, paths["lvl2"])]


def get_epochs():
    config_path = pathlib.Path(__file__).parent / "configs" / "search_space.yaml"
    cfg = yaml.safe_load(open(config_path))
    return cfg["fixed"]["epochs"]


def construct_output_path(job, base_dir=None):
    """
    Construct output directory matching batch structure:
    base_dir/architecture/dataset/modality/[ontology/]seed_X/
    
    Example paths:
    - vanillix: results/vanillix/tcga/DNA_CLIN/seed_1/
    - ontix: results/ontix/tcga/DNA_CLIN/go_biological_process/seed_2/
    """
    if base_dir is None:
        base_dir = os.environ.get("AUTOENCODIX_RESULTS_DIR", "./results")

    path_parts = [
        base_dir,
        job["architecture"],
        job["dataset"],
        "_".join(job["modalities"]),  # Join modalities with underscore
    ]
    
    # Add ontology for ontix
    if job["architecture"] == "ontix" and job.get("ontology"):
        path_parts.append(job["ontology"])
    
    # Add seed directory
    path_parts.append(f"seed_{job['seed']}")
    
    result_dir = pathlib.Path(*path_parts)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir

# Helper function to handle Numpy/Tensor types automatically
def json_numpy_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

# -----------------------------------------------------------------------------
# Main job logic
# -----------------------------------------------------------------------------
def run_job(config_path):
    logger.info("Loading config from: %s", config_path)
    #torch.set_float32_matmul_precision('high')
    with open(config_path, "r") as f:
        job = yaml.safe_load(f)

    logger.info("Starting Run ID: %s", job["run_id"])
    logger.info("Architecture: %s | Dataset: %s", job["architecture"], job["dataset"])
    logger.info("Hyperparameters: %s", job["hyperparameters"])

    # 1. Setup Data
    logger.info("Creating data configuration")
    data_config = create_data_config(job["dataset"], job["modalities"])

    # 2. Setup Ontology (if needed)
    ontology_paths = None
    if job["architecture"] == "ontix":
        if not job.get("ontology"):
            raise ValueError("Ontix architecture requires ontology defined in config.")
        logger.info("Loading ontology: %s", job["ontology"])
        ontology_paths = load_ontology_paths(job["dataset"], job["ontology"])

    # 3. Create Model
    logger.info("Creating model")
    model = create_model(
        arch=job["architecture"],
        data_config=data_config,
        hyperparams=job["hyperparameters"],
        seed=job["seed"],
        ontologies=ontology_paths,
        sep="\t",
    )
    logger.info("Model created successfully")

    # 4. Execute
    logger.info("Starting model run")
    start_time = time.perf_counter()
    model.run()
    runtime_sec = time.perf_counter() - start_time
    #result = model.visualizer._make_loss_format(model.result, data_config)
    logger.info("Model run finished")

    # 5. Evaluate
    tasks = {
        "tcga": [
            "CANCER_TYPE",
            "SUBTYPE",
            "ONCOTREE_CODE",
            "SEX",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "GRADE",
            "PATH_N_STAGE",
            "DSS_STATUS",
            "OS_STATUS",
        ],
        "schc": ["author_cell_type", "age_group", "sex"],
    }[job["dataset"]]

    logger.info("Starting evaluation")
    avg, per_task, rec, loss_per_epoch = evaluate(model, tasks, get_epochs())
    logger.info("Evaluation finished (runtime %.2f sec)", runtime_sec)

    # 6. Save Results
    results = {
        "RUN_ID": job["run_id"],
        "ARCHITECTURE": job["architecture"],
        "SEED": job["seed"],
        "DATASET": job["dataset"],
        "MODALITIES": job["modalities"],
        "ONTOLOGY": job.get("ontology", "N/A"),
        "HYPERPARAMETERS": job["hyperparameters"],
        "AVG_ML_TASK_PERFORMANCE": avg,
        "PER_TASK_PERFORMANCE": per_task,
        "VALID_RECON_LOSS": rec,
        "loss_per_epoch": loss_per_epoch,
        "RUNTIME_SECONDS": round(runtime_sec, 4),
    }

    # Construct output directory matching batch structure
    result_dir = construct_output_path(job)
    result_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    output_path = result_dir / f"{job['run_id']}_result.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, default=json_numpy_serializer)
    
    result_df_path = result_dir / f"{job['run_id']}_result_df.parquet"
    #result.to_parquet(result_df_path)
    logger.info("Finished successfully. Results saved to %s", output_path)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting run")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the job specific yaml file"
    )
    args = parser.parse_args()

    logger.info("Arguments parsed, launching job")
    run_job(args.config)
