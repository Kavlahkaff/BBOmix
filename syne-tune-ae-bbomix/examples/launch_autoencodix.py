"""
Run HPO methods on AutoEncodix blackboxes created by autoencodix_import.py.
"""
import logging

from syne_tune.blackbox_repository import load_blackbox, BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import (
    RandomSearch,
    TPE,
    REA,
    BORE,
    BOTorch,
    CQR,
    ASHA,
    BOHB,
)
from syne_tune import StoppingCriterion, Tuner


METRIC_DOWNSTREAM_PERFORMANCE = "valid_recon_loss"
METRIC_ELAPSED_TIME = "runtime_seconds"
MAX_RESOURCE_ATTR = "epoch"
N_WORKERS = 4
MAX_TRIALS = 100
SEEDS = range(0, 30)

BLACKBOXES: dict[str, list[str]] = {
    "autoencodix_vanillix": [
        "schc_RNA_METH_CLIN",
        "schc_METH_CLIN",
        "schc_RNA_CLIN" "tcga_RNA_CLIN",
        "tcga_METH_CLIN",
        "tcga_DNA_CLIN",
        "tcga_RNA_DNA_METH_CLIN",
    ],
    # "autoencodix_varix": [
    #    "schc_RNA_METH_CLIN",
    #    "schc_METH_CLIN",
    # ],
    "autoencodix_ontix": [
        "schc_RNA_METH_CLIN_chromosome",
        "schc_METH_CLIN_chromosome",
        "schc_RNA_CLIN_chromosome" "tcga_RNA_CLIN_chromosome",
        "tcga_METH_CLIN_chromosome",
        "tcga_DNA_CLIN_chromosome",
        "tcga_RNA_DNA_METH_CLIN_chromosome",
        "schc_RNA_METH_CLIN_reactome",
        "schc_METH_CLIN_reactome",
        "schc_RNA_CLIN_reactome" "tcga_RNA_CLIN_reactome",
        "tcga_METH_CLIN_reactome",
        "tcga_DNA_CLIN_reactome",
        "tcga_RNA_DNA_METH_CLIN_reactome",
    ],
    # "autoencodix_disentanglix": [
    #    "schc_RNA_METH_CLIN",
    #    "schc_METH_CLIN",
    # ],
}

METHODS = {
    "ASHA": ASHA,
    "RS": RandomSearch,
    "TPE": TPE,
    "REA": REA,
    "BORE": BORE,
    "BOHB": BOHB,
    "BOTorch": BOTorch,
    "CQR": CQR,
}


def simulate_benchmark(
    blackbox,
    trial_backend,
    seed: int,
    method_name: str,
    benchmark_name: str,
) -> None:
    method_cls = METHODS[method_name]

    config_space = blackbox.configuration_space_with_max_resource_attr(
        MAX_RESOURCE_ATTR
    )

    if method_name == "RS":
        scheduler = method_cls(
            config_space=config_space,
            metrics=[METRIC_DOWNSTREAM_PERFORMANCE],
            random_seed=seed,
            do_minimize=True,
        )
    elif method_name == "ASHA" or method_name == "BOHB":
        scheduler = method_cls(
            config_space=config_space,
            metric=METRIC_DOWNSTREAM_PERFORMANCE,
            time_attr="epoch",
            max_t=300,
            random_seed=seed,
            do_minimize=True,
        )
    else:
        scheduler = method_cls(
            config_space=config_space,
            metric=METRIC_DOWNSTREAM_PERFORMANCE,
            random_seed=seed,
            do_minimize=True,
        )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_num_trials_finished=MAX_TRIALS),
        n_workers=N_WORKERS,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        tuner_name=f"results/{method_name}-{seed}-{benchmark_name}".replace("_", "-"),
        save_tuner=False,
        suffix_tuner_name=False,
        metadata={
            "seed": seed,
            "algorithm": method_name,
            "benchmark": benchmark_name,
        },
    )
    tuner.run()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    for blackbox_name, task_names in BLACKBOXES.items():
        bb_dict = load_blackbox(blackbox_name, local_files_only=True)

        for task_name in task_names:
            if task_name not in bb_dict:
                logging.warning(
                    f"Task '{task_name}' not found in '{blackbox_name}'. "
                    f"Available: {sorted(bb_dict)}"
                )
                continue

            blackbox = bb_dict[task_name]
            benchmark_name = f"{blackbox_name}_{task_name}"

            for method_name in METHODS:
                for seed in SEEDS:
                    logging.info(
                        f"Running {method_name} | seed={seed} | {benchmark_name}"
                    )
                    trial_backend = BlackboxRepositoryBackend(
                        blackbox_name=blackbox_name,
                        dataset=task_name,
                        elapsed_time_attr=METRIC_ELAPSED_TIME,  # "runtime_seconds"
                        surrogate="KNeighborsRegressor",
                    )
                    simulate_benchmark(
                        blackbox=blackbox,
                        trial_backend=trial_backend,
                        seed=seed,
                        method_name=method_name,
                        benchmark_name=benchmark_name,
                    )
