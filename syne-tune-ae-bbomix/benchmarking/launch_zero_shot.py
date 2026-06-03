from __future__ import annotations
import argparse
import logging
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import BlackboxRepositoryBackend, load_blackbox
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.schedulers.transfer_learning.zero_shot import ZeroShotTransfer
from utils import load_transfer_learning_evaluations, METRIC_MODE, METRIC_DOWNSTREAM, METRIC_RECON_LOSS, METRIC_ELAPSED_TIME
import numpy as np
logger = logging.getLogger(__name__)


def make_tuner_name(
    architecture: str,
    test_task: str,
    metric: str,
    use_surrogates: bool,
    all_datasets: bool,
    cross_arch_only: bool,
    extra_architectures: list[str] | None,
    seed: int,
) -> str:
    # --- metric abbreviation ---
    metric_short = {
        METRIC_DOWNSTREAM:   "ds",
        METRIC_RECON_LOSS:   "rl",
        METRIC_ELAPSED_TIME: "et",
    }.get(metric, metric[:4])

    # --- extra-architectures suffix (sorted for stability) ---
    extra = "+".join(sorted(extra_architectures)) if extra_architectures else "none"

    # --- boolean flags as compact bits ---
    flags = "".join([
        "S" if use_surrogates  else "s",   # Surrogate on/off
        "A" if all_datasets    else "a",   # All-datasets on/off
        "X" if cross_arch_only else "x",   # Cross-arch-only on/off
    ])

    tuner_name = (
        f"ZeroShot"
        f"-{architecture}"          # target architecture
        f"-{test_task}"             # held-out task
        f"-{metric_short}"               # optimisation metric
        f"-{flags}"                      # three boolean switches
        f"-x{extra}"                     # extra source architectures
        f"-r{seed}"                 # random seed
    )
    return tuner_name


def run(
    architecture: str,
    test_task: str,
    metric: str,
    do_minimize: bool,
    use_surrogates: bool,
    max_wallclock_time: float,
    n_workers: int,
    random_seed: int,
    same_dataset_only: bool = True,
    extra_architectures: list[str] | None = None,
    cross_arch_only: bool = False,
) -> None:
    np.random.seed(random_seed)
    test_dataset = test_task.split("_")[0].lower()
    blackbox_name = f"autoencodix_{architecture}_{test_dataset}"

    bb_dict = load_blackbox(blackbox_name, local_files_only=True)
    available_tasks = sorted(bb_dict)
    logger.info("Blackbox '%s' — tasks: %s", blackbox_name, available_tasks)

    if test_task not in available_tasks:
        raise ValueError(
            f"test_task='{test_task}' not found. Available: {available_tasks}"
        )

    all_datasets_list = ["tcga", "schc"] if not same_dataset_only else [test_dataset]
    extra_bb_dicts: dict[str, dict[str, BlackboxTabular]] = {}
    
    if not same_dataset_only:
        other_ds = "tcga" if test_dataset == "schc" else "schc"
        arch_label = f"{architecture}_{other_ds}"
        arch_bb_name = f"autoencodix_{architecture}_{other_ds}"
        logger.info("Loading extra-dataset blackbox '%s' …", arch_bb_name)
        extra_bb_dicts[arch_label] = load_blackbox(arch_bb_name, local_files_only=True)

    for arch in (extra_architectures or []):
        if arch == architecture:
            logger.warning(
                "--extra-architecture '%s' is the same as the target; skipping.", arch
            )
            continue
        for ds in all_datasets_list:
            arch_label = f"{arch}_{ds}"
            arch_bb_name = f"autoencodix_{arch}_{ds}"
            logger.info("Loading extra-architecture blackbox '%s' …", arch_bb_name)
            extra_bb_dicts[arch_label] = load_blackbox(arch_bb_name, local_files_only=True)


    transfer_learning_evaluations = load_transfer_learning_evaluations(
        blackbox_name=blackbox_name,
        test_task=test_task,
        metric=metric,
        bb_dict=bb_dict,
        same_dataset_only=same_dataset_only,
        extra_bb_dicts=extra_bb_dicts if extra_bb_dicts else None,
        cross_arch_only=cross_arch_only,
    )

    scheduler = ZeroShotTransfer(
        config_space=bb_dict[test_task].configuration_space,
        metric=metric,
        do_minimize=do_minimize,
        transfer_learning_evaluations=transfer_learning_evaluations,
        sort_transfer_learning_evaluations=not use_surrogates,
        use_surrogates=use_surrogates,
        random_seed=random_seed,
    )

    stop_criterion = StoppingCriterion(max_num_trials_finished=100)

    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=METRIC_ELAPSED_TIME,
        dataset=test_task,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )
    name = make_tuner_name(
        architecture=architecture,
        test_task=test_task,
        metric=metric,
        use_surrogates=use_surrogates,
        all_datasets=not same_dataset_only,
        cross_arch_only=cross_arch_only,
        extra_architectures=extra_architectures,
        seed=random_seed,
    )
    tuner = Tuner(
        tuner_name=name,
        suffix_tuner_name=False,
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        metadata={
            "seed": random_seed,
            "algorithm": "zero_shot",
            "benchmark": architecture + "_" + test_task,
        },
    )

    tuner.run()

    #print(f"\n{'='*60}")
    #print("Overall best result (zero-shot transfer)")
    #print(f"Architecture    : {architecture}")
    #print(f"Test task       : {test_task}")
    #print(f"Metric          : {metric}  ({'min' if do_minimize else 'max'})")
    #print(f"Use surrogates  : {use_surrogates}")
    #if extra_architectures:
    #    print(f"Extra sources   : {extra_architectures}")
    #print(f"{'='*60}")

    #best_experiment = load_experiment(tuner.name)
    #best_config     = best_experiment.best_config()
    #target_cs       = bb_dict[test_task].configuration_space
    #print("\n".join(
    #    f"  {k}: {v}" for k, v in best_config.items()
    #    if k in target_cs
    #))

    #best_experiment.plot()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ZeroShotTransfer tuner for autoencodix blackboxes."
    )
    p.add_argument(
        "--architecture",
        default="varix",
        choices=["vanillix", "varix", "ontix", "disentanglix"],
    )
    p.add_argument(
        "--test-task",
        default=None,
        help="Task held out as the optimisation target (default: first task).",
    )
    p.add_argument(
        "--metric",
        default=METRIC_DOWNSTREAM,
        choices=[METRIC_DOWNSTREAM, METRIC_RECON_LOSS, METRIC_ELAPSED_TIME],
    )
    p.add_argument(
        "--use-surrogates",
        action="store_true",
        default=False,
        help=(
            "Fit an XGBRegressor per source task and impute a shared random "
            "candidate grid.  Required if HP grids differ across tasks; "
            "optional otherwise."
        ),
    )
    p.add_argument(
        "--max-wallclock-time",
        type=float,
        default=3600 * 4,
        help="Simulated wall-clock budget in seconds (default: 4 h).",
    )
    p.add_argument("--n-workers",  type=int, default=1)
    p.add_argument(
        "--all-datasets",
        action="store_true",
        default=False,
        help=(
            "Use source tasks from ALL datasets, not just the same dataset as "
            "the test task. Default: same-dataset only."
        ),
    )
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--cross-arch-only",
        action="store_true",
        default=False,
        help=(
            "Use only cross-architecture source tasks; exclude all same-architecture "
            "tasks. Requires --extra-architectures to be set."
        ),
    )
    p.add_argument(
        "--extra-architectures",
        nargs="*",
        default=[],
        choices=["vanillix", "varix", "ontix", "disentanglix"],
        help=(
            "Additional architectures to use as transfer-learning sources. "
            "Hyperparameters not shared with the target architecture are padded "
            "to span the full domain (effectively unconstrained). "
            "Example: --extra-architectures varix vanillix"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    if args.test_task is None:
        _bb = load_blackbox(f"autoencodix_{args.architecture}_tcga", local_files_only=True)
        args.test_task = sorted(_bb)[0]
        print(f"No --test-task given, defaulting to '{args.test_task}'")

    run(
        architecture=args.architecture,
        test_task=args.test_task,
        metric=args.metric,
        do_minimize=METRIC_MODE[args.metric],
        use_surrogates=args.use_surrogates,
        same_dataset_only=not args.all_datasets,
        max_wallclock_time=args.max_wallclock_time,
        n_workers=args.n_workers,
        random_seed=args.seed,
        extra_architectures=args.extra_architectures or None,
        cross_arch_only=args.cross_arch_only,
    )