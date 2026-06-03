import itertools
import logging
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from benchmarking.benchmarks import (
    benchmark_definitions,
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from benchmarking.utils import load_transfer_learning_evaluations, _sanitize_config_space, METRIC_MODE

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler

from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import QuantileBasedSurrogateSearcher
from syne_tune.optimizer.schedulers.transfer_learning.zero_shot import ZeroShotTransfer

transfer_methods = {
    "BoundingBox": BoundingBox,
    "QuantileTransfer": QuantileBasedSurrogateSearcher,
    "ZeroShot": ZeroShotTransfer,
}


def run(
    method_names,
    benchmark_names,
    seeds,
    max_num_trials_completed=100,
    n_workers: int = 1,
    all_datasets: bool = False,
    cross_arch_only: bool = False,
    cross_dataset_only: bool = False,
    match_modality_only: bool = False,
    extra_architectures: list = None,
):
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    print(f"Going to evaluate: {combinations}")
    exp_names = []
    
    if extra_architectures is None:
        extra_architectures = []

    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]

        print(f"Starting experiment ({method}/{benchmark_name}/{seed})")

        if "bbomix" in benchmark.blackbox_name:
            parts = benchmark.blackbox_name.split("_")
            target_arch = parts[1]
            test_dataset = parts[2]
        else:
            target_arch = benchmark.blackbox_name
            test_dataset = ""
            
        test_task = benchmark.dataset_name
        metric = benchmark.metric
        do_minimize = METRIC_MODE.get(metric, True)

        bb_dict = load_blackbox(benchmark.blackbox_name, local_files_only=True)
        
        if cross_dataset_only:
            dataset_list = ["tcga"] if test_dataset == "schc" else ["schc"]
        elif all_datasets:
            dataset_list = ["tcga", "schc"]
        else:
            dataset_list = [test_dataset]

        extra_bb_dicts = {}
        
        # 1. Add other dataset for the SAME architecture
        if (all_datasets or cross_dataset_only) and "bbomix" in benchmark.blackbox_name:
            if not cross_arch_only:
                other_ds = "tcga" if test_dataset == "schc" else "schc"
                arch_label = f"{target_arch}_{other_ds}"
                arch_bb_name = f"bbomix_{target_arch}_{other_ds}"
                extra_bb_dicts[arch_label] = load_blackbox(arch_bb_name, local_files_only=True)
            
        # 2. Add extra architectures
        for arch in extra_architectures:
            if arch == target_arch:
                continue
            for ds in dataset_list:
                arch_label = f"{arch}_{ds}"
                arch_bb_name = f"bbomix_{arch}_{ds}" if "bbomix" in benchmark.blackbox_name else arch
                extra_bb_dicts[arch_label] = load_blackbox(arch_bb_name, local_files_only=True)

        transfer_learning_evaluations = load_transfer_learning_evaluations(
            blackbox_name=benchmark.blackbox_name,
            test_task=test_task,
            metric=metric,
            bb_dict=bb_dict,
            same_dataset_only=(not all_datasets and not cross_dataset_only),
            extra_bb_dicts=extra_bb_dicts if extra_bb_dicts else None,
            cross_arch_only=cross_arch_only,
            cross_dataset_only=cross_dataset_only,
            match_modality_only=match_modality_only,
        )

        config_space = bb_dict[test_task].configuration_space

        if method == "BoundingBox":
            def _make_inner_scheduler(new_config_space, metric, do_minimize, random_seed):
                clean_space = _sanitize_config_space(new_config_space)
                return SingleObjectiveScheduler(
                    clean_space,
                    do_minimize=do_minimize,
                    searcher="random_search",
                    metric=metric,
                    random_seed=random_seed,
                )

            scheduler = BoundingBox(
                scheduler_fun=_make_inner_scheduler,
                config_space=config_space,
                metric=metric,
                do_minimize=do_minimize,
                num_hyperparameters_per_task=20,
                transfer_learning_evaluations=transfer_learning_evaluations,
                random_seed=seed,
            )
        elif method == "QuantileTransfer":
            searcher = QuantileBasedSurrogateSearcher(
                config_space=config_space,
                transfer_learning_evaluations=transfer_learning_evaluations,
                normalization="gaussian",
                max_fit_samples=100000,
                random_seed=seed,
            )

            scheduler = SingleObjectiveScheduler(
                config_space=config_space,
                searcher=searcher,
                metric=metric,
                do_minimize=do_minimize,
                random_seed=seed,
            )
        elif method == "ZeroShot":
            scheduler = ZeroShotTransfer(
                config_space=config_space,
                metric=metric,
                do_minimize=do_minimize,
                transfer_learning_evaluations=transfer_learning_evaluations,
                sort_transfer_learning_evaluations=not True, # use_surrogates=True by default for ZeroShot transfer cross runs
                use_surrogates=True,
                random_seed=seed,
            )
        else:
            raise ValueError(f"Unknown transfer method {method}")

        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=benchmark.surrogate,
            surrogate_kwargs=benchmark.surrogate_kwargs,
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time,
            #max_num_trials_completed=max_num_trials_completed
            #if max_num_trials_completed
            #else benchmark.max_num_trials_completed,
        )
        
        # Determine flags for naming
        flags = "".join([
            "A" if all_datasets else ("D" if cross_dataset_only else "a"),
            "X" if cross_arch_only else "x",
            "M" if match_modality_only else "m",
        ])
        
        if method == "ZeroShot":
            flags = "S" + flags # Assuming surrogates are used for zero shot
        
        extra_str = "+".join(sorted(extra_architectures)) if extra_architectures else "none"
        metric_short = {"metric_avg_ml_task_performance": "ds", "metric_valid_recon_loss": "rl", "metric_elapsed_time": "et"}.get(metric, metric[:4])
        
        tuner_name = f"results/{method}-{target_arch}-{test_task}-{metric_short}-{flags}-x{extra_str}-r{seed}"
        
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=0,
            callbacks=[SimulatorCallback()],
            results_update_interval=600,
            print_update_interval=30,
            tuner_name=tuner_name.replace("_", "-"),
            save_tuner=False,
            suffix_tuner_name=False,
            metadata={
                "seed": seed,
                "algorithm": method,
                "benchmark": benchmark_name,
            },
        )
        tuner.run()
        exp_names.append(tuner.name)
    return exp_names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="seed to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        required=False,
        default=0,
        help="If 1 runs all seeds between [0, args.seed] if 0 run only args.seed.",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=False,
        help="a method to run (BoundingBox, QuantileTransfer, ZeroShot), run all by default.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        help="a benchmark to run from benchmarks.py, run all by default.",
    )
    parser.add_argument(
        "--n_workers",
        help="number of workers to use when tuning.",
        type=int,
        default=1,
    )
    
    parser.add_argument("--all_datasets", action="store_true")
    parser.add_argument("--cross_arch_only", action="store_true")
    parser.add_argument("--cross_dataset_only", action="store_true")
    parser.add_argument("--match_modality_only", action="store_true")
    parser.add_argument("--extra_architectures", nargs="*", default=[])

    args, _ = parser.parse_known_args()
    if args.run_all_seeds:
        seeds = list(range(args.seed))
    else:
        seeds = [args.seed]
        
    method_names = [args.method] if args.method is not None else list(transfer_methods.keys())
    
    benchmark_names = (
        [args.benchmark]
        if args.benchmark is not None
        else [k for k in benchmark_definitions.keys() if "bbomix" in k]
    )
    
    run(
        method_names=method_names,
        benchmark_names=benchmark_names,
        seeds=seeds,
        n_workers=args.n_workers,
        all_datasets=args.all_datasets,
        cross_arch_only=args.cross_arch_only,
        cross_dataset_only=args.cross_dataset_only,
        match_modality_only=args.match_modality_only,
        extra_architectures=args.extra_architectures,
    )
