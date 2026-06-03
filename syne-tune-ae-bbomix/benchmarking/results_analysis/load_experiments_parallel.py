import json
import logging
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
from pyparfor import parfor
from tqdm import tqdm

from syne_tune.util import catchtime


def load_result(name, metadata, path, mode=None):
    metric_name = metadata["metric_names"][0]
    usecols = [metric_name, "st_tuner_time"]
    try:
        df = pd.read_csv(path / name / "results.csv.zip", usecols=usecols)
        
        benchmark = metadata["benchmark"]
        if mode is None:
            if "lcbench" in benchmark or "nas301" in benchmark:
                current_mode = "max"
            else:
                current_mode = "min"
        else:
            current_mode = mode
            
        if current_mode == "min":
            best = df[metric_name].cummin().values
        else:
            best = df[metric_name].cummax().values
            
        t = df["st_tuner_time"].values
        return t, best
    except Exception:
        return None


def process_benchmark(benchmark_data, num_time_steps: int = 20):
    t_min = 0
    # the last time step is the median of the stopping time of all algorithms
    max_times = []
    for algorithm, seed_data in benchmark_data.items():
        max_time = max([np.max(t) if len(t) > 0 else 0 for seed, t, y in seed_data] + [0])
        max_times.append(max_time)
    t_max = np.median(max_times) if max_times else 0
    
    t_range = np.linspace(t_min, t_max, num_time_steps)
    seed_results = {}
    for algorithm, seed_data in benchmark_data.items():
        seed_data.sort(key=lambda x: x[0])  # sort by seed
        
        y_ranges = []
        for seed, t, y in seed_data:
            if len(t) == 0:
                continue
            indices = np.searchsorted(t, t_range, side="left")
            y_range = y[np.clip(indices, 0, len(y) - 1)]
            y_ranges.append(y_range)

        if len(y_ranges) > 0:
            seed_results[algorithm] = np.stack(y_ranges)
            
    return t_range, seed_results


def show_number_seeds(results_by_benchmark):
    seed_counts = defaultdict(dict)
    for benchmark, benchmark_data in results_by_benchmark.items():
        for algorithm, seed_data in benchmark_data.items():
            seed_counts[benchmark][algorithm] = len(seed_data)
            
    df_seeds = pd.DataFrame(seed_counts)
    print("number of seeds available:")
    print(df_seeds.to_string())


def convert_all_to_numpy(
    results_by_benchmark: dict,
    num_time_steps: int,
    max_seed: int,
    engine: str,
):
    benchmarks_numpy = parfor(
        f=process_benchmark,
        inputs=[
            {"benchmark_data": benchmark_data, "num_time_steps": num_time_steps}
            for benchmark_data in results_by_benchmark.values()
        ],
        engine=engine,
    )
    
    # Wrap in list to avoid multiple generator consumption issues if any
    benchmarks_numpy = list(benchmarks_numpy)
    
    if len(benchmarks_numpy) == 0:
        return {}
        
    min_num_seeds = min(
        values.shape[0] for x in benchmarks_numpy for method, values in x[1].items() if len(x[1]) > 0
    )
    if max_seed is not None and min_num_seeds < max_seed:
        logging.warning(
            f"some methods have only {min_num_seeds} instead of the {max_seed} seeds asked, slicing all results "
            f"to {min_num_seeds} seeds."
        )
        num_seeds = {
            method: values.shape[0]
            for x in benchmarks_numpy
            for method, values in x[1].items()
        }
        logging.warning(num_seeds)
        benchmarks_numpy = [
            (
                t_range,
                {
                    method: values[:min_num_seeds]
                    for method, values in method_values.items()
                },
            )
            for (t_range, method_values) in benchmarks_numpy
        ]
    return dict(zip(results_by_benchmark.keys(), benchmarks_numpy))


def get_metadata(root: Path):
    metadatas = {}
    for metadata_path in root.rglob(f"*metadata.json"):
        with open(metadata_path, "r") as f:
            folder = metadata_path.parent.name
            try:
                metadata = json.load(f)
                metadata["tuner_name"] = folder
                metadatas[folder] = metadata
            except JSONDecodeError as e:
                print(metadata_path)
                raise e

    return metadatas


def load_benchmark_results(
    path: str | Path,
    methods: list[str],
    num_time_steps: int = 20,
    max_seed: int = None,
    experiment_filter=None,
    engine: str = "joblib",
    mode: str = None,
) -> dict[str, tuple[np.array, dict[str, np.array]]]:
    """
    :param path: where results are stored
    :param methods: list of methods to consider
    :param num_time_steps: number of time steps considered for results aggregation
    :param max_seed: maximum seed to load, default to None to load all seeds
    :param experiment_filter:
    :param engine: parallel engine to use, can be ["sequential", "ray", "joblib", "futures"]
    :param mode: "min" or "max"
    :return:
    """
    path = Path(path)

    with catchtime("Load metadata"):
        metadatas = get_metadata(root=path)

    if experiment_filter:
        metadatas = {k: v for k, v in metadatas.items() if experiment_filter(v)}

    # todo strict metadata filtering as the one above may fail
    methods = set(methods) if methods is not None else None
    metadatas = {
        k: v
        for k, v in metadatas.items()
        if (max_seed is None or v["seed"] < max_seed)
        and (methods is None or v["algorithm"] in methods)
    }
    print(f"loaded {len(metadatas)} experiment metadata")

    with catchtime("Load results dataframes"):
        # load results in parallel
        dfs = parfor(
            lambda name, metadata: load_result(name, metadata, path, mode),
            inputs=list(metadatas.items()),
            engine=engine,
        )

    with catchtime("Compute best result over time"):
        results_by_benchmark = defaultdict(lambda: defaultdict(list))
        for (name, metadata), result in zip(metadatas.items(), dfs):
            if result is None:
                continue
            t, y = result
            benchmark = metadata["benchmark"]
            algorithm = metadata["algorithm"]
            seed = metadata["seed"]
            results_by_benchmark[benchmark][algorithm].append((seed, t, y))

    show_number_seeds(results_by_benchmark)

    with catchtime("Convert to numpy (num_seeds, num_time_steps)"):
        benchmark_results = convert_all_to_numpy(
            results_by_benchmark,
            num_time_steps,
            max_seed,
            engine=engine,
        )
    return benchmark_results
