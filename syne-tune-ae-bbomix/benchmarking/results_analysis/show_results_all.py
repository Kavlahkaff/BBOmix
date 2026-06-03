import logging
from argparse import ArgumentParser
from pathlib import Path

import dill
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarking.results_analysis.load_experiments_parallel import (
    load_benchmark_results,
)
from benchmarking.results_analysis.method_styles import (
    plot_range,
)
from syne_tune.util import catchtime


def figure_folder(path):
    import syne_tune

    root = Path(syne_tune.__path__[0]).parent
    figure_path = root / path
    figure_path.mkdir(exist_ok=True, parents=True)
    print(figure_path)
    return figure_path


lw = 2.5
alpha = 0.7
matplotlib.rcParams.update({"font.size": 20})

def get_experiment_filter(average_transfer_methods: bool):
    def parse_transfer_algorithm_name(metadata):
        tuner_name = metadata.get("tuner_name", "")
        alg = metadata.get("algorithm", "")
        if alg in ["BoundingBox", "QuantileTransfer", "ZeroShot"] and tuner_name:
            parts = tuner_name.split("-")
            try:
                flags = parts[-3]
                extra_str = parts[-2]
                
                is_all = '+' in extra_str
                
                if average_transfer_methods:
                    if is_all:
                        metadata["algorithm"] = f"{alg} (All Archs)"
                    else:
                        metadata["algorithm"] = alg
                    return True
                
                if flags.startswith('S'):
                    flags = flags[1:]
                    
                all_datasets = 'A' in flags
                
                if not all_datasets and extra_str == "xnone":
                    variation = "Modality"
                elif all_datasets and extra_str == "xnone":
                    variation = "Cross-Dataset"
                elif all_datasets and extra_str != "xnone":
                    archs_str = extra_str[1:]
                    variation = f"Cross-Arch ({archs_str})"
                else:
                    variation = f"Transfer ({flags}, {extra_str})"
                    
                metadata["algorithm"] = f"{alg} - {variation}"
            except IndexError:
                if average_transfer_methods:
                    metadata["algorithm"] = alg
        return True
    return parse_transfer_algorithm_name

def plot_result_benchmark(
    t_range: np.array,
    method_dict: dict[str, np.array],
    title: str,
    rename_dict: dict,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = True,
    y_log_scale: bool = False,
    x_log_scale: bool = False,
    y_limits: tuple[float, float] = None,
    mode: str = "min",
    color_dict: dict = None,
):
    agg_results = {}

    if plot_regret:
        min_value = min([np.nanmin(v) for v in method_dict.values()])
        max_value = max([np.nanmax(v) for v in method_dict.values()])
        best_result, worse_result = (
            (min_value, max_value) if mode == "min" else (max_value, min_value)
        )

    if len(method_dict) > 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm in method_dict.keys():
            if methods_to_show is not None and algorithm not in methods_to_show:
                continue
            renamed_algorithm = rename_dict.get(algorithm, algorithm)

            # (num_seeds, num_time_steps)
            y_ranges = method_dict[algorithm]
            if plot_regret:
                diff = worse_result - best_result
                if diff == 0:
                    diff = 1.0
                y_ranges = (y_ranges - best_result) / diff
            mean = np.nanmean(y_ranges, axis=0)
            std = np.nanstd(y_ranges, axis=0, ddof=1) / np.sqrt(y_ranges.shape[0])
            color = color_dict.get(algorithm) if color_dict else None
            linestyle = "--" if "(All Archs)" in algorithm else "-"
            ax.fill_between(
                t_range,
                mean - std,
                mean + std,
                color=color,
                alpha=0.1,
            )
            ax.plot(
                t_range,
                mean,
                label=renamed_algorithm,
                color=color,
                alpha=alpha,
                lw=lw,
                linestyle=linestyle,
            )

            agg_results[algorithm] = mean
        if y_log_scale:
            ax.set_yscale("log")
        if x_log_scale:
            ax.set_xscale("log")
        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_xlabel("Wallclock time")
        
        # Adjust legend if there are many methods
        ax.legend(loc='upper center', fontsize=10)
        ax.set_title(title)
        ax.grid(True, which="major", axis="both")
    return ax


def plot_task_performance_over_time(
    benchmark_results: dict[str, tuple[np.array, dict[str, np.array]]],
    rename_dict: dict,
    result_folder: Path,
    title: str = None,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = False,
    y_log_scale: bool = False,
    x_log_scale: bool = False,
    y_limits: tuple[float, float] = None,
    mode: str = "min",
    color_dict: dict = None,
):
    print(f"plot rank through time on {result_folder}")
    for benchmark, (t_range, method_dict) in benchmark_results.items():
        # filter benchmark_results method_dict to only include methods_to_show
        if methods_to_show is not None:
            method_dict_filtered = {k: v for k, v in method_dict.items() if k in methods_to_show}
            if len(method_dict_filtered) == 0:
                continue
        else:
            method_dict_filtered = method_dict

        ax = plot_result_benchmark(
            t_range=t_range,
            method_dict=method_dict_filtered,
            title=benchmark,
            ax=ax,
            methods_to_show=methods_to_show,
            rename_dict=rename_dict,
            plot_regret=plot_regret,
            y_log_scale=y_log_scale,
            x_log_scale=x_log_scale,
            y_limits=y_limits,
            mode=mode,
            color_dict=color_dict,
        )
        if ax is not None:
            ax.set_ylabel("objective")
            if title is not None:
                ax.set_title(title)
            if not plot_regret and y_limits is None:
                if benchmark in plot_range:
                    plotargs = plot_range[benchmark]
                    ax.set_ylim([plotargs.ymin, plotargs.ymax / 3])
                    if x_log_scale and plotargs.xmin <= 0:
                        ax.set_xlim([max(1e-3, plotargs.xmin), plotargs.xmax])
                    else:
                        ax.set_xlim([plotargs.xmin, plotargs.xmax])

            plt.tight_layout()
            filepath = result_folder / f"{benchmark}.pdf"
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
        ax = None


def load_and_cache(
    path: Path,
    methods: list[str] | None = None,
    load_cache_if_exists: bool = True,
    num_time_steps=100,
    max_seed=30,
    experiment_filter=None,
    mode: str = "min",
    average_transfer_methods: bool = False,
):
    suffix = "-averaged" if average_transfer_methods else ""
    result_file = (Path(path) / f"results-cache-all-{mode}{suffix}.dill").expanduser()
    if load_cache_if_exists and result_file.exists():
        with catchtime(f"loading results from {result_file}"):
            with open(result_file, "rb") as f:
                benchmark_results = dill.load(f)
    else:
        print(f"regenerating results to {result_file}")
        with catchtime("load benchmark results"):
            benchmark_results = load_benchmark_results(
                path=path,
                methods=methods,
                num_time_steps=num_time_steps,
                max_seed=max_seed,
                experiment_filter=experiment_filter,
                mode=mode,
            )

        with open(result_file, "wb") as f:
            dill.dump(benchmark_results, f)

    return benchmark_results

def merge_benchmark_results(results_dicts):
    merged = {}
    for res in results_dicts:
        for bench, (t_range, method_dict) in res.items():
            if bench not in merged:
                merged[bench] = (t_range, method_dict.copy())
            else:
                existing_t_range, existing_method_dict = merged[bench]
                for alg, y_ranges in method_dict.items():
                    if alg not in existing_method_dict:
                        existing_method_dict[alg] = y_ranges
                    else:
                        existing_method_dict[alg] = np.concatenate(
                            [existing_method_dict[alg], y_ranges], axis=0
                        )

    # Enforce uniform seed count across all methods and benchmarks
    all_shapes = [
        values.shape[0]
        for bench, (t_range, method_dict) in merged.items()
        for values in method_dict.values()
    ]
    if all_shapes:
        min_seeds = min(all_shapes)
        for bench, (t_range, method_dict) in merged.items():
            for alg, y_ranges in method_dict.items():
                method_dict[alg] = y_ranges[:min_seeds]

    return merged


def plot_ranks(
    ranks,
    benchmark_results,
    title: str,
    rename_dict: dict,
    result_folder: Path,
    methods_to_show: list[str],
    color_dict: dict = None,
    x_log_scale: bool = False,
    t_range: np.array = None,
):
    plt.figure(figsize=(10, 6))
    ys = np.nanmean(ranks.reshape(benchmark_results.shape), axis=(1, 2))

    if t_range is not None:
        xs = t_range
    else:
        if x_log_scale:
            xs = np.linspace(1/ys.shape[-1], 1, ys.shape[-1])
        else:
            xs = np.linspace(0, 1, ys.shape[-1])

    for i, method in enumerate(methods_to_show):
        color = color_dict.get(method) if color_dict else None
        linestyle = "--" if "(All Archs)" in method else "-"
        plt.plot(
            xs,
            ys[i],
            label=rename_dict.get(method, method),
            color=color,
            alpha=alpha,
            lw=lw,
            linestyle=linestyle,
        )
    plt.xlabel("Wallclock time" if t_range is not None else "% Budget Used")
    plt.ylabel("Method rank")
    if x_log_scale:
        plt.xscale("log")
        if t_range is not None:
            plt.xlim(max(1e-3, xs[0]), xs[-1])
        else:
            plt.xlim(1/ys.shape[-1], 1)
    else:
        if t_range is not None:
            plt.xlim(xs[0], xs[-1])
        else:
            plt.xlim(0, 1)
    plt.grid(True, which="major", axis="both")
    plt.title(title)
    plt.legend(loc='upper center', fontsize=10)
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}_rank.pdf", bbox_inches='tight')
    plt.close()


def stack_benchmark_results(
    benchmark_results_dict: dict[str, tuple[np.array, dict[str, np.array]]],
    methods_to_show: list[str],
    benchmark_families: list[str],
    mode: str = "min",
) -> tuple[dict[str, np.array], list[str], dict[str, np.array]]:
    methods_to_keep = list(methods_to_show)

    res = {}
    t_ranges = {}
    for benchmark_family in benchmark_families:
        benchmarks_family = [
            benchmark
            for benchmark in benchmark_results_dict.keys()
            if benchmark_family in benchmark
        ]

        if not benchmarks_family:
            continue

        benchmark_results = []
        for benchmark in benchmarks_family:
            benchmark_result = []

            valid_arrays = list(benchmark_results_dict[benchmark][1].values())
            if not valid_arrays:
                continue
            shape = valid_arrays[0].shape

            for method in methods_to_keep:
                if method in benchmark_results_dict[benchmark][1]:
                    benchmark_result.append(benchmark_results_dict[benchmark][1][method])
                else:
                    benchmark_result.append(np.full(shape, np.nan))

            benchmark_result = np.stack(benchmark_result)
            benchmark_results.append(benchmark_result)

        if len(benchmark_results) == 0:
            continue

        benchmark_results = np.stack(benchmark_results)

        if mode == "max":
            benchmark_results *= -1

        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        res[benchmark_family] = benchmark_results.swapaxes(0, 1)

        family_t_ranges = [benchmark_results_dict[benchmark][0] for benchmark in benchmarks_family]
        t_ranges[benchmark_family] = np.mean(family_t_ranges, axis=0)

    return res, methods_to_keep, t_ranges


def generate_rank_results(
    stacked_benchmark_results: dict[str, np.array],
    stacked_t_ranges: dict[str, np.array],
    methods_to_show: list[str],
    rename_dict: dict,
    result_folder: Path,
    color_dict: dict = None,
    x_log_scale: bool = False,
):
    rows = []
    for benchmark_family, benchmark_results in stacked_benchmark_results.items():
        print(benchmark_family)
        ranks = pd.DataFrame(
            benchmark_results.reshape(len(benchmark_results), -1)
        ).rank()
        ranks = ranks.values.reshape(benchmark_results.shape)
        avg_ranks_per_tasks = np.nanmean(ranks, axis=(2, 3))
        for i in range(benchmark_results.shape[1]):
            row = {"benchmark": f"{benchmark_family}-{i}"}
            row.update(dict(zip(methods_to_show, avg_ranks_per_tasks[:, i])))
            rows.append(row)

        plot_ranks(
            ranks,
            benchmark_results,
            benchmark_family,
            rename_dict,
            result_folder,
            methods_to_show,
            color_dict=color_dict,
            x_log_scale=x_log_scale,
            t_range=stacked_t_ranges[benchmark_family],
        )

    all_results = np.concatenate(list(stacked_benchmark_results.values()), axis=1)
    all_ranks = pd.DataFrame(all_results.reshape(len(all_results), -1)).rank()

    all_t_ranges = np.mean(list(stacked_t_ranges.values()), axis=0) if stacked_t_ranges else None

    plot_ranks(
        all_ranks.values,
        all_results,
        "Average-rank",
        rename_dict,
        result_folder,
        methods_to_show,
        color_dict=color_dict,
        x_log_scale=x_log_scale,
        t_range=all_t_ranges,
    )


def plot_average_normalized_regret(
    stacked_benchmark_results,
    stacked_t_ranges,
    rename_dict: dict,
    result_folder: Path,
    title: str = None,
    show_ci: bool = False,
    ax=None,
    methods_to_show: list = None,
    color_dict: dict = None,
    x_log_scale: bool = False,
):
    import warnings
    normalized_regrets = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for benchmark_family, benchmark_results in stacked_benchmark_results.items():
            benchmark_results_best = np.nanmin(benchmark_results, axis=(0, 2, 3), keepdims=True)
            benchmark_results_worse = np.nanmax(benchmark_results, axis=(0, 2, 3), keepdims=True)
            # Avoid division by zero
            diff = benchmark_results_worse - benchmark_results_best
            diff[diff == 0] = 1.0
            normalized_regret = (benchmark_results - benchmark_results_best) / diff
            normalized_regrets.append(normalized_regret)

        if not normalized_regrets:
            return

        normalized_regrets = np.concatenate(normalized_regrets, axis=1)

        avg_regret = np.nanmean(normalized_regrets, axis=(1, 2))
        std_regret = np.nanmean(np.nanstd(normalized_regrets, axis=2), axis=1) if show_ci else None

        all_t_ranges = np.mean(list(stacked_t_ranges.values()), axis=0) if stacked_t_ranges else None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for i, algorithm in enumerate(methods_to_show):
        renamed_algorithm = rename_dict.get(algorithm, algorithm)
        mean = avg_regret[i]
        color = color_dict.get(algorithm) if color_dict else None
        linestyle = "--" if "(All Archs)" in algorithm else "-"

        if all_t_ranges is not None:
            xs = all_t_ranges
        else:
            xs = np.arange(1, len(mean) + 1) / len(mean) if x_log_scale else np.arange(len(mean)) / len(mean)

        ax.plot(
            xs,
            mean,
            label=renamed_algorithm,
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
        )
        if show_ci:
            std = std_regret[i]
            ax.fill_between(
                xs,
                mean - std,
                mean + std,
                color=color,
                alpha=0.1,
            )
        ax.set_yscale("log")
        if x_log_scale:
            ax.set_xscale("log")

    plt.xlabel("Wallclock time" if all_t_ranges is not None else "% Budget Used")
    ax.set_ylabel("Average normalized regret")
    if x_log_scale:
        if all_t_ranges is not None:
            plt.xlim(max(1e-3, xs[0]), xs[-1])
        else:
            plt.xlim(1/len(mean) if len(mean) > 0 else 1e-3, 1)
    else:
        if all_t_ranges is not None:
            plt.xlim(xs[0], xs[-1])
        else:
            plt.xlim(0, 1)
    # plt.ylim(6e-4, 1e-1)
    plt.grid(True, which="major", axis="both")
    if title is not None:
        plt.title(title)
    plt.legend(loc='upper center', fontsize=10)
    plt.tight_layout()
    plt.savefig(result_folder / f"{title if title else 'Normalized-regret'}.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        required=True,
        help="paths where to find the results",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        required=False,
        help="Whether the optimization trajectories represent a maximization problem",
    )
    parser.add_argument(
        "--x_log_scale",
        action="store_true",
        required=False,
        help="Whether to make the x-axis log scale",
    )
    parser.add_argument(
        "--average_transfer_methods",
        action="store_true",
        required=False,
        help="Whether to average transfer learning methods (BoundingBox, QuantileTransfer, ZeroShot)",
    )
    parser.add_argument(
        "--methods_to_show",
        type=str,
        nargs="+",
        required=False,
        help="List of algorithms to include in the plots.",
    )

    args, _ = parser.parse_known_args()

    mode = "max" if args.maximize else "min"
    max_seed = args.max_seed
    num_time_steps = 100

    all_results_dicts = []

    experiment_filter = get_experiment_filter(args.average_transfer_methods)

    for path_str in args.paths:
        path = Path(path_str)
        assert path.exists(), f"Path {path} does not exist"

        with catchtime(f"load benchmark results from {path}"):
            benchmark_results = load_and_cache(
                path=path,
                load_cache_if_exists=args.reuse_cache,
                max_seed=max_seed,
                num_time_steps=num_time_steps,
                methods=None, # Load ALL methods
                experiment_filter=experiment_filter,
                mode=mode,
                average_transfer_methods=args.average_transfer_methods,
            )
            all_results_dicts.append(benchmark_results)

    with catchtime("Merge results across all paths"):
        merged_results = merge_benchmark_results(all_results_dicts)

    assert len(merged_results) > 0, "Could not find any results in paths provided."

    # Dynamically extract all benchmark families from the merged data
    benchmark_families = sorted(list(merged_results.keys()))

    # Find all unique methods across the merged dataset
    all_methods = set()
    for bench, (t_range, method_dict) in merged_results.items():
        all_methods.update(method_dict.keys())

    if args.methods_to_show:
        methods_to_show = []
        for m in all_methods:
            if any(m.startswith(choice) for choice in args.methods_to_show):
                methods_to_show.append(m)
        methods_to_show = sorted(methods_to_show)
    else:
        methods_to_show = sorted(list(all_methods))

    print(f"Total methods to plot ({len(methods_to_show)}):")
    for m in methods_to_show:
        print(f" - {m}")

    # Generate a consistent color dictionary for all methods
    HARDCODED_COLORS = {
        "TPE": "tab:blue",
        "RS": "tab:orange",
        "Random Search": "black",
        "BORE": "tab:green",
        "CQR": "tab:red",
        "REA": "#8172B3",
        "ZeroShot": "tab:purple",
        "BoundingBox": "tab:brown",
        "QuantileTransfer": "tab:pink",
        "ASHA": "tab:gray",
        "BOHB": "tab:olive",
        "ASHACQR": "tab:cyan",
        "ASHABORE": "tab:orange",
    }

    cmap = plt.get_cmap("tab20")
    color_dict = {}
    fallback_idx = 0

    for m in methods_to_show:
        assigned_color = None
        for base_name, color in sorted(HARDCODED_COLORS.items(), key=lambda x: len(x[0]), reverse=True):
            if m.startswith(base_name):
                assigned_color = color
                break
        if assigned_color is None:
            assigned_color = cmap(fallback_idx % 20)
            fallback_idx += 1
        color_dict[m] = assigned_color

    result_folder = figure_folder(Path("figures") / "single_bbomix")
    result_folder.mkdir(parents=True, exist_ok=True)

    # Rename dict can be empty to use raw names
    rename_dict = {
        "BoundingBox": "Bounding Box",
        "BoundingBox (All Archs)": "Bounding Box (All)",
        "QuantileTransfer": "Quantile Transfer",
        "QuantileTransfer (All Archs)": "Quantile Transfer (All)",
        "ZeroShot": "Zero Shot",
        "ZeroShot (All Archs)": "Zero Shot (All)",
        "RS": "Random Search",
    }

    stacked_benchmark_results, final_methods, stacked_t_ranges = stack_benchmark_results(
        benchmark_results_dict=merged_results,
        methods_to_show=methods_to_show,
        benchmark_families=benchmark_families,
        mode=mode,
    )

    if len(stacked_benchmark_results) > 0:
        with catchtime("generating rank table"):
            generate_rank_results(
                stacked_benchmark_results=stacked_benchmark_results,
                stacked_t_ranges=stacked_t_ranges,
                methods_to_show=final_methods,
                rename_dict=rename_dict,
                result_folder=result_folder,
                color_dict=color_dict,
                x_log_scale=args.x_log_scale,
            )

        with catchtime("generating plots per task"):
            plot_task_performance_over_time(
                benchmark_results=merged_results,
                methods_to_show=methods_to_show,
                rename_dict=rename_dict,
                result_folder=result_folder,
                y_log_scale=True,
                x_log_scale=args.x_log_scale,
                y_limits=None,
                mode=mode,
                color_dict=color_dict,
            )

        with catchtime("generating average normalized regret"):
            plot_average_normalized_regret(
                stacked_benchmark_results=stacked_benchmark_results,
                stacked_t_ranges=stacked_t_ranges,
                methods_to_show=final_methods,
                rename_dict=rename_dict,
                result_folder=result_folder,
                title="Normalized-regret",
                color_dict=color_dict,
                x_log_scale=args.x_log_scale,
            )
        
    print(f"Finished generating all plots in {result_folder}")
