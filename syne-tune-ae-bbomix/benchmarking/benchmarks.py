from dataclasses import dataclass


@dataclass
class BenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_num_evaluations: int | None = None
    surrogate: str | None = None
    surrogate_kwargs: dict | None = None
    datasets: list[str] | None = None
    max_num_trials_finished: int | None = None


n_full_evals = 100


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        # allow to stop after having seen the equivalent of `n_full_evals` evaluations
        max_num_evaluations=100 * n_full_evals,
    )


def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=72000 if dataset_name == "ImageNet16-120" else 36000,
        max_num_evaluations=200 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
    )


def lcbench_benchmark(dataset_name, datasets):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=52 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        datasets=datasets,
    )


def tabrepo_benchmark(blackbox_name: str, dataset_name: str, datasets: list[str]):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=1 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",  # todo should also include time_train_s + time_infer_s as metric
        metric="metric_error_val",  # could also do rank
        mode="min",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        datasets=datasets,
    )


def hpob_benchmark(blackbox_name: str, dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=1 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_accuracy",
        mode="max",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


def bbomix_benchmark(blackbox_name: str, dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=72000,
        #max_num_evaluations=300*n_full_evals,
        n_workers=1,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_avg_ml_task_performance",
        mode="max",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


benchmark_definitions = {

}




bbomix_schc_search_spaces = [
    "vanillix_schc",
    "varix_schc",
    "ontix_schc",
    "disentanglix_schc",
    ]

bbomix_tcga_search_spaces = [
    "vanillix_tcga",
    "varix_tcga",
    "ontix_tcga",
    "disentanglix_tcga",
    ]

bbomix_schc_tasks = [
    "schc_RNA_METH_CLIN",
    "schc_METH_CLIN",
    "schc_RNA_CLIN",
]
bbomix_tcga_tasks = [
    "tcga_RNA_CLIN",
    "tcga_METH_CLIN",
    "tcga_DNA_CLIN",
    "tcga_RNA_DNA_METH_CLIN",
]


for task in bbomix_schc_tasks:
    for search_space in bbomix_schc_search_spaces:
        if "ontix" in search_space:
            for ontology in ["reactome", "chromosome"]:
                benchmark_definitions[
                    f"bbomix-{search_space}-"
                    + task.replace("_", "-").replace(".", "")
                    + "-"
                    + ontology
                ] = bbomix_benchmark(
                    blackbox_name=f"bbomix_{search_space}",
                    dataset_name=task + "_" + ontology,
                )
        else:
            benchmark_definitions[
                f"bbomix-{search_space}-" + task.replace("_", "-").replace(".", "")
            ] = bbomix_benchmark(
                blackbox_name=f"bbomix_{search_space}",
                dataset_name=task,
            )

for task in bbomix_tcga_tasks:
    for search_space in bbomix_tcga_search_spaces:
        if "ontix" in search_space:
            for ontology in ["reactome", "chromosome"]:
                benchmark_definitions[
                    f"bbomix-{search_space}-"
                    + task.replace("_", "-").replace(".", "")
                    + "-"
                    + ontology
                ] = bbomix_benchmark(
                    blackbox_name=f"bbomix_{search_space}",
                    dataset_name=task + "_" + ontology,
                )
        else:
            benchmark_definitions[
                f"bbomix_{search_space}-" + task.replace("_", "-").replace(".", "")
            ] = bbomix_benchmark(
                blackbox_name=f"bbomix_{search_space}",
                dataset_name=task,
            )


if __name__ == "__main__":
    from syne_tune.blackbox_repository import load_blackbox

    for benchmark, benchmark_case in benchmark_definitions.items():
        print(benchmark)
        #bb = load_blackbox(benchmark_case.blackbox_name)
