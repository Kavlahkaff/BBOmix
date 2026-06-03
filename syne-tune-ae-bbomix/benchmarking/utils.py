from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import \
    TransferLearningTaskEvaluations
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

METRIC_ELAPSED_TIME = "metric_elapsed_time"
METRIC_RECON_LOSS   = "metric_valid_recon_loss"
METRIC_DOWNSTREAM   = "metric_avg_ml_task_performance"
TIME_ATTR           = "epoch"

METRIC_MODE = {
    METRIC_DOWNSTREAM:   False,   # maximise AUC
    METRIC_RECON_LOSS:   True,    # minimise reconstruction loss
    METRIC_ELAPSED_TIME: True,    # minimise runtime
}

def _sanitize_hyperparameters(hp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast every DataFrame column to native Python scalar types (int/float/str).

    Parquet-backed blackboxes store values as np.int64 / np.float64.
    syne-tune's HyperparameterRangeCategorical asserts plain Python int, float,
    or str — np.int64 fails that check, causing an AssertionError deep inside
    the searcher factory.  This sanitizer is the single place we fix that.
    """
    out = hp_df.copy()
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype(int)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(float)
        else:
            out[col] = out[col].astype(str)
    return out


def _sanitize_config_space(config_space: dict) -> dict:
    """
    Walk a syne-tune config space and cast any numpy scalars inside Categorical
    domains to native Python types.

    BoundingBox._compute_box builds the restricted space by calling
    ``hp_df.loc[:, name].unique()``, which yields np.int64 / np.float64 values
    that end up as category choices.  syne-tune's HyperparameterRangeCategorical
    asserts plain int / float / str, so we must convert them here — in the
    scheduler_fun lambda — before the inner scheduler is constructed.
    """
    import numpy as np
    from syne_tune.config_space import Categorical, choice

    def _to_native(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    new_cs = {}
    for name, domain in config_space.items():
        if isinstance(domain, Categorical):
            new_cs[name] = choice([_to_native(c) for c in domain.categories])
        else:
            new_cs[name] = domain
    return new_cs


def _project_to_target_space(
    hp_df: pd.DataFrame,
    source_config_space: dict,
    target_config_space: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Align a source task's hyperparameters to the full target config space.

    - Shared keys: kept as-is (source values).
    - Target-only keys: padded with values that span the domain's full range,
      so BoundingBox computes trivially wide bounds for those dimensions and
      leaves them unconstrained.
      * Categorical  → cycle through all categories.
      * Float/Int    → alternate between lower and upper bound.

    transfer_learning_mixin._check_consistency asserts that every key in the
    *target* config space exists in each source task's hyperparameters columns,
    so we must include all target keys rather than dropping unknown ones.

    Returns:
        padded_hp_df  – DataFrame with exactly the target config-space columns.
        target_config_space – unchanged (returned for call-site symmetry).
    """
    from syne_tune.config_space import Categorical, Float, Integer

    if not any(k in source_config_space for k in target_config_space):
        logger.warning(
            "Source task shares no hyperparameter keys with the target — skipping."
        )
        return pd.DataFrame(), {}

    n = len(hp_df)
    out = pd.DataFrame(index=hp_df.index)

    for key, domain in target_config_space.items():
        if key in hp_df.columns:
            out[key] = hp_df[key].values
        else:
            # Pad with spanning values so min/max = full domain range
            if isinstance(domain, Categorical):
                cats = domain.categories
                out[key] = [cats[i % len(cats)] for i in range(n)]
            elif isinstance(domain, (Float, Integer)):
                lo, hi = domain.lower, domain.upper
                out[key] = [lo if i % 2 == 0 else hi for i in range(n)]
            else:
                # Fallback: repeat the first value of whatever the domain exposes
                try:
                    val = next(iter(domain))
                except Exception:
                    val = 0
                out[key] = val

    return out, target_config_space


def load_transfer_learning_evaluations(
    blackbox_name: str,
    test_task: str,
    metric: str,
    bb_dict: dict[str, BlackboxTabular] | None = None,
    same_dataset_only: bool = True,
    extra_bb_dicts: dict[str, dict[str, BlackboxTabular]] | None = None,
    cross_arch_only: bool = False,
    cross_dataset_only: bool = False,
    match_modality_only: bool = False,
) -> dict[str, TransferLearningTaskEvaluations]:
    """
    Build transfer-learning evaluations for all source tasks.

    Parameters
    ----------
    blackbox_name :
        Name of the *target* architecture's blackbox (e.g. ``"autoencodix_ontix"``).
    test_task :
        Task name held out as the optimisation target.
    metric :
        Objective to transfer.
    bb_dict :
        Pre-loaded blackbox dict for ``blackbox_name``.  Loaded on demand when
        ``None``.
    same_dataset_only :
        When ``True``, only include source tasks whose dataset prefix matches
        the test task's prefix (e.g. ``"TCGA"``).
    extra_bb_dicts :
        Optional mapping ``{architecture_label: bb_dict}`` of *additional*
        architectures to use as transfer sources.  Their config spaces may
        differ from the target; mismatched hyperparameters are handled by
        :func:`_project_to_target_space` — unknown dimensions are left
        unconstrained so BoundingBox samples them freely.
    cross_arch_only :
        When ``True``, same-architecture source tasks are excluded entirely.
        Only tasks from ``extra_bb_dicts`` are used as transfer sources.
        Useful for measuring how well transfer works across architectures
        without any same-architecture signal leaking in.
    """
    if bb_dict is None:
        bb_dict = load_blackbox(blackbox_name)

    if "BBOmix" in blackbox_name or "bbomix" in blackbox_name:
        target_arch = blackbox_name.split("_")[1]
    elif "autoencodix" in blackbox_name:
        target_arch = blackbox_name.split("_")[1]
    else:
        target_arch = blackbox_name

    # Target config space — used to project cross-architecture sources
    target_config_space = bb_dict[test_task].configuration_space

    # Determine the column index for the requested metric from the test task
    reference_names = bb_dict[test_task].objectives_names
    try:
        metric_index = reference_names.index(metric)
    except ValueError:
        raise ValueError(
            f"Metric '{metric}' not found in blackbox objectives {reference_names}"
        )

    # Collect all (task, blackbox, is_same_architecture) triples to iterate
    source_candidates: list[tuple[str, BlackboxTabular, bool]] = []
    if not cross_arch_only and not cross_dataset_only:
        source_candidates = [
            (task, bb, True)
            for task, bb in bb_dict.items()
            if task != test_task
        ]
    else:
        if cross_arch_only:
            logger.info("cross_arch_only=True: skipping all same-architecture source tasks.")
        if cross_dataset_only:
            logger.info("cross_dataset_only=True: skipping all same-dataset source tasks.")
    if extra_bb_dicts:
        for arch_label, ext_bb in extra_bb_dicts.items():
            arch_name = arch_label.split("_")[0]
            is_same_arch = (arch_name == target_arch)

            if cross_arch_only and is_same_arch:
                continue

            for task, bb in ext_bb.items():
                source_candidates.append((f"{arch_label}/{task}", bb, is_same_arch))

    transfer_evals: dict[str, TransferLearningTaskEvaluations] = {}
    test_dataset = test_task.split("_")[0]
    target_modality = test_task.split("_", 1)[1] if "_" in test_task else ""

    for task_key, bb, same_arch in source_candidates:
        # Dataset-prefix filtering
        raw_task   = task_key.split("/")[-1]          # strip arch prefix if present
        task_dataset = raw_task.split("_")[0]
        if same_dataset_only and task_dataset != test_dataset:
            logger.debug(
                "Skipping cross-dataset source '%s' (test dataset: '%s')",
                task_key, test_dataset,
            )
            continue
        if cross_dataset_only and task_dataset == test_dataset:
            logger.debug(
                "Skipping same-dataset source '%s' (test dataset: '%s')",
                task_key, test_dataset,
            )
            continue

        if match_modality_only:
            source_modality = raw_task.split("_", 1)[1] if "_" in raw_task else ""
            if source_modality != target_modality:
                logger.debug(
                    "Skipping mismatched modality source '%s' (target modality: '%s')",
                    task_key, target_modality,
                )
                continue

        # ── Metric index for this source blackbox ─────────────────────────────
        # Different architectures may order objectives differently.
        try:
            src_metric_index = bb.objectives_names.index(metric)
        except ValueError:
            logger.warning(
                "Metric '%s' not found in source task '%s' — skipping.",
                metric, task_key,
            )
            continue

        evals = bb.objectives_evaluations                       # (H, S, E, O)
        single_obj = evals[..., src_metric_index : src_metric_index + 1]

        # Reduce to last epoch, keep fidelity axis (size 1)
        last_epoch = single_obj[:, :, -1:, :]                  # (H, S, 1, 1)
        nan_mask   = np.isnan(last_epoch).squeeze(axis=(-2, -1))
        if nan_mask.any():
            fallback   = np.nanmax(single_obj, axis=2, keepdims=True)
            last_epoch = np.where(nan_mask[:, :, None, None], fallback, last_epoch)

        hp_df = _sanitize_hyperparameters(bb.hyperparameters)
        # Identify configs that are NaN across ALL seeds and ALL epochs
        # single_obj shape: (H, S, E, 1)

        invalid_config_mask = np.all(np.isnan(single_obj), axis=(1, 2, 3))  # (H,)

        num_invalid = invalid_config_mask.sum()

        if num_invalid > 0:
            logger.warning(
                "Task '%s': %d completely invalid hyperparameter configs detected",
                task_key,
                num_invalid,
            )

            # Extract the failing configs
            failed_configs = hp_df.loc[invalid_config_mask].copy()

            # Add index for traceability
            failed_configs["config_index"] = failed_configs.index

            # Optional: sort for readability
            failed_configs = failed_configs.sort_index()

            print(f"\n=== Failed configs for task: {task_key} ===")
            print(failed_configs.head(20))  # preview

        # ── Cross-architecture: project to shared hyperparameters ─────────────
        if not same_arch:
            hp_df, projected_cs = _project_to_target_space(
                hp_df, bb.configuration_space, target_config_space
            )
            if hp_df.empty:
                logger.warning("Skipping source '%s': no shared hyperparameters.", task_key)
                continue
            config_space = projected_cs
            logger.info(
                "Cross-architecture source '%s': using %d / %d shared HP dimensions. "
                "Unconstrained (target-only) dims will be sampled freely: %s",
                task_key,
                len(projected_cs),
                len(target_config_space),
                sorted(set(target_config_space) - set(projected_cs)),
            )
        else:
            config_space = bb.configuration_space

        transfer_evals[task_key] = TransferLearningTaskEvaluations(
            hyperparameters=hp_df,
            configuration_space=config_space,
            objectives_evaluations=last_epoch,
            objectives_names=[metric],
        )

    logger.info(
        "Transfer source tasks (%d): %s", len(transfer_evals), sorted(transfer_evals)
    )
    return transfer_evals