"""
hpo_analysis.py
===============
Reproduces all HPO-analysis figures for the paper.

  4.1  Reconstruction loss vs. downstream performance (correlation heatmap + scatter)
  4.2  Hyperparameter importance via Random-Forest permutation importance
  4.3  Cost of default / random configurations (performance distribution + regret)
  4.4  Loss landscape visualisation (HP-space PCA coloured by downstream score)
  4.5  Cross-modality correlation of HP rankings (transfer learning motivation)

Usage
-----
    python hpo_analysis.py --results_root /path/to/results --out_dir ./figures

Requires: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon, Rectangle
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Metric name constants (mirrored from import script) ───────────────────────

METRIC_ELAPSED_TIME = "metric_elapsed_time"
METRIC_RECON_LOSS   = "metric_valid_recon_loss"        # multi-fidelity (per epoch)
METRIC_DOWNSTREAM   = "metric_avg_ml_task_performance" # aggregate downstream proxy

# SCHC-specific downstream metrics
METRIC_AUTHOR_CELL_TYPE = "metric_author_cell_type"
METRIC_AGE_GROUP        = "metric_age_group"
METRIC_SEX              = "metric_sex"   # shared name in both datasets

# TCGA-specific downstream metrics
METRIC_CANCER_TYPE                 = "metric_cancer_type"
METRIC_SUBTYPE                     = "metric_subtype"
METRIC_ONCOTREE_CODE               = "metric_oncotree_code"
METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE = "metric_ajcc_pathologic_tumor_stage"
METRIC_GRADE                       = "metric_grade"
METRIC_PATH_N_STAGE                = "metric_path_n_stage"
METRIC_DSS_STATUS                  = "metric_dss_status"
METRIC_OS_STATUS                   = "metric_os_status"

# Per-dataset downstream metric lists (excluding elapsed time and recon loss).
# These are the classification/regression targets that were evaluated at the
# final epoch; METRIC_DOWNSTREAM is the pre-computed average across all of them.
TCGA_DOWNSTREAM_METRICS = [
    METRIC_SEX,
    METRIC_DSS_STATUS,
    METRIC_OS_STATUS,
    METRIC_PATH_N_STAGE,
    METRIC_GRADE,
    METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE,
    METRIC_CANCER_TYPE,
    METRIC_SUBTYPE,
    METRIC_ONCOTREE_CODE,
]

SCHC_DOWNSTREAM_METRICS = [
    METRIC_AUTHOR_CELL_TYPE,
    METRIC_AGE_GROUP,
    METRIC_SEX,
]

DATASET_DOWNSTREAM_METRICS: dict[str, list[str]] = {
    "tcga": TCGA_DOWNSTREAM_METRICS,
    "schc": SCHC_DOWNSTREAM_METRICS,
}

# Map metric_ name → human-readable label for figures
METRIC_LABELS: dict[str, str] = {
    METRIC_DOWNSTREAM:                  "Avg. downstream (AUC)",
    METRIC_RECON_LOSS:                  "Reconstruction loss",
    METRIC_ELAPSED_TIME:                "Runtime (s)",
    METRIC_AUTHOR_CELL_TYPE:            "Cell type (AUC)",
    METRIC_AGE_GROUP:                   "Age group (AUC)",
    METRIC_SEX:                         "Sex (AUC)",
    METRIC_CANCER_TYPE:                 "Cancer type (AUC)",
    METRIC_SUBTYPE:                     "Subtype (AUC)",
    METRIC_ONCOTREE_CODE:               "Oncotree code (AUC)",
    METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE: "AJCC stage (AUC)",
    METRIC_GRADE:                       "Grade (AUC)",
    METRIC_PATH_N_STAGE:                "Path N stage (AUC)",
    METRIC_DSS_STATUS:                  "DSS status (AUC)",
    METRIC_OS_STATUS:                   "OS status (AUC)",
}

# JSON record key → metric_ name (mirrors _METRIC_TO_RECORD_KEY in import script)
_RECORD_KEY_TO_METRIC: dict[str, str] = {
    "RUNTIME_SECONDS":              METRIC_ELAPSED_TIME,
    "AVG_ML_TASK_PERFORMANCE":      METRIC_DOWNSTREAM,
    "author_cell_type":             METRIC_AUTHOR_CELL_TYPE,
    "age_group":                    METRIC_AGE_GROUP,
    "SEX":                          METRIC_SEX,
    "CANCER_TYPE":                  METRIC_CANCER_TYPE,
    "SUBTYPE":                      METRIC_SUBTYPE,
    "ONCOTREE_CODE":                METRIC_ONCOTREE_CODE,
    "AJCC_PATHOLOGIC_TUMOR_STAGE":  METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE,
    "GRADE":                        METRIC_GRADE,
    "PATH_N_STAGE":                 METRIC_PATH_N_STAGE,
    "DSS_STATUS":                   METRIC_DSS_STATUS,
    "OS_STATUS":                    METRIC_OS_STATUS,
}

ONTOLOGY_ARCHITECTURES = {"ontix"}

# ── Hyperparameter metadata ────────────────────────────────────────────────────

SHARED_HPS = [
    "k_filter", "n_layers", "enc_factor", "batch_size",
    "learning_rate", "drop_p", "weight_decay",
]

# latent_dim is shared by all architectures EXCEPT ontix (which derives its
# latent structure from the ontology graph). Keep it separate so that HP
# columns are resolved per-architecture and ontix rows are never silently
# dropped by a dropna() over a column full of NaN.
ARCH_EXTRA_HPS = {
    "vanillix":     ["latent_dim"],
    "varix":        ["latent_dim", "beta"],
    "ontix":        ["beta"],                                    # no latent_dim
    "disentanglix": ["latent_dim", "beta_mi", "beta_tc", "beta_dimKL"],
}

HP_LABELS = {
    "k_filter":      "Input dim $D$",
    "n_layers":      "$n_{\\mathrm{layers}}$",
    "enc_factor":    "enc factor $r$",
    "batch_size":    "Batch size",
    "learning_rate": "Learning rate",
    "drop_p":        "Dropout $p$",
    "weight_decay":  "Weight decay",
    "latent_dim":    "Latent dim $d_z$",
    "beta":          r"$\beta$ (VAE)",
    "beta_mi":       r"$\beta_{mi}$",
    "beta_tc":       r"$\beta_{tc}$",
    "beta_dimKL":    r"$\beta_{\mathrm{dimKL}}$",
    "trainable_params": "Model capacity",
}

ARCH_COLORS = {
    "vanillix":     "#4C72B0",
    "varix":        "#DD8452",
    "disentanglix": "#55A868",
    "ontix":        "#C44E52",
}

DATASET_MARKERS = {"tcga": "o", "schc": "s"}

# ── Plotting style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          16,
    "axes.labelsize":     16,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "axes.titlesize":     16,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def _seed_parent_map(modalities_dir: Path, architecture: str) -> dict[str, Path]:
    if architecture not in ONTOLOGY_ARCHITECTURES:
        return {"": modalities_dir}
    subdirs = [d for d in sorted(modalities_dir.iterdir()) if d.is_dir()]
    if not subdirs or any(d.name.startswith("seed_") for d in subdirs):
        return {"": modalities_dir}
    return {d.name: d for d in subdirs}


def _estimate_params(k_filter, n_layers, enc_factor, latent_dim) -> float:
    """Rough trainable-parameter count for a symmetric FC autoencoder."""
    try:
        k  = float(k_filter)
        nl = int(n_layers)
        r  = float(enc_factor)
        lz = float(latent_dim)
    except (TypeError, ValueError):
        return np.nan
    widths = [k / (r ** i) for i in range(nl)] + [lz]
    params = sum(w_in * w_out + w_out for w_in, w_out in zip(widths[:-1], widths[1:]))
    return params * 2  # symmetric encoder + decoder


def _convergence_ratio(loss_dict: dict) -> float:
    """Ratio of final epoch loss to epoch-1 loss; < 1 means the model improved."""
    if not loss_dict or len(loss_dict) < 2:
        return np.nan
    epochs = sorted(int(k) for k in loss_dict)
    first  = float(loss_dict[str(epochs[0])])
    last   = float(loss_dict[str(epochs[-1])])
    return last / first if first != 0 else np.nan


def _dataset_prefix(dataset_name: str) -> str:
    return dataset_name.split("_")[0].lower()


def _flatten_record(
        data: dict,
        architecture: str,
        dataset: str,
        modalities: str,
        task: str,
        seed: int,
) -> dict:
    hp = data.get("HYPERPARAMETERS", {})
    loss_dict = data.get("loss_per_epoch", {})

    # 1. Get the nested performance dictionary
    # It seems to be named "PER_TASK_PERFORMANCE" in your JSON
    per_task = data.get("PER_TASK_PERFORMANCE", {})

    # Create a lowercase version of the nested dict for case-insensitive lookup
    per_task_lower = {k.lower(): v for k, v in per_task.items()}

    trainable = _estimate_params(
        hp.get("k_filter"), hp.get("n_layers"),
        hp.get("enc_factor"), hp.get("latent_dim"),
    )

    row: dict = {
        "architecture": architecture,
        "dataset": dataset,
        "modalities": modalities,
        "task": task,
        "seed": seed,
        "k_filter": hp.get("k_filter"),
        "n_layers": hp.get("n_layers"),
        "enc_factor": hp.get("enc_factor"),
        "batch_size": hp.get("batch_size"),
        "learning_rate": hp.get("learning_rate"),
        "drop_p": hp.get("drop_p"),
        "weight_decay": hp.get("weight_decay"),
        "latent_dim": hp.get("latent_dim"),
        "beta": hp.get("beta"),
        "beta_mi": hp.get("beta_mi"),
        "beta_tc": hp.get("beta_tc"),
        "beta_dimKL": hp.get("beta_dimKL"),
        "trainable_params": trainable,
    }

    # Normalize reconstruction loss by k_filter
    recon_loss = min(float(v) for v in loss_dict.values()) if loss_dict else np.nan
    try:
        k_val = float(hp.get("k_filter"))
        if k_val > 0:
            recon_loss /= k_val
    except (TypeError, ValueError):
        pass

    row[METRIC_RECON_LOSS] = recon_loss

    # 2. Extract metrics from the nested dictionary
    for record_key, metric_name in _RECORD_KEY_TO_METRIC.items():
        # Check top level (for RUNTIME/AVG) OR the nested per_task dict
        val = data.get(record_key) or per_task_lower.get(record_key.lower())

        try:
            row[metric_name] = float(val) if val is not None else np.nan
        except (ValueError, TypeError):
            row[metric_name] = np.nan

    return row


def load_all_runs(results_root: Path) -> pd.DataFrame:
    records, n_skipped = [], 0
    for arch_dir in sorted(results_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        architecture = arch_dir.name.lower()
        for dataset_dir in sorted(arch_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for modalities_dir in sorted(dataset_dir.iterdir()):
                if not modalities_dir.is_dir():
                    continue
                for ontology_suffix, seed_parent in _seed_parent_map(
                    modalities_dir, architecture
                ).items():
                    task_modalities = (
                        f"{modalities_dir.name}_{ontology_suffix}"
                        if ontology_suffix else modalities_dir.name
                    )
                    task = f"{dataset_dir.name}_{task_modalities}"
                    for seed_dir in sorted(seed_parent.iterdir()):
                        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                            continue
                        seed_str = seed_dir.name[len("seed_"):]
                        if not seed_str.isdigit():
                            continue
                        seed = int(seed_str)
                        for json_file in sorted(seed_dir.glob("*.json")):
                            try:
                                with open(json_file) as fh:
                                    data = json.load(fh)
                                records.append(_flatten_record(
                                    data, architecture, dataset_dir.name,
                                    task_modalities, task, seed,
                                ))
                            except Exception as exc:
                                n_skipped += 1
                                print(f"  [WARN] skipping {json_file}: {exc}")
    print(f"Loaded {len(records)} runs  ({n_skipped} skipped / malformed)")
    return pd.DataFrame(records)


# ── Helper utilities ───────────────────────────────────────────────────────────

def _aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average all metric columns over random seeds so each
    (architecture, task, config) is one row.
    """
    all_hps = SHARED_HPS + ["beta", "beta_mi", "beta_tc", "beta_dimKL"]
    group_cols = ["architecture", "dataset", "modalities", "task"] + all_hps
    group_cols = [c for c in group_cols if c in df.columns]

    # Average over all numeric metric columns present in the frame
    metric_cols = [
        c for c in df.columns
        if c.startswith("metric_") and c in df.select_dtypes("number").columns
    ]
    return (
        df.groupby(group_cols, dropna=False)[metric_cols]
          .mean()
          .reset_index()
    )


def _downstream_metrics_for(dataset: str) -> list[str]:
    """Return the individual downstream metric columns available for a dataset."""
    prefix = _dataset_prefix(dataset)
    return DATASET_DOWNSTREAM_METRICS.get(prefix, [])


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path)
    print(f"  saved → {path}")
    plt.close(fig)


# ── Figure 4.1 — Proxy correlation ────────────────────────────────────────────

def plot_proxy_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Panel A: Spearman-ρ heatmap (architecture × task).
             Ontix cells are split diagonally for chromosome / reactome variants.
    Panel B: Scatter of reconstruction loss vs. METRIC_DOWNSTREAM per architecture,
             coloured by dataset; includes individual per-task downstream metrics
             as additional scatter panels so the per-metric spread is visible.
    """
    agg = _aggregate_seeds(df)
    agg = agg.dropna(subset=[METRIC_RECON_LOSS, METRIC_DOWNSTREAM])

    print("\n===== DEBUG: AGG BASIC =====")
    print("agg shape:", agg.shape)
    print("architectures:", agg["architecture"].unique())
    print("tasks (first 5):", agg["task"].unique()[:5],
          "... total:", agg["task"].nunique())

    archs = sorted(agg["architecture"].unique())
    tasks = sorted(agg["task"].unique())

    def split_task(task: str) -> tuple[str, str]:
        m = re.match(r"(.*)_(chromosome|reactome)$", task)
        return (m.group(1), m.group(2)) if m else (task, "base")

    grouped: dict[str, dict[str, str]] = {}
    for t in tasks:
        base, var = split_task(t)
        grouped.setdefault(base, {})[var] = t
    base_tasks = sorted(grouped.keys())

    # ── Panel A: heatmap ──────────────────────────────────────────────────
    rho: dict[tuple, float] = {}
    for arch in archs:
        for base in base_tasks:
            for var, full_task in grouped[base].items():
                sub = agg[(agg["architecture"] == arch) & (agg["task"] == full_task)]
                if len(sub) < 5 or sub[METRIC_RECON_LOSS].nunique() <= 1 \
                        or sub[METRIC_DOWNSTREAM].nunique() <= 1:
                    rho[(arch, base, var)] = np.nan
                else:
                    val, _ = spearmanr(sub[METRIC_RECON_LOSS], sub[METRIC_DOWNSTREAM])
                    rho[(arch, base, var)] = val
                    print(f"[rho] {arch} | {full_task} | ρ={val:.3f}  n={len(sub)}")

    cmap    = plt.get_cmap("RdYlGn_r")
    norm    = mcolors.Normalize(vmin=-1, vmax=1)
    missing = (0.93, 0.93, 0.93, 1)

    # Increase base height to ensure squares and x-labels fit well
    fig_a, ax_a = plt.subplots(figsize=(max(8, len(base_tasks) * 0.95), max(5.0, len(archs) * 0.95 + 1.5)))
    
    for i, arch in enumerate(archs):
        for j, base in enumerate(base_tasks):
            val_base = rho.get((arch, base, "base"), np.nan)
            val_chr  = rho.get((arch, base, "chromosome"), np.nan)
            val_rea  = rho.get((arch, base, "reactome"), np.nan)
            is_split = np.isfinite(val_chr) or np.isfinite(val_rea)

            if is_split:
                ax_a.add_patch(Rectangle((j, i), 1, 1,
                                         facecolor=missing, edgecolor="white", lw=0.7))
                ax_a.add_patch(Polygon(
                    [(j, i + 1), (j, i), (j + 1, i + 1)],
                    facecolor=cmap(norm(val_chr)) if np.isfinite(val_chr) else missing,
                    edgecolor="white", lw=0.5,
                ))
                ax_a.add_patch(Polygon(
                    [(j, i), (j + 1, i), (j + 1, i + 1)],
                    facecolor=cmap(norm(val_rea)) if np.isfinite(val_rea) else missing,
                    edgecolor="white", lw=0.5,
                ))
                if np.isfinite(val_chr):
                    ax_a.text(j + 0.3, i + 0.7, f"{val_chr:.2f}", ha="center", va="center", fontsize=10)
                if np.isfinite(val_rea):
                    ax_a.text(j + 0.7, i + 0.3, f"{val_rea:.2f}", ha="center", va="center", fontsize=10)
            else:
                ax_a.add_patch(Rectangle(
                    (j, i), 1, 1,
                    facecolor=cmap(norm(val_base)) if np.isfinite(val_base) else missing,
                    edgecolor="white", lw=0.7,
                ))
                if np.isfinite(val_base):
                    ax_a.text(j + 0.5, i + 0.5, f"{val_base:.2f}",
                              ha="center", va="center", fontsize=14)

    ax_a.set_xlim(0, len(base_tasks))
    ax_a.set_ylim(len(archs), 0)
    ax_a.set_xticks(np.arange(len(base_tasks)) + 0.5)
    ax_a.set_yticks(np.arange(len(archs)) + 0.5)
    
    # Remove "clin" from labels and replace underscores with newlines
    clean_labels = []
    for t in base_tasks:
        t_clean = re.sub(r'(?i)_?clin_?', '_', t).strip('_')
        if not t_clean or t_clean.lower() == 'clin':
            t_clean = t.replace("clin", "").strip('_')
        clean_labels.append(t_clean.replace("_", "\n"))
        
    ax_a.set_xticklabels(clean_labels, rotation=0, ha="center")
    ax_a.set_yticklabels([a.capitalize() for a in archs])
    
    # Force cells to be square
    ax_a.set_aspect('equal')
    
    ax_a.set_title(
        "Spearman correlation: reconstruction loss ↔ downstream performance\n"
        "(Ontix: chromosome / reactome split cells)",
    )
    # Colorbar (scale) and label removed for paper insertion
    fig_a.tight_layout()
    _save(fig_a, out_dir, "fig4_1a_proxy_heatmap.pdf")

    # ── Panel B: scatter — aggregate downstream + per-task metrics ────────
    # For each architecture we show one column per dataset.
    # Rows: row 0 = aggregate METRIC_DOWNSTREAM; subsequent rows = individual metrics.
    datasets = sorted(agg["dataset"].unique())
    palette  = sns.color_palette("tab10", len(datasets))

    for arch in archs:
        arch_sub = agg[agg["architecture"] == arch]

        # Collect which individual metrics exist for each dataset
        all_individual: list[str] = []
        for dset in datasets:
            all_individual += [
                m for m in _downstream_metrics_for(dset)
                if m in arch_sub.columns and arch_sub.loc[arch_sub["dataset"] == dset, m].notna().any()
            ]
        # De-duplicate while preserving order
        seen: set[str] = set()
        individual_metrics: list[str] = []
        for m in all_individual:
            if m not in seen:
                individual_metrics.append(m)
                seen.add(m)

        plot_metrics = [METRIC_DOWNSTREAM] + individual_metrics
        nrows = len(plot_metrics)
        ncols = len(datasets)

        fig_b, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.2 * ncols, 3.5 * nrows),
            squeeze=False,
        )

        for col_idx, (dset, col) in enumerate(zip(datasets, palette)):
            dset_sub = arch_sub[arch_sub["dataset"] == dset]
            for row_idx, metric in enumerate(plot_metrics):
                ax = axes[row_idx][col_idx]
                valid = dset_sub.dropna(subset=[METRIC_RECON_LOSS, metric])

                ax.scatter(
                    valid[METRIC_RECON_LOSS], valid[metric],
                    c=[col], alpha=0.35, s=12,
                    marker=DATASET_MARKERS.get(dset.lower(), "o"),
                    rasterized=True,
                )

                if len(valid) > 10 and valid[METRIC_RECON_LOSS].nunique() > 1:
                    m_coef, b, *_ = stats.linregress(
                        valid[METRIC_RECON_LOSS], valid[metric]
                    )
                    xs = np.linspace(valid[METRIC_RECON_LOSS].min(),
                                     valid[METRIC_RECON_LOSS].max(), 100)
                    ax.plot(xs, m_coef * xs + b, color="black", lw=1.2, ls="--", alpha=0.6)
                    rho_val, p = spearmanr(valid[METRIC_RECON_LOSS], valid[metric])
                    star = "*" if p < 0.05 else ""
                    ax.set_title(f"{dset.upper()}  ρ={rho_val:.2f}{star}")
                else:
                    ax.set_title(dset.upper())

                ax.set_xlabel("Recon. loss (min)")
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

        fig_b.suptitle(
            f"{arch.capitalize()} — reconstruction loss vs. downstream metrics", y=1.01,
        )
        fig_b.tight_layout()
        _save(fig_b, out_dir, f"fig4_1b_proxy_scatter_{arch}.pdf")


# ── Figure 4.2 — HP importance ────────────────────────────────────────────────

def _hp_cols_for(arch: str, df_cols: pd.Index) -> list[str]:
    """
    Return the HP columns that are actually present AND have sufficient non-NaN
    values for a given architecture.  This is the single source of truth used
    by every importance / landscape call so ontix is never broken by latent_dim.
    """
    candidates = SHARED_HPS + ARCH_EXTRA_HPS.get(arch, [])
    return [c for c in candidates if c in df_cols]


def _fit_importance(
    sub: pd.DataFrame,
    hp_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    Fit a RF on (hp_cols → target_col), run permutation importance, and return
    (imp_mean, imp_std, used_hp_cols).  Rows with any NaN in hp_cols OR
    target_col are dropped, but only the columns that are present for this
    architecture are used — so an architecture without latent_dim never triggers
    a full-row drop just because that column is NaN.

    Returns None if fewer than 20 usable rows remain.
    """
    # Drop rows where any of the actual HP columns or the target are NaN.
    Xy = sub[hp_cols + [target_col]].dropna()
    if len(Xy) < 20:
        return None

    X = Xy[hp_cols].values
    y = Xy[target_col].values

    rf = RandomForestRegressor(
        n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    return perm.importances_mean, perm.importances_std, hp_cols


def _draw_importance_bars(
    ax: plt.Axes,
    imp_mean: np.ndarray,
    imp_std: np.ndarray,
    hp_cols: list[str],
    arch: str,
    title: str,
) -> None:
    """Render a vertical importance bar chart onto ax."""
    order  = np.argsort(imp_mean)[::-1]
    labels = [HP_LABELS.get(hp_cols[i], hp_cols[i]) for i in order]
    colors = [
        ARCH_COLORS.get(arch, "#888888") if imp_mean[i] > 0 else "#cccccc"
        for i in order
    ]
    ax.bar(
        range(len(order)), imp_mean[order],
        yerr=imp_std[order], color=colors,
        align="center", width=0.65,
        error_kw={"linewidth": 0.8, "ecolor": "grey"},
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Permutation importance\n(mean decrease in R²)")
    ax.set_title(title, color=ARCH_COLORS.get(arch, "black"))


def plot_hp_importance(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each architecture, one figure with RF permutation importance panels:
      col 0 — METRIC_DOWNSTREAM (aggregate, all datasets)
      col 1 — METRIC_RECON_LOSS (all datasets)
      col 2+ — per-dataset avg. of individual downstream metrics

    Fix: hp_cols are resolved per-architecture BEFORE any dropna(), so ontix
    (which has no latent_dim) is never silently emptied.
    """
    agg   = _aggregate_seeds(df)
    archs = sorted(agg["architecture"].unique())

    for arch in archs:
        arch_sub = agg[agg["architecture"] == arch].copy()
        hp_cols  = _hp_cols_for(arch, arch_sub.columns)
        datasets = sorted(arch_sub["dataset"].unique())

        # Build target list: aggregate metrics first, then one slot per dataset
        target_specs: list[tuple[str, str, str, pd.DataFrame]] = [
            (METRIC_DOWNSTREAM, "Avg. downstream (all datasets)", "downstream", arch_sub),
            (METRIC_RECON_LOSS,  "Reconstruction loss (all datasets)", "recon_loss", arch_sub),
        ]
        for dset in datasets:
            indiv   = [m for m in _downstream_metrics_for(dset) if m in arch_sub.columns]
            dset_df = arch_sub[arch_sub["dataset"] == dset].copy()
            avail   = [m for m in indiv if dset_df[m].notna().any()]
            if not avail:
                continue
            dset_df = dset_df.copy()
            dset_df["__target__"] = dset_df[avail].mean(axis=1)
            target_specs.append((
                "__target__",
                f"{dset.upper()} per-task avg.",
                dset.lower(),
                dset_df,
            ))

        for target_col, label, setting_name, source_df in target_specs:
            result = _fit_importance(source_df, hp_cols, target_col)
            if result is None:
                print(f"  [skip importance] {arch} → {label}: too few rows")
                continue
            
            fig, ax = plt.subplots(figsize=(0.65 * len(hp_cols) + 2.5, 5.5))
            imp_mean, imp_std, used_cols = result
            _draw_importance_bars(ax, imp_mean, imp_std, used_cols, arch,
                                  f"{arch.capitalize()} → {label}")

            fig.suptitle(f"HP importance — {arch.capitalize()}")
            fig.tight_layout()
            _save(fig, out_dir, f"fig4_2_hp_importance_{arch}_{setting_name}.pdf")


# ── Figure 4.2b — Per-dataset downstream performance distribution ─────────────

def plot_downstream_per_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    """
    One figure: split violin plots of METRIC_DOWNSTREAM by architecture,
    splitting the datasets (SCHC vs TCGA) on the left and right sides of the violin.
    """
    agg = _aggregate_seeds(df)
    agg = agg.dropna(subset=[METRIC_DOWNSTREAM, "dataset", "architecture"])
    
    # Filter only for schc and tcga just in case
    agg = agg[agg["dataset"].isin(["schc", "tcga"])]
    if agg.empty:
        return
        
    archs = sorted(agg["architecture"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=agg,
        x="architecture",
        y=METRIC_DOWNSTREAM,
        hue="dataset",
        split=True,
        inner="quartile",
        order=archs,
        hue_order=["schc", "tcga"],
        ax=ax,
        palette={"schc": "#8da0cb", "tcga": "#fc8d62"}
    )

    ax.set_xticklabels([a.capitalize() for a in archs])
    ax.set_xlabel("Architecture")
    ax.set_ylabel(METRIC_LABELS[METRIC_DOWNSTREAM])
    ax.set_title(
        "Downstream performance by architecture\n"
        "(split by dataset, higher is better)",
    )
    
    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=[l.upper() for l in labels], title="Dataset")

    fig.tight_layout()
    _save(fig, out_dir, "fig4_2b_downstream_violin_combined.pdf")


def plot_recon_loss_per_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    """
    One figure: split violin plots of METRIC_RECON_LOSS by architecture,
    splitting the datasets (SCHC vs TCGA) on the left and right sides of the violin.
    """
    agg = _aggregate_seeds(df)
    agg = agg.dropna(subset=[METRIC_RECON_LOSS, "dataset", "architecture"])
    
    # Filter only for schc and tcga just in case
    agg = agg[agg["dataset"].isin(["schc", "tcga"])]
    if agg.empty:
        return
        
    archs = sorted(agg["architecture"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=agg,
        x="architecture",
        y=METRIC_RECON_LOSS,
        hue="dataset",
        split=True,
        inner="quartile",
        order=archs,
        hue_order=["schc", "tcga"],
        ax=ax,
        palette={"schc": "#8da0cb", "tcga": "#fc8d62"}
    )

    ax.set_xticklabels([a.capitalize() for a in archs])
    ax.set_xlabel("Architecture")
    ax.set_ylabel(METRIC_LABELS[METRIC_RECON_LOSS])
    ax.set_title(
        "Reconstruction loss by architecture\n"
        "(split by dataset, lower is better)",
    )

    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=[l.upper() for l in labels], title="Dataset")

    fig.tight_layout()
    _save(fig, out_dir, "fig4_2b_recon_loss_violin_combined.pdf")


# ── Figure 4.2c — Per-objective importance ────────────────────────────────────

def plot_importance_per_objective(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For every individual downstream metric (not just the aggregate), produce
    one separate figure per architecture and objective showing HP importance
    for that specific objective.
    """
    agg   = _aggregate_seeds(df)
    archs = sorted(agg["architecture"].unique())

    for arch in archs:
        arch_sub = agg[agg["architecture"] == arch].copy()
        hp_cols  = _hp_cols_for(arch, arch_sub.columns)
        datasets = sorted(arch_sub["dataset"].unique())

        # Collect every individual metric present for any dataset of this arch
        all_objectives: list[tuple[str, str, pd.DataFrame, str]] = []
        for dset in datasets:
            dset_df = arch_sub[arch_sub["dataset"] == dset].copy()
            for metric in _downstream_metrics_for(dset):
                if metric not in dset_df.columns:
                    continue
                if dset_df[metric].notna().sum() < 20:
                    continue
                label = f"{dset.upper()} — {METRIC_LABELS.get(metric, metric)}"
                all_objectives.append((metric, label, dset_df, dset))

        if not all_objectives:
            print(f"  [skip per-obj importance] {arch}: no objectives with enough data")
            continue

        for metric, label, source_df, dset in all_objectives:
            result = _fit_importance(source_df, hp_cols, metric)
            if result is None:
                print(f"  [skip] {arch} → {label}: too few rows after dropna")
                continue
            
            fig, ax = plt.subplots(figsize=(0.65 * len(hp_cols) + 2.5, 5.5))
            imp_mean, imp_std, used_cols = result
            _draw_importance_bars(ax, imp_mean, imp_std, used_cols, arch, label)

            fig.suptitle(
                f"HP importance per objective — {arch.capitalize()}",
            )
            fig.tight_layout()
            clean_metric = metric.replace("metric_", "")
            _save(fig, out_dir, f"fig4_2c_importance_per_objective_{arch}_{dset}_{clean_metric}.pdf")


# ── Figure 4.3 — Cost of random / default configurations ──────────────────────

def plot_random_cost(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Panel A: Violin of METRIC_DOWNSTREAM per architecture.
    Panel B: Normalised regret distribution per architecture.
    Panel C: Fraction of configs within X% of the best (per architecture).
    Panel D: Per-dataset, per-individual-metric violin — shows whether the
             aggregate METRIC_DOWNSTREAM masks variance in specific tasks.
    """
    agg   = _aggregate_seeds(df)
    agg   = agg.dropna(subset=[METRIC_DOWNSTREAM])
    archs = sorted(agg["architecture"].unique())

    # ── Panel A: violin ────────────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(7, 4))
    parts = ax_a.violinplot(
        [agg.loc[agg["architecture"] == a, METRIC_DOWNSTREAM].values for a in archs],
        positions=range(len(archs)),
        showmedians=True, showextrema=True,
    )
    for body, arch in zip(parts["bodies"], archs):
        body.set_facecolor(ARCH_COLORS.get(arch, "#aaaaaa"))
        body.set_alpha(0.7)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)
    ax_a.set_xticks(range(len(archs)))
    ax_a.set_xticklabels([a.capitalize() for a in archs])
    ax_a.set_ylabel(METRIC_LABELS[METRIC_DOWNSTREAM])
    ax_a.set_title(
        "Distribution of downstream performance across configurations\n"
        "(wide violin → high HPO value)",
    )
    fig_a.tight_layout()
    _save(fig_a, out_dir, "fig4_3a_random_downstream.pdf")

    # ── Panel B: normalised regret ─────────────────────────────────────────
    fig_b, axes_b = plt.subplots(1, len(archs),
                                  figsize=(3.5 * len(archs), 3.5),
                                  sharey=False)
    axes_b = [axes_b] if len(archs) == 1 else list(axes_b)

    for ax, arch in zip(axes_b, archs):
        sub     = agg[agg["architecture"] == arch]
        regrets = []
        for task in sub["task"].unique():
            t_vals = sub[sub["task"] == task][METRIC_DOWNSTREAM].dropna()
            if len(t_vals) < 2:
                continue
            best = t_vals.max()
            regrets.append(((best - t_vals) / (best + 1e-12)).values)
        if regrets:
            all_r = np.concatenate(regrets)
            ax.hist(all_r, bins=40,
                    color=ARCH_COLORS.get(arch, "#888888"),
                    alpha=0.8, edgecolor="white", linewidth=0.3)
            med = np.median(all_r)
            ax.axvline(med, color="black", lw=1.5, ls="--",
                       label=f"median={med:.2f}")
            ax.legend(frameon=False)
        ax.set_xlabel("Normalised regret")
        ax.set_ylabel("Count")
        ax.set_title(arch.capitalize(), color=ARCH_COLORS.get(arch, "black"))

    fig_b.suptitle(
        "Normalised regret  (best − config) / best  across tasks\n"
        "(right-skewed → most random configs are far from optimal)",
    )
    fig_b.tight_layout()
    _save(fig_b, out_dir, "fig4_3b_regret.pdf")

    # ── Panel C: fraction within X% of best ───────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    thresholds  = np.linspace(0, 0.5, 200)
    for arch in archs:
        sub   = agg[agg["architecture"] == arch]
        fracs = []
        for thr in thresholds:
            task_fracs = []
            for task in sub["task"].unique():
                t_vals = sub[sub["task"] == task][METRIC_DOWNSTREAM].dropna()
                if len(t_vals) < 2:
                    continue
                best = t_vals.max()
                task_fracs.append((t_vals >= best * (1 - thr)).mean())
            fracs.append(np.mean(task_fracs) if task_fracs else np.nan)
        ax_c.plot(thresholds * 100, fracs,
                  label=arch.capitalize(),
                  color=ARCH_COLORS.get(arch, None), lw=2)
    ax_c.set_xlabel("Tolerance  (% below best configuration)")
    ax_c.set_ylabel("Fraction of configs within tolerance")
    ax_c.set_title("How easy is it to find a good configuration by random sampling?")
    ax_c.legend(frameon=False)
    ax_c.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig_c.tight_layout()
    _save(fig_c, out_dir, "fig4_3c_fraction_within_tolerance.pdf")

    # ── Panel D: per-dataset individual-metric violin ─────────────────────
    # Shows whether a good aggregate score comes at the cost of specific tasks.
    for dset in sorted(agg["dataset"].unique()):
        indiv = [
            m for m in _downstream_metrics_for(dset)
            if m in agg.columns
        ]
        if not indiv:
            continue

        dset_agg = agg[agg["dataset"] == dset].dropna(subset=indiv, how="all")
        if dset_agg.empty:
            continue

        fig_d, ax_d = plt.subplots(figsize=(max(6, len(indiv) * 1.4), 4.5))
        plot_vals = [dset_agg[m].dropna().values for m in indiv]
        vp = ax_d.violinplot(plot_vals, positions=range(len(indiv)),
                              showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_alpha(0.65)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in vp:
                vp[key].set_color("black")
                vp[key].set_linewidth(1.0)
        ax_d.set_xticks(range(len(indiv)))
        ax_d.set_xticklabels(
            [METRIC_LABELS.get(m, m) for m in indiv],
            rotation=35, ha="right",
        )
        ax_d.set_ylabel("Performance (AUC)")
        ax_d.set_title(
            f"{dset.upper()} — per-task downstream performance distribution\n"
            "(all architectures combined)",
        )
        fig_d.tight_layout()
        _save(fig_d, out_dir, f"fig4_3d_per_task_violin_{dset}.pdf")


# ── Figure 4.4 — Loss landscape ───────────────────────────────────────────────

def plot_landscape(df: pd.DataFrame, out_dir: Path) -> None:
    """
    PCA projection of the HP space, coloured by downstream performance.
    One panel per (architecture, dataset) pair.
    A smooth gradient → Bayesian optimisation will work well.
    """
    agg    = _aggregate_seeds(df)
    agg    = agg.dropna(subset=[METRIC_DOWNSTREAM])
    archs  = sorted(agg["architecture"].unique())
    dsets  = sorted(agg["dataset"].unique())

    log_hps = {
        "learning_rate", "weight_decay", "beta",
        "beta_mi", "beta_tc", "beta_dimKL", "trainable_params",
    }

    nrows = len(dsets)
    ncols = len(archs)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4 * nrows),
                              squeeze=False)

    for col_idx, arch in enumerate(archs):
        for row_idx, dset in enumerate(dsets):
            ax  = axes[row_idx][col_idx]
            sub = agg[(agg["architecture"] == arch) & (agg["dataset"] == dset)]
            hp_cols = _hp_cols_for(arch, sub.columns)
            hp_cols = [c for c in hp_cols if sub[c].notna().sum() > 5]
            clean = sub[hp_cols + [METRIC_DOWNSTREAM]].dropna()
            if len(clean) < 10:
                ax.set_visible(False)
                continue

            X_raw = clean[hp_cols].copy()
            for col in hp_cols:
                if col in log_hps:
                    X_raw[col] = np.log1p(X_raw[col].clip(lower=0))

            pca     = PCA(n_components=2, random_state=42)
            coords2 = pca.fit_transform(StandardScaler().fit_transform(X_raw.values))
            var_exp = pca.explained_variance_ratio_
            perf    = clean[METRIC_DOWNSTREAM].values
            sc      = ax.scatter(
                coords2[:, 0], coords2[:, 1],
                c=perf, cmap="viridis",
                alpha=0.5, s=10, rasterized=True,
                vmin=np.percentile(perf, 5),
                vmax=np.percentile(perf, 95),
            )
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                         label=METRIC_LABELS[METRIC_DOWNSTREAM])
            ax.set_xlabel(f"PC1 ({var_exp[0]:.0%} var)")
            ax.set_ylabel(f"PC2 ({var_exp[1]:.0%} var)")
            ax.set_title(
                f"{arch.capitalize()} | {dset.upper()}", color=ARCH_COLORS.get(arch, "black"),
            )

    fig.suptitle(
        "HP-space loss landscape (PCA projection, coloured by downstream performance)\n"
        "Smooth gradient → suitable for Bayesian optimisation", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig4_4_landscape_pca.pdf")


# ── Figure 4.5 — Transfer motivation ─────────────────────────────────────────

def plot_transfer_motivation(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Spearman-ρ between per-task HP importance vectors.
    High off-diagonal ρ → importance is stable across tasks → meta-learning likely to help.
    """
    agg   = _aggregate_seeds(df)
    archs = sorted(agg["architecture"].unique())

    for arch in archs:
        sub     = agg[agg["architecture"] == arch]
        tasks   = sorted(sub["task"].unique())
        hp_cols = _hp_cols_for(arch, sub.columns)

        imp_matrix: dict[str, np.ndarray] = {}
        for task in tasks:
            t_sub = sub[sub["task"] == task][hp_cols + [METRIC_DOWNSTREAM]].dropna()
            if len(t_sub) < 20:
                continue
            rf = RandomForestRegressor(n_estimators=100, max_features="sqrt",
                                       random_state=42, n_jobs=-1)
            rf.fit(t_sub[hp_cols].values, t_sub[METRIC_DOWNSTREAM].values)
            imp_matrix[task] = rf.feature_importances_

        if len(imp_matrix) < 2:
            print(f"  [skip] not enough tasks for {arch}")
            continue

        task_names = list(imp_matrix.keys())
        n_tasks    = len(task_names)
        rho_mat    = np.eye(n_tasks)
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                r, _ = spearmanr(imp_matrix[task_names[i]],
                                  imp_matrix[task_names[j]])
                rho_mat[i, j] = rho_mat[j, i] = r

        short = [t.replace(f"{arch}_", "").replace("_", "\n") for t in task_names]
        fig, ax = plt.subplots(figsize=(max(5, n_tasks * 0.9), max(4, n_tasks * 0.9)))
        sns.heatmap(
            rho_mat, ax=ax,
            xticklabels=short, yticklabels=short,
            cmap="RdYlGn", vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f", linewidths=0.3,
            cbar_kws={"label": "Spearman ρ of HP importances"},
            annot_kws={"size": 10},
        )
        ax.set_title(
            f"{arch.capitalize()} — Cross-task HP importance agreement\n"
            "(high ρ → HP rankings transfer across tasks)",
        )
        fig.tight_layout()
        _save(fig, out_dir, f"fig4_5_transfer_motivation_{arch}.pdf")


# ── Figure 4.6 — Individual downstream metric correlations (new) ──────────────

def plot_individual_metric_correlations(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each dataset AND each architecture, show a heatmap of Spearman-ρ
    between all individual downstream metrics across configs.
    """
    agg = _aggregate_seeds(df)
    # Critical: Remove duplicate columns that cause 'Series is ambiguous' errors
    agg = agg.loc[:, ~agg.columns.duplicated()].copy()

    datasets = sorted(agg["dataset"].unique())
    architectures = sorted(agg["architecture"].unique())

    for dset in datasets:
        for arch in architectures:
            # Filter for specific dataset and architecture
            mask = (agg["dataset"] == dset) & (agg["architecture"] == arch)
            arch_dset_agg = agg[mask]

            if arch_dset_agg.empty:
                continue

            # Identify valid metrics for this dataset
            indiv = [
                m for m in _downstream_metrics_for(dset)
                if m in arch_dset_agg.columns
            ]

            # Filter out columns that are entirely NaN for this specific arch/dset combo
            indiv = [m for m in indiv if arch_dset_agg[m].notna().any()]

            if len(indiv) < 2:
                continue

            # Prepare data and clean missing values
            dset_agg = arch_dset_agg[indiv].dropna(how="all")
            if len(dset_agg) < 5:
                continue

            rho_mat = np.full((len(indiv), len(indiv)), np.nan)

            for i, m1 in enumerate(indiv):
                for j, m2 in enumerate(indiv):
                    # Robust extraction of values
                    valid = dset_agg[[m1, m2]].dropna()
                    if len(valid) < 5:
                        continue

                    s1 = valid[m1].values
                    s2 = valid[m2].values

                    # Ensure both columns have variance
                    if np.unique(s1).size > 1 and np.unique(s2).size > 1:
                        rho, _ = spearmanr(s1, s2)
                        # Handle case where spearmanr might return a matrix
                        rho_mat[i, j] = rho[0, 1] if isinstance(rho, np.ndarray) else rho

            # --- Plotting ---
            labels = [METRIC_LABELS.get(m, m) for m in indiv]
            figsize_dim = max(5, len(indiv) * 1.1)
            fig, ax = plt.subplots(figsize=(figsize_dim + 1, figsize_dim))

            sns.heatmap(
                rho_mat, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cmap="RdYlGn", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", linewidths=0.4,
                cbar_kws={"label": "Spearman ρ"},
                annot_kws={"size": 10},
            )

            ax.set_title(
                f"{dset.upper()} ({arch.capitalize()}) — Metric Correlations\n"
                "(Low ρ indicates tasks disagree on HPO rankings)",
            )
            plt.xticks(rotation=35, ha="right")
            plt.yticks(rotation=0)
            fig.tight_layout()

            # Save with architecture name in the filename
            _save(fig, out_dir, f"fig4_6_metric_corr_{dset}_{arch}.pdf")


# ── Figure 4.9 — Dropout Analysis ─────────────────────────────────────────────

def plot_dropout_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each architecture, plot the relationship between dropout (drop_p)
    and downstream performance for every task (modality).
    Highlights the top 10% of configurations to see if the best performance
    tends to occur at drop_p -> 0.
    """
    agg = _aggregate_seeds(df)
    if "drop_p" not in agg.columns:
        print("  [skip dropout analysis] 'drop_p' column not found")
        return

    agg = agg.dropna(subset=["drop_p", METRIC_DOWNSTREAM])
    archs = sorted(agg["architecture"].unique())

    for arch in archs:
        arch_sub = agg[agg["architecture"] == arch]
        tasks = sorted(arch_sub["task"].unique())
        
        if not tasks:
            continue
            
        ncols = min(4, len(tasks))
        nrows = (len(tasks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
        
        for idx, task in enumerate(tasks):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            
            task_df = arch_sub[arch_sub["task"] == task].copy()
            if len(task_df) < 5:
                ax.set_visible(False)
                continue
                
            threshold = task_df[METRIC_DOWNSTREAM].quantile(0.90)
            top_df = task_df[task_df[METRIC_DOWNSTREAM] >= threshold]
            rest_df = task_df[task_df[METRIC_DOWNSTREAM] < threshold]
            
            # Plot the rest
            ax.scatter(
                rest_df["drop_p"], rest_df[METRIC_DOWNSTREAM], 
                alpha=0.3, s=15, color="grey", rasterized=True
            )
            # Plot the top 10%
            ax.scatter(
                top_df["drop_p"], top_df[METRIC_DOWNSTREAM], 
                alpha=0.8, s=20, color=ARCH_COLORS.get(arch, "red"), 
                label="Top 10%", rasterized=True
            )
            
            if len(task_df) > 5 and task_df["drop_p"].nunique() > 1:
                slope, intercept, *_ = stats.linregress(task_df["drop_p"], task_df[METRIC_DOWNSTREAM])
                xs = np.linspace(task_df["drop_p"].min(), task_df["drop_p"].max(), 100)
                ax.plot(xs, slope * xs + intercept, color="black", lw=1.2, ls="--", alpha=0.7)
                
                rho, p = spearmanr(task_df["drop_p"], task_df[METRIC_DOWNSTREAM])
                star = "*" if p < 0.05 else ""
                ax.set_title(f"{task}\nρ={rho:.2f}{star}", fontsize=12)
            else:
                ax.set_title(f"{task}", fontsize=12)
                
            ax.set_xlabel("Dropout (drop_p)", fontsize=12)
            ax.set_ylabel(METRIC_LABELS.get(METRIC_DOWNSTREAM, "Downstream"), fontsize=12)
            ax.legend(frameon=False, fontsize=10)
            
        for idx in range(len(tasks), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)
            
        fig.suptitle(
            f"Dropout vs Downstream Performance — {arch.capitalize()}\n"
            "(Top 10% configs highlighted to show optimal dropout range)",
            y=1.02 + (0.02 if nrows <= 2 else 0.0),
        )
        fig.tight_layout()
        _save(fig, out_dir, f"fig4_9_dropout_analysis_{arch}.pdf")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate HPO analysis figures for the benchmark paper."
    )
    p.add_argument("--results_root", type=Path, required=True,
                   help="Root of the results tree "
                        "(architecture/dataset/modality/seed_N/*.json).")
    p.add_argument("--out_dir", type=Path, default=Path("./hpo-figures"),
                   help="Output directory (default: ./hpo-figures).")
    p.add_argument(
        "--skip", nargs="*", default=[],
        choices=["proxy", "importance", "downstream_per_dataset", "recon_loss_per_dataset",
                 "importance_per_objective", "random_cost", "landscape",
                 "transfer", "metric_corr", "hp_distributions", "beta_tradeoff", "dropout"],
        help="Skip specific figure groups.",
    )
    p.add_argument("--cache", type=Path, default=None,
                   help="Parquet cache path. Loaded if it exists, written otherwise.")
    return p

# ── Figure 4.8 — Hyperparameter Distributions (new) ───────────────────────────

def plot_hp_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each architecture, plot the distribution of evaluated hyperparameter values.
    This helps visualize how evenly the HP space was explored and identifies the ranges.
    """
    archs = sorted(df["architecture"].unique())
    
    shared_hps = {
        "k_filter": ("choice", [128, 256, 512, 1024, 2048, 4096]),
        "n_layers": ("choice", [2, 3, 4]),
        "enc_factor": ("choice", [1, 2, 3, 4]),
        "batch_size": ("choice", [32, 64, 128, 256]),
        "learning_rate": ("loguniform", 1e-5, 1e-1),
        "drop_p": ("uniform", 0.0, 0.9),
        "weight_decay": ("loguniform", 1e-5, 1e-1),
    }
    arch_spaces = {
        "vanillix": {**shared_hps, "latent_dim": ("choice", [2, 4, 8, 16, 32, 64])},
        "varix": {**shared_hps, "beta": ("loguniform", 0.001, 10), "latent_dim": ("choice", [2, 4, 8, 16, 32, 64])},
        "ontix": {**shared_hps, "beta": ("loguniform", 0.0001, 1)},
        "disentanglix": {
            **shared_hps,
            "latent_dim": ("choice", [2, 4, 8, 16, 32, 64]),
            "beta_mi": ("loguniform", 0.001, 10.0),
            "beta_tc": ("loguniform", 0.1, 10000),
            "beta_dimKL": ("loguniform", 0.001, 10.0),
        },
    }

    for arch in archs:
        sub = df[df["architecture"] == arch]
        hp_cols = _hp_cols_for(arch, sub.columns)
        hp_cols = [c for c in hp_cols if sub[c].notna().sum() > 0]
        
        if not hp_cols:
            continue
            
        ncols = 4
        nrows = (len(hp_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), squeeze=False)
        
        space = arch_spaces.get(arch, {})
        
        for idx, hp in enumerate(hp_cols):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            
            vals = sub[hp].dropna()
            if len(vals) == 0:
                ax.set_visible(False)
                continue
                
            config = space.get(hp)
            color = ARCH_COLORS.get(arch, "#333333")
            
            if config is None:
                # Fallback for undefined HPs
                counts, bin_edges = np.histogram(vals, bins=20)
                ax.bar(bin_edges[:-1], counts / 3.0, width=np.diff(bin_edges), color=color, align="edge")
                min_val, max_val = vals.min(), vals.max()
                ax.set_xlabel(f"Range: [{min_val:.2g}, {max_val:.2g}]")
            else:
                ctype = config[0]
                
                if ctype == "choice":
                    choices = config[1]
                    counts = vals.value_counts()
                    for c in choices:
                        if c not in counts:
                            counts[c] = 0
                    
                    x_pos = np.arange(len(choices))
                    y_vals = [counts.get(c, 0) / 3.0 for c in choices]
                    
                    ax.bar(x_pos, y_vals, color=color, width=0.8)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([str(c) for c in choices], rotation=45 if len(choices) > 4 else 0)
                    ax.set_xlabel(f"Choices: {len(choices)}")
                    
                elif ctype == "loguniform":
                    lower, upper = config[1], config[2]
                    # Bins exactly from bounds
                    bins = np.logspace(np.log10(lower), np.log10(upper), 21)
                    counts, bin_edges = np.histogram(vals, bins=bins)
                    ax.bar(bin_edges[:-1], counts / 3.0, width=np.diff(bin_edges), color=color, align="edge")
                    ax.set_xscale("log")
                    ax.set_xlim(lower, upper)
                    ax.set_xlabel(f"Range: [{lower:.1e}, {upper:.1e}]")
                    
                elif ctype == "uniform":
                    lower, upper = config[1], config[2]
                    bins = np.linspace(lower, upper, 21)
                    counts, bin_edges = np.histogram(vals, bins=bins)
                    ax.bar(bin_edges[:-1], counts / 3.0, width=np.diff(bin_edges), color=color, align="edge")
                    ax.set_xlim(lower, upper)
                    ax.set_xlabel(f"Range: [{lower:.2g}, {upper:.2g}]")
                    
            ax.set_title(HP_LABELS.get(hp, hp))
            ax.set_ylabel("Count")
            
        for idx in range(len(hp_cols), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)
            
        fig.suptitle(f"Hyperparameter Distributions — {arch.capitalize()}")
        fig.tight_layout()
        _save(fig, out_dir, f"fig4_8_hp_distributions_{arch}.pdf")


def plot_disentanglix_beta_vs_downstream(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter plots for disentanglix only:
    beta_mi, beta_tc, and beta_dimKL vs. avg downstream performance.
    One panel per beta parameter, coloured by dataset.
    """
    agg = _aggregate_seeds(df)
    agg = agg[agg["architecture"] == "disentanglix"].copy()

    if agg.empty:
        print("  [skip beta tradeoff] disentanglix: no runs found")
        return

    beta_cols = [
        c for c in ["beta_mi", "beta_tc", "beta_dimKL"]
        if c in agg.columns and agg[c].notna().any()
    ]
    if not beta_cols:
        print("  [skip beta tradeoff] disentanglix: no beta columns with data")
        return

    datasets = sorted(agg["dataset"].dropna().unique())
    palette = sns.color_palette("tab10", len(datasets))

    fig, axes = plt.subplots(
        1, len(beta_cols),
        figsize=(5.2 * len(beta_cols), 4.2),
        squeeze=False,
    )

    for ax, beta_col in zip(axes[0], beta_cols):
        plot_df = agg.dropna(subset=[beta_col, METRIC_DOWNSTREAM]).copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        for dset, color in zip(datasets, palette):
            dset_df = plot_df[plot_df["dataset"] == dset]
            if dset_df.empty:
                continue
            ax.scatter(
                dset_df[beta_col],
                dset_df[METRIC_DOWNSTREAM],
                alpha=0.45,
                s=18,
                color=color,
                marker=DATASET_MARKERS.get(dset.lower(), "o"),
                label=dset.upper(),
                rasterized=True,
            )

        if len(plot_df) > 10 and plot_df[beta_col].nunique() > 1:
            slope, intercept, *_ = stats.linregress(
                plot_df[beta_col], plot_df[METRIC_DOWNSTREAM]
            )
            xs = np.linspace(plot_df[beta_col].min(), plot_df[beta_col].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="black", lw=1.2, ls="--", alpha=0.7)

            rho, p = spearmanr(plot_df[beta_col], plot_df[METRIC_DOWNSTREAM])
            ax.set_title(
                f"{HP_LABELS.get(beta_col, beta_col)}  |  ρ={rho:.2f}" + ("*" if p < 0.05 else ""),
            )
        else:
            ax.set_title(HP_LABELS.get(beta_col, beta_col))

        # Ensure strictly positive values for log scale
        min_positive = plot_df[beta_col][plot_df[beta_col] > 0].min()
        if pd.notna(min_positive):
            ax.set_xscale("log")
        else:
            # fallback: shift if zeros exist (rare but safe)
            shifted = plot_df[beta_col] + 1e-8
            ax.set_xscale("log")

        ax.set_xlabel(HP_LABELS.get(beta_col, beta_col) + " (log scale)")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylabel(METRIC_LABELS[METRIC_DOWNSTREAM])
        ax.legend(frameon=False)

    fig.suptitle(
        "Disentanglix β values vs. average downstream performance",
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig4_7_disentanglix_beta_vs_downstream.pdf")


def main() -> None:
    args = _build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache and args.cache.exists():
        print(f"Loading from cache: {args.cache}")
        df = pd.read_parquet(args.cache)
    else:
        df = load_all_runs(args.results_root)
        if args.cache:
            df.to_parquet(args.cache, index=False)
            print(f"Cached to {args.cache}")

    print(f"\nDataset shape : {df.shape}")
    print(f"Architectures : {sorted(df['architecture'].unique())}")
    print(f"Datasets      : {sorted(df['dataset'].unique())}")
    print(f"Tasks         : {sorted(df['task'].unique())}")
    print(f"Seeds         : {sorted(df['seed'].unique())}\n")

    skip = set(args.skip)

    if "proxy" not in skip:
        print("─── 4.1  Proxy correlation …")
        plot_proxy_correlation(df, args.out_dir)

    if "importance" not in skip:
        print("─── 4.2  HP importance …")
        plot_hp_importance(df, args.out_dir)

    if "downstream_per_dataset" not in skip:
        print("─── 4.2b Per-dataset downstream violin …")
        plot_downstream_per_dataset(df, args.out_dir)

    if "recon_loss_per_dataset" not in skip:
        print("─── 4.2b_recon Per-dataset recon loss violin …")
        plot_recon_loss_per_dataset(df, args.out_dir)

    if "importance_per_objective" not in skip:
        print("─── 4.2c HP importance per objective …")
        plot_importance_per_objective(df, args.out_dir)

    if "random_cost" not in skip:
        print("─── 4.3  Cost of random configs …")
        plot_random_cost(df, args.out_dir)

    if "landscape" not in skip:
        print("─── 4.4  Loss landscape …")
        plot_landscape(df, args.out_dir)

    if "transfer" not in skip:
        print("─── 4.5  Transfer motivation …")
        plot_transfer_motivation(df, args.out_dir)

    if "metric_corr" not in skip:
        print("─── 4.6  Individual metric correlations …")
        plot_individual_metric_correlations(df, args.out_dir)

    if "beta_tradeoff" not in skip:
        print("─── 4.7  Disentanglix beta trade-off …")
        plot_disentanglix_beta_vs_downstream(df, args.out_dir)

    if "hp_distributions" not in skip:
        print("─── 4.8  Hyperparameter distributions …")
        plot_hp_distributions(df, args.out_dir)

    if "dropout" not in skip:
        print("─── 4.9  Dropout analysis …")
        plot_dropout_analysis(df, args.out_dir)

    print("\n✓ All figures saved to", args.out_dir)


if __name__ == "__main__":
    main()