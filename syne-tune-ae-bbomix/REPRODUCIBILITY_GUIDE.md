# BBOmix HPO Benchmarks: Reproducibility Guide

This guide explains how to reproduce the experiments for the BBOmix benchmark paper. It covers importing raw experiment data into `syne-tune` blackboxes, running hyperparameter optimization (HPO) algorithms (including transfer learning), and generating the figures and visualizations presented in the paper.

## Dependencies and Installation

Before running the scripts, you must install the required dependencies.
You can install the package and its dependencies locally using `uv` or `pip`:

```bash
# Using uv (recommended for speed):
uv pip install -e .

# Or using pip:
pip install -e .
```

**Note:** If you plan to launch experiments on a compute cluster using `benchmarking/launch_slurmpilot.py`, you will also need to manually install `slurmpilot` since it is not included in the default `syne-tune` dependencies:
```bash
uv pip install slurmpilot # or pip install slurmpilot
```

## 1. Importing Raw Data into Blackboxes

The raw experiment results must first be converted into Tabular Blackboxes compatible with the `syne-tune` framework. This is handled by `syne_tune/blackbox_repository/conversion_scripts/scripts/bbomix_import.py`.

This script loads the raw JSON run files, extracts hyperparameter configurations and metrics. It then serializes these into a format that `syne-tune` can use for rapid simulated evaluations.

**How to run it:**
If you have the raw JSON experiment data downloaded on your machine, you can run the import script to generate the blackboxes locally.

```bash
# Modify the script or pass your custom path to the `generate_bbomix_from_json` method if necessary.
# By default, it looks for the data at the path defined by RESULTS_ROOT in the script.
# Edit `RESULTS_ROOT` in `BBOmix_import.py` to point to your raw JSON directory, then run:

python syne_tune/blackbox_repository/conversion_scripts/scripts/bbomix_import.py
```
This process will create the necessary blackbox files in your local syne-tune blackbox repository (typically under `~/.blackbox-repository/`).

## 2. Running Optimizers on Benchmark Tasks

Once the blackboxes are imported, you can run various optimizers on the BBOmix benchmark tasks. The benchmark tasks and configurations are defined in `benchmarking/benchmarks.py`.

### Single and Multi-Fidelity Optimizers
Use `benchmarking/benchmark_main.py` to evaluate standard single-fidelity and multi-fidelity HPO algorithms (such as Random Search, TPE, BORE, ASHA, BOHB, etc.).

**Example Command:**
```bash
# Run Random Search on a specific bbomix benchmark for seed 0
python benchmarking/benchmark_main.py \
    --method RS \
    --benchmark bbomix-vanillix_tcga-tcga-RNA-CLIN \
    --seed 0 \
    --n_workers 1
```

### Transfer Learning Optimizers
To evaluate transfer learning optimizers (such as `BoundingBox`, `QuantileTransfer`, and `ZeroShot`), use `benchmarking/benchmark_transfer.py`. This script loads the previously executed runs from the same or different architectures/datasets to bootstrap the optimization process.

**Example Command:**
```bash
# Run ZeroShot transfer learning on a specific benchmark
python benchmarking/benchmark_transfer.py \
    --method ZeroShot \
    --benchmark bbomix-vanillix_tcga-tcga-RNA-CLIN \
    --seed 0 \
    --all_datasets # Optional: Include to transfer knowledge across different datasets
```

Results of the optimization runs will be logged and saved (by default under the `results/` folder or `syne-tune`'s default output directory).

### Running on a Compute Cluster (Slurmpilot)
To reproduce the large-scale evaluation presented in the paper, experiments were distributed across a compute cluster using `slurmpilot`. The `benchmarking/launch_slurmpilot.py` script automates the generation and scheduling of SLURM jobs for multiple optimizers and benchmark tasks concurrently.

**Example Command:**
```bash
# Launch experiments on a SLURM cluster using the defined partition
python benchmarking/launch_slurmpilot.py \
    --cluster my_cluster_name \
    --partition my_partition_name \
    --experiment_tag my_benchmarks \
    --num_seeds 30 \
    --n_workers 1
```
This script will construct job definitions for the selected methods (e.g., RS, TPE, BOTorch, etc.) and all defined benchmarks, submitting them to the SLURM scheduler.

## 3. Visualizing Optimization Trajectories

After running the optimizers, you can visualize their optimization trajectories (e.g., objective vs. wallclock time) and compute average normalized regret across tasks using `benchmarking/results_analysis/show_results_all.py`.

**Example Command:**
```bash
# Generate plots from the tuning results
python benchmarking/results_analysis/show_results_all.py \
    --paths /path/to/your/results/folder \
    --x_log_scale
```
This script will produce PDF figures of the optimization trajectories, normalized regret curves and rank plots, saving them into a `figures/single-fidelity/` directory.

## 4. HPO Analysis from Raw Experiment Data

To recreate the HPO analysis figures from the paper directly from the raw experiment data (without running `syne-tune` simulations), you can use `benchmarking/hpo_analysis.py`.

This script reproduces figures such as:
- Reconstruction loss vs. downstream performance correlation
- Hyperparameter importance via Random-Forest permutation importance
- Cost of default/random configurations
- Loss landscape visualizations (HP-space PCA)
- Cross-modality correlation of HP rankings

**Example Command:**
```bash
# Generate the paper figures from the raw data
python benchmarking/hpo_analysis.py \
    --results_root /path/to/your/raw/BBOmix_results \
    --out_dir ./hpo-figures
```
The resulting plots will be saved as PDFs in the specified output directory (`./hpo-figures` by default).