# Notice

> **Notice:** This repository contains the exact code and scripts necessary to reproduce our paper submission and links to result data on figshare. It serves as a static archive to ensure scientific transparency and **is not actively maintained**.
---

### 📊 Data & Active Maintenance

* **Active Repositories:** All maintained code, updates, and extended documentation can be found at **[Syne Tune](https://github.com/syne-tune/syne-tune)** and **[Autoencodix](https://github.com/jan-forest/autoencodix_package)**.
* **Data Access:** The complete, Syne Tune compatible blackboxes are hosted and accessible via **[Hugging Face](https://huggingface.co/datasets/synetune/blackbox-repository)**.
* **Raw Data** The raw experiment data stored as JSON files is available via **[Figshare](https://figshare.com/s/c9da074d5c0b371ea3bd?file=64553808)**. 

If you are looking to build upon this work or use the dataset for your own projects, we highly recommend cloning or forking the maintained repositories instead of this archive.

---

### 🔬 Citation
If you use the code or data from this reproducibility package in your research, please cite our paper:

```bibtex
@article{thale2026,
  author    = {Thale-Bombien, Luca and Ewald, Jan and Koenig, Ralf and Klein, Aaron},
  title     = {BBOmix: A Tabular Benchmark for Hyperparameter Optimization of Unsupervised Biological Representation Learning},
  journal   = {arXiv preprint arXiv:},
  year      = {2026}
```

## # BBOmix End-to-End Reproducibility Guide
This meta-guide provides complete instructions to reproduce the experiments for the BBOmix benchmark paper. The pipeline is divided into two distinct phases, corresponding to the two codebases used in the project:

1. **Phase 1: Autoencodix** - Generating the raw experiment data by training autoencoders across various datasets, architectures, and modalities.
2. **Phase 2: Syne Tune** - Importing the generated raw data into blackboxes, running hyperparameter optimization (HPO) algorithms, and generating the final analysis figures.

## Submitted Files Overview

The provided `.zip` archive contains the following components to facilitate the reproduction of the BBOmix benchmark:

- `META_REPRODUCIBILITY_GUIDE.md` (this file): The end-to-end instructions for reproducing the paper's results. Note, that both provided repositories contain further instructions, if needed.
- `autoencodix.zip`: The source code repository for the Autoencodix package, used to train the models and generate the benchmark (Phase 1).
- `syne-tune.zip`: The source code repository for the Syne Tune framework, tailored for BBOmix HPO evaluation (Phase 2).
- **Raw HPO Experiment Data and Syne Tune Blackboxes**: Can be downloaded from via figshare, see above.
---

## Phase 1: Autoencodix (Data Generation)

This section contains the scripts and configurations required to reproduce the large-scale benchmark BBOmix using the Autoencodix package. The codebase is designed to be easily reproducible on any machine.

### Installation

To run the benchmarking experiments, you must first install the `autoencodix` package and its dependencies. The project requires **Python >=3.9, <3.13**.

#### Using uv (Recommended)

Since this project includes a `uv.lock` file, the fastest and most reliable way to create an identical environment is using [uv](https://github.com/astral-sh/uv). From the root of the repository:

```bash
# Create a virtual environment and sync dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

#### Using standard pip

Alternatively, you can install the package using standard `pip` in your preferred virtual environment (e.g., conda or venv). From the root of the repository:

```bash
pip install -e .
```

### Getting Started

To run the benchmarking experiments, you need to configure paths to your datasets, results, and configuration directories. We use environment variables for this to prevent hardcoded absolute paths, ensuring cross-platform reproducibility.

#### Environment Variables

Before running the scripts, you can optionally configure the following environment variables:

- `AUTOENCODIX_DATA_DIR`: The directory containing your dataset and ontology files. (Default: `./data`)
- `AUTOENCODIX_RESULTS_DIR`: The output directory where experiment results will be saved. (Default: `./results`)
- `AUTOENCODIX_BASE_CONFIG_DIR`: The directory where job configurations are generated. (Default: `./experiments`)

#### Data Download

The full experiment data (datasets and ontologies) is hosted on Zenodo. We have provided a script to automatically download and extract it to your `AUTOENCODIX_DATA_DIR`.

```bash
cd benchmarking
./download_data.sh
```

This will fetch `AutoencodixZenodoReproducibility_v2.zip` and extract it into the `./data` folder.

**Expected Files in the Data Directory:**

**Datasets (TCGA/SCHC):**
- `data_methylation_per_gene_formatted.parquet`
- `data_mrna_seq_v2_rsem_formatted.parquet`
- `data_combi_MUT_CNA_formatted.parquet`
- `data_clinical_formatted.parquet`
- `scATAC_human_cortex_formatted.parquet`
- `scRNA_human_cortex_formatted.parquet`
- `scATAC_human_cortex_clinical_formatted.parquet`

**Ontologies:**
- `chromosome_ont_lvl2.txt`
- `chromosome_ont_lvl1_ncbi.txt`
- `full_ont_lvl2_reactome.txt`
- `full_ont_lvl1_reactome.txt`
- `chromosome_ont_lvl1_ensembl.txt`
- `full_ont_lvl1_ensembl_reactome.txt`

### 1. Quick Start: Minimal Example

If you want to quickly test your setup and verify that the models train correctly, run the provided minimal example. This will generate a single job configuration for `vanillix` and run it on the downloaded data.

```bash
cd benchmarking
./minimal_example.sh
```

This script will:
1. Generate a single `.yaml` configuration in `./minimal_experiment`.
2. Run `run_experiment.py` using this configuration.
3. Save the results in the `./results` directory.

### 2. Generating Full Configurations

To run the large-scale benchmarks, first generate the configurations for all jobs across different architectures, datasets, and modalities.

```bash
export AUTOENCODIX_BASE_CONFIG_DIR="/path/to/your/experiments/dir"
./generate_all_jobs.sh
```

This will populate your `AUTOENCODIX_BASE_CONFIG_DIR` with YAML configuration files and create an `all_jobs.txt` manifest file.

### 3. Running Large-Scale Experiments

Once configs are generated, you can either run individual jobs or a batch of configurations.

#### Running a Single Experiment

```bash
export AUTOENCODIX_DATA_DIR="/path/to/your/data"
export BBOMIX_RESULTS_DIR="/path/to/your/results"

python run_experiment.py --config /path/to/specific/job_config.yaml
```

#### Running Batched Experiments

For cluster environments or sequential execution, you can pass multiple configs to `run_experiments_batched.py`.

```bash
export AUTOENCODIX_DATA_DIR="/path/to/your/data"
export BBOMIX_RESULTS_DIR="/path/to/your/results"

# Pass a list of configurations
python run_experiments_batched.py --configs /path/to/job1.yaml /path/to/job2.yaml

# Or use a batch config file
python run_experiments_batched.py --batch-config /path/to/batch_config.yaml
```

---

## Phase 2: Syne Tune (HPO Benchmarks & Analysis)

Once the raw experiment data has been generated in Phase 1, it can be used to run Hyperparameter Optimization (HPO) algorithms and evaluate performance. This phase covers importing raw experiment data into `syne-tune` blackboxes, running optimizers (including transfer learning), and generating the figures and visualizations presented in the paper.

### Dependencies and Installation

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

### 1. Importing Raw Data into Blackboxes

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

### 2. Running Optimizers on Benchmark Tasks

Once the blackboxes are imported, you can run various optimizers on the BBOmix benchmark tasks. The benchmark tasks and configurations are defined in `benchmarking/benchmarks.py`.
Note that if you want to use the blackboxes provided in the submission files, you will need to adjust the path, as Syne Tune per default looks under `~/.blackbox-repository/` or copy them to this location.

#### Single and Multi-Fidelity Optimizers
Use `benchmarking/benchmark_main.py` to evaluate standard single-fidelity and multi-fidelity HPO algorithms (such as Random Search, TPE, BORE, ASHA, BOHB, etc.).

**Example Command:**
```bash
# Run Random Search on a specific bbomix benchmark for seed 0
python -m benchmarking.benchmark_main \
    --method RS \
    --benchmark bbomix_vanillix_tcga-tcga-RNA-CLIN \
    --seed 0 \
    --n_workers 1
```

#### Transfer Learning Optimizers
To evaluate transfer learning optimizers (such as `BoundingBox`, `QuantileTransfer`, and `ZeroShot`), use `benchmarking/benchmark_transfer.py`. This script loads the previously executed runs from the same or different architectures/datasets to bootstrap the optimization process.

**Example Command:**
```bash
# Run ZeroShot transfer learning on a specific benchmark
python -m benchmarking.benchmark_transfer \
    --method ZeroShot \
    --benchmark bbomix_vanillix_tcga-tcga-RNA-CLIN \
    --seed 0 \
    --all_datasets # Optional: Include to transfer knowledge across different datasets
```

Results of the optimization runs will be logged and saved (by default under the `results/` folder or `syne-tune`'s default output directory).

#### Running on a Compute Cluster (Slurmpilot)
To reproduce the large-scale evaluation presented in the paper, experiments were distributed across a compute cluster using `slurmpilot`. The `benchmarking/launch_slurmpilot.py` script automates the generation and scheduling of SLURM jobs for multiple optimizers and benchmark tasks concurrently.

**Example Command:**
```bash
# Launch experiments on a SLURM cluster using the defined partition
python -m benchmarking.launch_slurmpilot \
    --cluster my_cluster_name \
    --partition my_partition_name \
    --experiment_tag my_benchmarks \
    --num_seeds 30 \
    --n_workers 1
```
This script will construct job definitions for the selected methods (e.g., RS, TPE, BOTorch, etc.) and all defined benchmarks, submitting them to the SLURM scheduler.

### 3. Visualizing Optimization Trajectories

After running the optimizers, you can visualize their optimization trajectories (e.g., objective vs. wallclock time) and compute average normalized regret across tasks using `benchmarking/results_analysis/show_results_all.py`.

**Example Command:**
```bash
# Generate plots from the tuning results
python benchmarking/results_analysis/show_results_all.py \
    --paths /path/to/your/results/folder \
    --x_log_scale
```
This script will produce PDF figures of the optimization trajectories, normalized regret curves and rank plots, saving them into a `figures/single-fidelity/` directory.

### 4. HPO Analysis from Raw Experiment Data

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