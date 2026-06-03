# BBOmix Reproducibility Guide

This directory contains the scripts and configurations required to reproduce the large-scale benchmark BBOmix using the Autoencodix package. The codebase is designed to be easily reproducible on any machine.

## Installation

To run the benchmarking experiments, you must first install the `autoencodix` package and its dependencies. The project requires **Python >=3.9, <3.13**.

### Using uv (Recommended)

Since this project includes a `uv.lock` file, the fastest and most reliable way to create an identical environment is using [uv](https://github.com/astral-sh/uv). From the root of the repository:

```bash
# Create a virtual environment and sync dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Using standard pip

Alternatively, you can install the package using standard `pip` in your preferred virtual environment (e.g., conda or venv). From the root of the repository:

```bash
pip install -e .
```

## Getting Started

To run the benchmarking experiments, you need to configure paths to your datasets, results, and configuration directories. We use environment variables for this to prevent hardcoded absolute paths, ensuring cross-platform reproducibility.

### Environment Variables

Before running the scripts, you can optionally configure the following environment variables:

- `AUTOENCODIX_DATA_DIR`: The directory containing your dataset and ontology files. (Default: `./data`)
- `AUTOENCODIX_RESULTS_DIR`: The output directory where experiment results will be saved. (Default: `./results`)
- `AUTOENCODIX_BASE_CONFIG_DIR`: The directory where job configurations are generated. (Default: `./experiments`)

### Data Download

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

## 1. Quick Start: Minimal Example

If you want to quickly test your setup and verify that the models train correctly, run the provided minimal example. This will generate a single job configuration for `vanillix` and run it on the downloaded data.

```bash
cd benchmarking
./minimal_example.sh
```

This script will:
1. Generate a single `.yaml` configuration in `./minimal_experiment`.
2. Run `run_experiment.py` using this configuration.
3. Save the results in the `./results` directory.

## 2. Generating Full Configurations

To run the large-scale benchmarks, first generate the configurations for all jobs across different architectures, datasets, and modalities.

```bash
export AUTOENCODIX_BASE_CONFIG_DIR="/path/to/your/experiments/dir"
./generate_all_jobs.sh
```

This will populate your `AUTOENCODIX_BASE_CONFIG_DIR` with YAML configuration files and create an `all_jobs.txt` manifest file.

## 3. Running Large-Scale Experiments

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