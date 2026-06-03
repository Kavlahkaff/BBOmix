#!/bin/bash

echo "========================================"
echo "Running Autoencodix Minimal Example"
echo "========================================"

# Default paths
export AUTOENCODIX_DATA_DIR="${AUTOENCODIX_DATA_DIR:-./data}"
export AUTOENCODIX_RESULTS_DIR="${AUTOENCODIX_RESULTS_DIR:-./results}"
export AUTOENCODIX_BASE_CONFIG_DIR="./minimal_experiment"

# Check if data exists
if [ ! -d "$AUTOENCODIX_DATA_DIR" ] || [ -z "$(ls -A "$AUTOENCODIX_DATA_DIR")" ]; then
    echo "Warning: Data directory '$AUTOENCODIX_DATA_DIR' not found or empty."
    echo "Please run ./download_data.sh first to fetch the dataset."
    exit 1
fi

echo "1. Generating a minimal job configuration..."
# Clean up any previous minimal config
rm -rf "$AUTOENCODIX_BASE_CONFIG_DIR"
mkdir -p "$AUTOENCODIX_BASE_CONFIG_DIR"

# Generate 1 run for vanillix on TCGA with RNA modality
python generate_jobs.py \
    --architecture vanillix \
    --dataset tcga \
    --modalities RNA CLIN\
    --ontology none \
    --num-hp-configs 1 \
    --num-random-seeds 1 \
    --output-dir "$AUTOENCODIX_BASE_CONFIG_DIR"

# Find the generated yaml file
CONFIG_FILE=$(find "$AUTOENCODIX_BASE_CONFIG_DIR" -name "*.yaml" | head -n 1)

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Failed to generate configuration file."
    exit 1
fi

echo "Configuration generated at: $CONFIG_FILE"

echo "2. Running the experiment..."
python run_experiment.py --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Experiment execution failed."
    exit 1
fi

echo "========================================"
echo "Minimal example completed successfully!"
echo "Check the results in $AUTOENCODIX_RESULTS_DIR"
