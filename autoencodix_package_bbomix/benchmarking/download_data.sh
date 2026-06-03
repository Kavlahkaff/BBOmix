#!/bin/bash

# Configuration
DATA_DIR="${AUTOENCODIX_DATA_DIR:-./data}"
ZIP_URL="https://zenodo.org/records/15518831/files/AutoencodixZenodoReproducibility_v2.zip"
ZIP_FILE="AutoencodixZenodoReproducibility_v2.zip"

echo "========================================"
echo "Downloading Autoencodix Experiment Data"
echo "========================================"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download the zip file
echo "Downloading from Zenodo..."
curl -L -o "$ZIP_FILE" "$ZIP_URL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download the data."
    exit 1
fi

# Extract the zip file into the data directory
echo "Extracting data..."
unzip -q "$ZIP_FILE" -d "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract the data."
    exit 1
fi

# Clean up the zip file
echo "Cleaning up zip file..."
rm "$ZIP_FILE"

echo "Data download and extraction complete!"
echo "Data is located in: $DATA_DIR"
