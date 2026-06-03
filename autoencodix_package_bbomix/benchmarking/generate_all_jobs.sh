#!/bin/bash

# Configuration
NUM_HP=3000
NUM_SEEDS=3
BASE_CONFIG_DIR="${AUTOENCODIX_BASE_CONFIG_DIR:-./experiments}"
MANIFEST="all_jobs.txt"

# Clear old configs to avoid mixing experiments
rm -rf "$BASE_CONFIG_DIR"
rm -f "$MANIFEST"
mkdir -p "$BASE_CONFIG_DIR"

generate_call() {
    local ARCH=$1
    local DATASET=$2
    local MODS=$3
    local ONT=$4
    
    local MOD_FOLDER=$(echo $MODS | tr ' ' '_')

    # If it's not ontix, save directly in the modality folder.
    # If it is ontix, save in a subfolder named after the ontology.
    if [[ "$ARCH" == "ontix" ]]; then
        local SUBDIR="${BASE_CONFIG_DIR}/${ARCH}/${DATASET}/${MOD_FOLDER}/${ONT}"
    else
        local SUBDIR="${BASE_CONFIG_DIR}/${ARCH}/${DATASET}/${MOD_FOLDER}"
    fi

    mkdir -p "$SUBDIR"

    echo "Generating: $ARCH | $DATASET | $MODS | Ontology: $ONT"
    
    python generate_jobs.py \
        --architecture "$ARCH" \
        --dataset "$DATASET" \
        --modalities $MODS \
        --ontology "$ONT" \
        --num-hp-configs "$NUM_HP" \
        --num-random-seeds "$NUM_SEEDS" \
        --output-dir "$SUBDIR"
}

# --- EXPERIMENTAL GRID ---
for DATASET in tcga schc; do
  for MOD in RNA DNA METH; do
    if [[ "$DATASET" == "schc" && "$MOD" == "DNA" ]]; then continue; fi
    MODS="$MOD CLIN"
    
    for ARCH in vanillix varix disentanglix; do
      generate_call "$ARCH" "$DATASET" "$MODS" "none"
    done
    
    for ONT in reactome chromosome; do
      generate_call "ontix" "$DATASET" "$MODS" "$ONT"
    done
  done

  # Multi-modalities
  if [[ "$DATASET" == "tcga" ]]; then
    MODS="RNA DNA METH CLIN"
  else
    MODS="RNA METH CLIN"
  fi

  for ARCH in vanillix varix disentanglix; do
    generate_call "$ARCH" "$DATASET" "$MODS" "none"
  done
  
  for ONT in reactome chromosome; do
    generate_call "ontix" "$DATASET" "$MODS" "$ONT"
  done
done

echo "------------------------------------------------"
echo "Creating manifest file: $MANIFEST"
find "$BASE_CONFIG_DIR" -name "*.yaml" | sort > "$MANIFEST"

TOTAL_JOBS=$(wc -l < "$MANIFEST")
echo "Total individual jobs generated: $TOTAL_JOBS"
echo "To run this on the cluster, use: #SBATCH --array=1-$TOTAL_JOBS"
echo "------------------------------------------------"
