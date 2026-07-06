import os
import subprocess
import numpy as np
import pandas as pd
import tempfile
import shutil

print("========================================")
print("Running Autoencodix Minimal Mock Example")
print("========================================")

# Create a temporary directory for mock data
mock_data_dir = tempfile.mkdtemp()
print(f"Created mock data directory at: {mock_data_dir}")

try:
    # 1. Create mock RNA data (numerical)
    # We need a small number of features and samples
    num_samples = 100
    num_rna_features = 50
    samples = [f"Sample_{i}" for i in range(num_samples)]
    
    df_rna = pd.DataFrame(
        np.random.rand(num_samples, num_rna_features),
        index=samples,
        columns=[f"Gene_{i}" for i in range(num_rna_features)]
    )
    rna_path = os.path.join(mock_data_dir, "data_mrna_seq_v2_rsem_formatted.parquet")
    df_rna.to_parquet(rna_path)
    print(f"Mock RNA data saved to {rna_path}")

    # 2. Create mock CLINICAL data (categorical/annotations)
    # Using the TCGA targets evaluated in evaluation.py
    clin_cols = [
        "CANCER_TYPE", "SUBTYPE", "ONCOTREE_CODE", "SEX",
        "AJCC_PATHOLOGIC_TUMOR_STAGE", "GRADE", "PATH_N_STAGE",
        "DSS_STATUS", "OS_STATUS"
    ]
    
    # We need enough categories so sklearn stratification doesn't fail.
    df_clin = pd.DataFrame({
        "CANCER_TYPE": np.random.choice(["TypeA", "TypeB", "TypeC"], num_samples),
        "SUBTYPE": np.random.choice(["Sub1", "Sub2", "Sub3"], num_samples),
        "ONCOTREE_CODE": np.random.choice(["CodeX", "CodeY"], num_samples),
        "SEX": np.random.choice(["M", "F"], num_samples),
        "AJCC_PATHOLOGIC_TUMOR_STAGE": np.random.choice(["I", "II", "III"], num_samples),
        "GRADE": np.random.choice(["G1", "G2", "G3"], num_samples),
        "PATH_N_STAGE": np.random.choice(["N0", "N1"], num_samples),
        "DSS_STATUS": np.random.choice(["0", "1"], num_samples),
        "OS_STATUS": np.random.choice(["0", "1"], num_samples),
    }, index=samples)
    
    clin_path = os.path.join(mock_data_dir, "data_clinical_formatted.parquet")
    df_clin.to_parquet(clin_path)
    print(f"Mock CLINICAL data saved to {clin_path}")

    # 3. Setup environment variables for the pipeline
    os.environ["AUTOENCODIX_DATA_DIR"] = mock_data_dir
    os.environ["AUTOENCODIX_RESULTS_DIR"] = "./mock_results"
    
    config_dir = "./mock_experiment_config"
    os.environ["AUTOENCODIX_BASE_CONFIG_DIR"] = config_dir

    # 4. Run the minimal example script
    # It will use the environment variables we just set
    print("\nExecuting minimal_example.sh with mock data...")
    script_path = os.path.join(os.path.dirname(__file__), "minimal_example.sh")
    
    # Call minimal_example.sh directly
    result = subprocess.run(
        ["bash", script_path], 
        env=os.environ,
        cwd=os.path.dirname(__file__)
    )

    if result.returncode == 0:
        print("\n========================================")
        print("Mock Example Completed Successfully!")
        print("The end-to-end evaluation pipeline works without a ValueError.")
        print("========================================")
    else:
        print("\n========================================")
        print("Mock Example Failed.")
        print("========================================")
        exit(1)

finally:
    # Cleanup mock data directory
    shutil.rmtree(mock_data_dir)
    print(f"Cleaned up mock data directory: {mock_data_dir}")
