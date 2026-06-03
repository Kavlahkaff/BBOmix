import subprocess
import os
import time
import itertools

# Use explicit sys.path modification to import definitions safely without running the files
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.benchmarks import benchmark_definitions
from benchmarking.baselines import methods as baseline_methods
from benchmarking.benchmark_transfer import transfer_methods

COMMANDS_FILE = "launch_experiments.txt"
LOG_DIR = "logs"

def generate_commands():
    # We want to compare Transfer methods vs Baselines.
    # Transfer methods:
    t_methods = ["BoundingBox", "QuantileTransfer", "ZeroShot"]
    
    seeds = list(range(30))
    
    # We will run these over all autoencodix benchmarks
    benchmarks = [b for b in benchmark_definitions.keys() if "autoencodix" in b]
    
    commands = []

    # 2. Transfer Learning methods
    all_architectures = ["varix", "ontix", "disentanglix",
                         #"vanillix"
                         ]
    for bench in benchmarks:
        # benchmark name format: autoencodix-varix_schc-schc-RNA-METH-CLIN
        parts = bench.split("-")
        target_arch = parts[1].split("_")[0] # e.g. "varix"
        
        other_archs = [a for a in all_architectures if a != target_arch]
        
        # Generate all permutations (combinations of any length) of other architectures
        arch_combinations = []
        for r in range(1, len(other_archs) + 1):
            arch_combinations.extend(itertools.combinations(other_archs, r))
        
        for method in t_methods:
            for seed in seeds:
                base_cmd = f"python benchmarking/benchmark_transfer.py --method {method} --benchmark {bench} --seed {seed}"
                
                # Setup 1: Modality Transfer (Same Dataset, Same Architecture)
                commands.append(f"{base_cmd}")
                
                # Setup 2: Cross-Dataset (All Datasets, Same Architecture)
                commands.append(f"{base_cmd} --all_datasets")
                
                # Setup 3: Cross-Architecture (All Datasets, All permutations of Extra Architectures)
                # To compare architecture transfer capabilities properly.
                for combo in arch_combinations:
                    extra_archs_str = " ".join(combo)
                    commands.append(f"{base_cmd} --all_datasets --extra_architectures {extra_archs_str}")
                
    with open(COMMANDS_FILE, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated {len(commands)} commands in {COMMANDS_FILE}")
    return commands


def execute_commands(commands):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    total_commands = len(commands)
    print(f"Starting sequential execution of {total_commands} commands.")
    print("-" * 60)

    for index, cmd in enumerate(commands):
        clean_cmd_name = cmd.replace("python ", "").replace("benchmarking/", "").replace("--", "").replace(" ", "_").replace("/", "_")[:80]
        log_path = os.path.join(LOG_DIR, f"{index + 1:04d}_{clean_cmd_name}.log")

        print(f"[{index + 1}/{total_commands}] Running: {cmd}")
        start_time = time.time()

        try:
            with open(log_path, "w") as log_file:
                subprocess.run(cmd, shell=True, check=True, stdout=log_file, stderr=log_file)

            elapsed = time.time() - start_time
            print(f"    Success (took {elapsed:.2f}s). Log: {log_path}")

        except subprocess.CalledProcessError:
            print(f"    ERROR: Command failed. Check {log_path} for details.")
            # Optional: break if you want to stop the whole queue on the first error
            # break

    print("-" * 60)
    print("All tasks completed.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute the generated commands sequentially.")
    args = parser.parse_args()

    commands = generate_commands()
    
    if args.execute:
        execute_commands(commands)
    else:
        print("Run with --execute to run the commands sequentially locally. "
              f"Otherwise, commands have been saved to '{COMMANDS_FILE}' and can be passed to SLURM or GNU parallel.")

if __name__ == "__main__":
    main()
