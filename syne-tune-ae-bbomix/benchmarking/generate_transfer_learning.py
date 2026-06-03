"""
This example serves as a test for transfer learning benchmarking setups.
It iterates over all combinations of methods and parameters and generates
a text file containing all the commands to run each experiment as a separate job.
"""
import logging
from benchmarking.benchmarks import benchmark_definitions
import sys
import itertools

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    methods = [
        "BoundingBox",
        "QuantileTransfer",
        "ZeroShot"
    ]
    
    # Run all benchmarks for all test tasks
    benchmarks = [b for b in benchmark_definitions.keys() if "bbomix" in b]

    all_architectures = ["ontix", "disentanglix", "vanillix" ,"varix"]

    print(f"Generating commands for methods {methods} on {len(benchmarks)} benchmarks for 30 seeds...")

    commands = []

    for method in methods:
        for bench in benchmarks:
            # benchmark format: bbomix-vanillix_schc-schc-RNA-CLIN
            parts = bench.split("-")
            target_arch = parts[1].split("_")[0]

            for seed in range(30):
                base_cmd_no_flags = (
                    f"python benchmarking/benchmark_transfer.py "
                    f"--method {method} --benchmark {bench} "
                    f"--seed {seed}"
                )
                
                base_cmd_cross_arch = f"{base_cmd_no_flags} --cross_arch_only"

                # Get all other architectures (exclude target)
                all_others = [arch for arch in all_architectures if arch != target_arch]
                all_others_str = " ".join(all_others)

                # --- 1. Source Architecture Configurations ---
                
                # Single source, each architecture individually
                for arch in all_others:
                    commands.append(f"{base_cmd_cross_arch} --extra_architectures {arch}")

                # All source architectures jointly
                commands.append(f"{base_cmd_cross_arch} --extra_architectures {all_others_str}")

                # Each pair of source architectures
                for pair in itertools.combinations(all_others, 2):
                    pair_str = " ".join(pair)
                    commands.append(f"{base_cmd_cross_arch} --extra_architectures {pair_str}")

                # --- 2. Same-Architecture Cross-Modality Transfer ---
                # No cross_arch_only, no extra architectures
                commands.append(f"{base_cmd_no_flags}")

                # --- 3. Cross-Dataset Transfer ---
                # Same architecture, same modality, different dataset
                commands.append(f"{base_cmd_no_flags} --cross_dataset_only --match_modality_only")

                # --- 4. Cross-Architecture Cross-Dataset Transfer ---
                # Different architecture, different dataset (all others jointly)
                commands.append(f"{base_cmd_cross_arch} --cross_dataset_only --extra_architectures {all_others_str}")

    output_file = "launch_transfer_learning_commands.txt"
    with open(output_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"Generated {len(commands)} commands and saved to '{output_file}'.")
    print("Done generating commands.")
