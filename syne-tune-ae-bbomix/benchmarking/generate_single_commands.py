import argparse
from pathlib import Path

# Use explicit sys.path modification to import definitions safely
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.baselines import Methods, methods
from benchmarking.benchmarks import benchmark_definitions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, required=False, default=1)
    parser.add_argument("--num_seeds", type=int, required=False, default=30)
    parser.add_argument("--output_file", type=str, required=False, default="commands_single_seeds.txt")

    args, _ = parser.parse_known_args()

    num_seeds = args.num_seeds
    n_workers = args.n_workers
    output_file = args.output_file

    # methods
    methods_selected = [
        #MF
        Methods.BOHB,
        Methods.ASHA,
        Methods.ASHACQR,
        Methods.ASHABORE,
        #SF
        Methods.BORE,
        Methods.CQR,
        Methods.REA,
        Methods.TPE,
        Methods.RS,
    ]

    benchmarks_selected = list(benchmark_definitions.keys())
    commands = []
    for method in methods_selected:
        assert method in methods, f"{method} not in {methods}"
        for benchmark in benchmarks_selected:
            for seed in range(num_seeds):
                # Generate one command for each seed, without --run_all_seeds 1
                cmd = f"python benchmark_main.py --method {method} --n_workers {n_workers} --benchmark {benchmark} --seed {seed}"
                commands.append(cmd)

    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated {len(commands)} commands and saved to {output_path}")
