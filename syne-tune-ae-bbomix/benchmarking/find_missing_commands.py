import argparse
import json
import re
import sys
from pathlib import Path


def get_completed_runs(results_dir: Path) -> set:
    """
    Scans the results directory for metadata.json and corresponding result files.
    Returns a set of tuples: (algorithm, benchmark, seed)
    """
    completed = set()

    for metadata_path in results_dir.rglob("metadata.json"):
        # Check if the run has written results (either .zip or raw .csv)
        results_zip = metadata_path.parent / "results.csv.zip"
        results_csv = metadata_path.parent / "results.csv"

        if not (results_zip.exists() or results_csv.exists()):
            continue

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            alg = metadata.get("algorithm")
            bench = metadata.get("benchmark")
            seed = metadata.get("seed")

            if alg is not None and bench is not None and seed is not None:
                completed.add((str(alg), str(bench), int(seed)))
        except Exception as e:
            print(f"Warning: Failed to read or parse {metadata_path}: {e}")

    return completed


def parse_command(cmd_str: str):
    """
    Parses the command string using regex to extract method, benchmark, and seed.
    Returns a tuple (method, benchmark, seed) or None if parsing fails.
    """
    method_match = re.search(r'--method\s+([^\s]+)', cmd_str)
    benchmark_match = re.search(r'--benchmark\s+([^\s]+)', cmd_str)
    seed_match = re.search(r'--seed\s+(\d+)', cmd_str)

    if method_match and benchmark_match and seed_match:
        return (method_match.group(1), benchmark_match.group(1), int(seed_match.group(1)))
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Find missing Syne Tune runs based on commands file.")
    parser.add_argument(
        "--commands_file", 
        type=str, 
        default="commands_single_seeds.txt", 
        help="File containing commands to run."
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True, 
        help="Directory containing Syne Tune results."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="missing_commands.txt", 
        help="Where to save the missing commands."
    )

    args = parser.parse_args()

    commands_file = Path(args.commands_file)
    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file)

    if not commands_file.exists():
        print(f"Error: Commands file not found at {commands_file}")
        sys.exit(1)

    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        sys.exit(1)

    print(f"Scanning results in {results_dir}...")
    completed_runs = get_completed_runs(results_dir)
    print(f"Found {len(completed_runs)} completed runs.")

    missing_commands = []
    total_commands = 0
    unparseable_commands = 0

    print(f"Analyzing commands in {commands_file}...")
    with open(commands_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_commands += 1
            run_info = parse_command(line)

            if run_info:
                # tuple is (algorithm/method, benchmark, seed)
                if run_info not in completed_runs:
                    missing_commands.append(line)
            else:
                print(f"Warning: Could not parse command: {line}")
                unparseable_commands += 1

    print(f"\n--- Summary ---")
    print(f"Total commands analyzed: {total_commands}")
    if unparseable_commands > 0:
        print(f"Unparseable commands: {unparseable_commands}")
    print(f"Missing commands: {len(missing_commands)}")

    with open(output_file, "w") as f:
        for cmd in missing_commands:
            f.write(cmd + "\n")

    print(f"\nSaved missing commands to {output_file}")


if __name__ == "__main__":
    main()
