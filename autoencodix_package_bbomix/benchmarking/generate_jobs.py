import argparse
import yaml
import pathlib as Path
from autoencodix_runner.hyperparams import sample_hp_configs

def main():
    parser = argparse.ArgumentParser()
    # Global settings for this batch of experiments
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--ontology", required=False, default=None)
    
    # Scale of experiment
    parser.add_argument("--num-hp-configs", type=int, default=3000)
    parser.add_argument("--hp-seed", type=int, default=42)
    parser.add_argument("--num-random-seeds", type=int, default=4, help="How many random seeds per HP config")
    
    parser.add_argument("--output-dir", default="./experiments")
    args = parser.parse_args()

    # 1. Sample HPs once
    print(f"Sampling {args.num_hp_configs} HPs for {args.architecture}...")
    hp_list = sample_hp_configs(args.architecture, args.num_hp_configs, args.hp_seed)

    output_dir = Path.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up ontology string if passed as "none" from bash
    effective_ontology = None if args.ontology == "none" else args.ontology

    job_counter = 0

    # 2. Loop through every combination and write a specific YAML file
    for hp_idx, hyperparams in enumerate(hp_list):
        for seed_idx in range(1, args.num_random_seeds + 1):
            
            # Construct a deterministic Run ID
            run_id = f"{args.architecture}_{args.dataset}_{seed_idx}_{'_'.join(args.modalities)}_hp{hp_idx}"
            
            job_config = {
                "run_id": run_id,
                "architecture": args.architecture,
                "dataset": args.dataset,
                "modalities": args.modalities,
                "ontology": effective_ontology,
                "seed": seed_idx,
                "hp_index": hp_idx,
                "hyperparameters": hyperparams
            }

            # CHANGE: Use run_id as the filename instead of job_{counter}
            filename = output_dir / f"{run_id}.yaml"
            
            with open(filename, "w") as f:
                yaml.dump(job_config, f, sort_keys=False)
            
            job_counter += 1

    print(f"Successfully generated {job_counter} job configuration files in {output_dir}")

if __name__ == "__main__":
    main()
