import math, random, yaml

def random_log(min_val, max_val):
    u = random.random()
    return math.exp(math.log(min_val) + u * (math.log(max_val) - math.log(min_val)))

import pathlib

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs" / "search_space.yaml"

def sample_hyperparams(config_path=str(DEFAULT_CONFIG_PATH), architecture=None):
    yaml_cfg = yaml.safe_load(open(config_path))  # keep full config

    cfg = {
        "epochs": yaml_cfg["fixed"]["epochs"],
        "checkpoint_interval": yaml_cfg["fixed"]["checkpoint_interval"],
        "loss_reduction": yaml_cfg["fixed"]["loss_reduction"],
        "k_filter": random.choice(yaml_cfg["search"]["k_filter"]),
        "n_layers": random.choice(yaml_cfg["search"]["n_layers"]),
        "enc_factor": random.choice(yaml_cfg["search"]["enc_factor"]),
        "learning_rate": random_log(*yaml_cfg["search"]["learning_rate"]),
        "weight_decay": random_log(*yaml_cfg["search"]["weight_decay"]),
        "batch_size": random.choice(yaml_cfg["search"]["batch_size"]),
        "drop_p": random.uniform(*yaml_cfg["search"]["drop_p"])
    }

    if architecture == "disentanglix":
        cfg["beta_mi"] = random_log(*yaml_cfg["search"]["beta_mi"])
        cfg["beta_tc"] = random_log(*yaml_cfg["search"]["beta_tc"])
        cfg["beta_dimKL"] = random_log(*yaml_cfg["search"]["beta_dimKL"])
    if architecture == "varix" or architecture == "ontix":
        cfg["beta"] = random_log(*yaml_cfg["search"]["beta"])
    if architecture != "ontix":
        cfg["latent_dim"] =  random.choice(yaml_cfg["search"]["latent_dim"])
    return cfg

def sample_hp_configs(architecture, num_hp, hp_seed):
    random.seed(hp_seed)

    hp_list = []
    for _ in range(num_hp):
        hp_list.append(sample_hyperparams(architecture=architecture))
    return hp_list

if __name__ == "__main__":
    print(sample_hp_configs("disentanglix", 3, 1))
