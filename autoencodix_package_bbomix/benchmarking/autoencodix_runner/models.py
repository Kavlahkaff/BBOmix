import autoencodix as acx
from autoencodix.configs import VarixConfig, VanillixConfig, DisentanglixConfig, OntixConfig

def create_model(arch, data_config, hyperparams, seed, ontologies=None, sep="\t"):
    if arch == "varix":
        cfg = VarixConfig(data_config=data_config, **hyperparams, global_seed=seed, scaling="MINMAX")
        return acx.Varix(config=cfg)
    elif arch == "vanillix":
        cfg = VanillixConfig(data_config=data_config, **hyperparams, global_seed=seed, scaling="MINMAX")
        return acx.Vanillix(config=cfg)
    elif arch == "disentanglix":
        cfg = DisentanglixConfig(data_config=data_config, **hyperparams, global_seed=seed, scaling="MINMAX")
        return acx.Disentanglix(config=cfg)
    elif arch == "ontix":
        cfg = OntixConfig(data_config=data_config, **hyperparams, global_seed=seed)
        return acx.Ontix(config=cfg, ontologies=ontologies, sep=sep)
    else:
        raise ValueError(f"Unknown architecture {arch}")
