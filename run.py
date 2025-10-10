import hydra
from omegaconf import DictConfig

# import torch

from src.train import train
from src.utils import utils

# torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def run(cfg: DictConfig) -> None:

    # optional utilities. Currently only disables warnings
    utils.extras(cfg)

    if cfg.get("print_config"):
        utils.print_config(cfg, resolve=True)

    return train(cfg)


if __name__ == "__main__":
    run()
