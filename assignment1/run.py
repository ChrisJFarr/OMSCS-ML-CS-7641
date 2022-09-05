import hydra
import inspect
from omegaconf import DictConfig, OmegaConf
import os

from src import services as services


@hydra.main(version_base=None, config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    funcs = inspect.getmembers(services, inspect.isfunction)
    funcs_dict = dict(funcs)
    return funcs_dict[config.func](config)

if __name__=="__main__":
    main()