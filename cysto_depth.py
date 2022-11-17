import hydra
from omegaconf import OmegaConf
from config.training_config import CystoDepthConfig
from simple_parsing import ArgumentParser
from typing import *


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def cysto_depth(cfg: CystoDepthConfig) -> None:
    config: Union[Any, CystoDepthConfig] = OmegaConf.merge(OmegaConf.structured(CystoDepthConfig()), cfg, )
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='specify gpu to use. defaults to all available')
    parser.add_arguments(CystoDepthConfig, dest='training_config')
    args, unknown_args = parser.parse_known_args()
    cysto_depth()