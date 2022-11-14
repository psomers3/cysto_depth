#! /usr/bin/env python
import os.path

import hydra
from omegaconf import OmegaConf
from config.training_config import CystoDepthConfig
from simple_parsing import ArgumentParser
from typing import *
import inspect
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from models.depth_model import DepthEstimationModel
from data.depth_datamodule import EndoDepthDataModule


def get_default_args(func) -> dict:
    """
    Get expected arguments and their defaults for a function
    :param func: function to get defaults from
    :return: dictionary of function argument names and default values
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


trainer_default_arguments = get_default_args(pl.Trainer.__init__)


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def cysto_depth(cfg: CystoDepthConfig) -> None:
    config: Union[Any, CystoDepthConfig] = OmegaConf.merge(OmegaConf.structured(CystoDepthConfig()), cfg, )
    # print(OmegaConf.to_yaml(config))

    if not os.path.exists(config.log_directory):
        os.makedirs(config.log_directory)
    trainer_dict = trainer_default_arguments

    # overwrite default trainer dict values with those from default TrainerDictConfig
    [trainer_dict.update({key: val}) for key, val in config.trainer_config.items() if key in trainer_dict]

    if config.training_stage == 'train':
        if config.mode == "synthetic":
            split = config.synthetic_config.training_split if not config.synthetic_config.training_split_file else \
                    config.synthetic_config.training_split
            data_module = EndoDepthDataModule(batch_size=config.synthetic_config.batch_size,
                                              color_image_directory=config.synthetic_config.data_directories[0],
                                              depth_image_directory=config.synthetic_config.data_directories[1],
                                              split=split,
                                              image_size=config.image_size,
                                              workers_per_loader=config.general_train_config.num_workers)
            model = DepthEstimationModel(adaptive_gating=config.adaptive_gating, **config.synthetic_config)
            # overwrite trainer dict values with those from synthetic config
            [trainer_dict.update({key: val})for key, val in config.synthetic_config.items() if key in trainer_dict]

        logger = pl_loggers.TensorBoardLogger(os.path.join(config.log_directory, config.mode))
        trainer_dict.update({'logger': logger})
        trainer = pl.Trainer(**trainer_dict)

        trainer.validate(model, data_module)
        trainer.fit(model, data_module)
        trainer.test(model, data_module)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CystoDepthConfig, dest='')
    args, unknown_args = parser.parse_known_args()
    cysto_depth()