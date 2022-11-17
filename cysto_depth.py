#! /usr/bin/env python

import os
import hydra
from omegaconf import OmegaConf
from config.training_config import CystoDepthConfig, CallbackConfig
from simple_parsing import ArgumentParser
from typing import *
import inspect
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from models.depth_model import DepthEstimationModel
from data.depth_datamodule import EndoDepthDataModule
from data.gan_datamodule import GANDataModule
from models.gan_model import GAN


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


def get_callbacks(configuration: CallbackConfig) -> List[pl.Callback]:
    callbacks = []
    if configuration.early_stop_patience:
        callbacks.append(pl.callbacks.EarlyStopping(monitor=configuration.early_stop_metric,
                                                    patience=configuration.early_stop_patience))
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor=configuration.ckpt_metric,
                                                  save_top_k=configuration.ckpt_save_top_k,
                                                  every_n_epochs=configuration.ckpt_every_n_epochs))
    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def cysto_depth(cfg: CystoDepthConfig) -> None:
    config: Union[Any, CystoDepthConfig] = OmegaConf.merge(OmegaConf.structured(CystoDepthConfig()), cfg, )
    if config.print_config:
        print(OmegaConf.to_yaml(config))
    assert config.training_stage.lower() in ['train', 'validate', 'test']
    assert config.mode.lower() in ['synthetic', 'gan']

    if not os.path.exists(config.log_directory):
        os.makedirs(config.log_directory)

    trainer_dict = get_default_args(pl.Trainer.__init__)
    [trainer_dict.update({key: val}) for key, val in config.trainer_config.items() if key in trainer_dict]

    if config.mode == "synthetic":
        split = config.synthetic_config.training_split if not config.synthetic_config.training_split_file else \
                config.synthetic_config.training_split
        data_module = EndoDepthDataModule(batch_size=config.synthetic_config.batch_size,
                                          color_image_directory=config.synthetic_config.data_directories[0],
                                          depth_image_directory=config.synthetic_config.data_directories[1],
                                          split=split,
                                          image_size=config.image_size,
                                          workers_per_loader=config.num_workers)
        model = DepthEstimationModel(adaptive_gating=config.adaptive_gating, **config.synthetic_config)
        [trainer_dict.update({key: val})for key, val in config.synthetic_config.items() if key in trainer_dict]
        trainer_dict.update({'callbacks': get_callbacks(config.synthetic_config.callbacks)})
    else:
        split = config.gan_config.synth_split if not config.gan_config.training_split_file else \
            config.gan_config.synth_split
        data_module = GANDataModule(batch_size=config.gan_config.batch_size,
                                    color_image_directory=config.gan_config.source_images,
                                    video_directory=config.gan_config.videos_folder,
                                    generate_output_directory=config.gan_config.image_output_folder,
                                    generate_data=config.gan_config.generate_data,
                                    synth_split=split,
                                    image_size=config.image_size,
                                    workers_per_loader=config.num_workers)
        model = GAN(adaptive_gating=config.adaptive_gating, image_gan=config.image_gan, **config.gan_config)
        [trainer_dict.update({key: val}) for key, val in config.gan_config.items() if key in trainer_dict]
        trainer_dict.update({'callbacks': get_callbacks(config.gan_config.callbacks)})

    logger = pl_loggers.TensorBoardLogger(os.path.join(config.log_directory, config.mode))
    trainer_dict.update({'logger': logger})
    trainer = pl.Trainer(**trainer_dict)

    if config.training_stage == 'train':
        trainer.fit(model, data_module)
    elif config.training_stage == 'validate':
        trainer.validate(model, data_module)
    else:
        trainer.test(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CystoDepthConfig, dest='')
    args, unknown_args = parser.parse_known_args()
    cysto_depth()
