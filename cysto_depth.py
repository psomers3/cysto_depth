#! /usr/bin/env python

import os
import hydra
from omegaconf import OmegaConf
from config.training_config import CystoDepthConfig
from simple_parsing import ArgumentParser
from typing import *
from utils.general import get_default_args, get_callbacks
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from models.depth_model import DepthEstimationModel
from data.depth_datamodule import EndoDepthDataModule
from data.phong_datamodule import PhongDataModule
from data.gan_datamodule import GANDataModule
from models.gan_model import GAN


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def cysto_depth(cfg: CystoDepthConfig) -> None:
    config: Union[Any, CystoDepthConfig] = OmegaConf.merge(OmegaConf.structured(CystoDepthConfig()), cfg,)
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
        if config.predict_normals:
            data_module = PhongDataModule(batch_size=config.synthetic_config.batch_size,
                                          color_image_directory=config.synthetic_config.data_directories[0],
                                          depth_image_directory=config.synthetic_config.data_directories[1],
                                          normals_image_directory=config.synthetic_config.data_directories[2],
                                          split=split,
                                          image_size=config.image_size,
                                          workers_per_loader=config.num_workers,
                                          phong_config=config.phong_config)
        else:
            data_module = EndoDepthDataModule(batch_size=config.synthetic_config.batch_size,
                                              color_image_directory=config.synthetic_config.data_directories[0],
                                              depth_image_directory=config.synthetic_config.data_directories[1],
                                              split=split,
                                              image_size=config.image_size,
                                              workers_per_loader=config.num_workers)
        model = DepthEstimationModel(config.synthetic_config)
        [trainer_dict.update({key: val}) for key, val in config.synthetic_config.items() if key in trainer_dict]
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
    trainer_dict.pop('resume_from_checkpoint')
    trainer = pl.Trainer(**trainer_dict)

    if config.split_save_dir:
        save_dir = os.path.join(config.split_save_dir, config.mode)
        os.makedirs(save_dir, exist_ok=True)
        data_module.save_split(os.path.join(save_dir, 'training_split'))

    try:
        if config.training_stage == 'train':
            trainer.fit(model, data_module)
        elif config.training_stage == 'validate':
            trainer.validate(model, data_module)
        else:
            trainer.test(model, data_module)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)

    with open(os.path.join(logger.log_dir, 'configuration.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    # parser = ArgumentParser()
    # cfg = CystoDepthConfig()
    # parser.add_arguments(CystoDepthConfig, dest='')
    # args, unknown_args = parser.parse_known_args()
    # TODO: The above code fails with missing values. Need to figure out how to get it to ignore them.
    cysto_depth()
