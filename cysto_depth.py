#! /usr/bin/env python

import os
import hydra
import torch
from omegaconf import OmegaConf
from config.training_config import CystoDepthConfig
from simple_parsing import ArgumentParser
from typing import *
from utils.general import get_default_args, get_callbacks
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins.environments import SLURMEnvironment
from models.depth_model import DepthEstimationModel
from models.depth_norm_model import DepthNormModel
from models.hail_mary import HailMary
from data.depth_datamodule import EndoDepthDataModule
from data.phong_datamodule import PhongDataModule
from data.general_data_module import DictDataLoaderCombine
from data.gan_datamodule import GANDataModule
from models.gan_model import GAN
import signal


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def cysto_depth(cfg: CystoDepthConfig) -> None:
    config: Union[Any, CystoDepthConfig] = OmegaConf.merge(OmegaConf.structured(CystoDepthConfig()), cfg, )
    if config.print_config:
        print(OmegaConf.to_yaml(config))

    assert config.training_stage.lower() in ['train', 'validate', 'test']
    assert config.mode.lower() in ['synthetic', 'gan', 'depthnorm', 'hail_mary']

    if not os.path.exists(config.log_directory):
        os.makedirs(config.log_directory)
    if config.torch_float_precision:
        torch.set_float32_matmul_precision(config.torch_float_precision)

    trainer_dict = get_default_args(pl.Trainer.__init__)
    if config.slurm_requeue:
        trainer_dict['plugins'] = [SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
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
                                          phong_config=config.phong_config,
                                          memorize_check=config.memorize_check,
                                          add_random_blur=config.synthetic_config.add_mask_blur,
                                          pin_memory=config.pin_dataloader_memory,
                                          seed=42)
        else:
            data_module = EndoDepthDataModule(batch_size=config.synthetic_config.batch_size,
                                              data_roles=config.synthetic_config.data_roles,
                                              data_directories=config.synthetic_config.data_directories,
                                              split=split,
                                              image_size=config.image_size,
                                              workers_per_loader=config.num_workers,
                                              depth_scale_factor=1e3,
                                              inverse_depth=config.inverse_depth,
                                              memorize_check=config.memorize_check,
                                              add_random_blur=config.synthetic_config.add_mask_blur,
                                              pin_memory=config.pin_dataloader_memory,
                                              seed=42)
        model = DepthEstimationModel(config.synthetic_config)
        [trainer_dict.update({key: val}) for key, val in config.synthetic_config.items() if key in trainer_dict]
        trainer_dict.update({'callbacks': get_callbacks(config.synthetic_config.callbacks)})
    elif config.mode == 'gan':
        split = config.gan_config.synth_split if not config.gan_config.training_split_file else \
            config.gan_config.synth_split
        data_module = GANDataModule(batch_size=config.gan_config.batch_size,
                                    color_image_directories=config.gan_config.source_images,
                                    video_directories=config.gan_config.videos_folder,
                                    generate_output_directory=config.gan_config.image_output_folder,
                                    generate_data=config.gan_config.generate_data,
                                    synth_split=split,
                                    image_size=config.image_size,
                                    workers_per_loader=config.num_workers,
                                    add_random_blur=config.add_mask_blur,
                                    pin_memory=config.pin_dataloader_memory,
                                    seed=42)
        model = GAN(synth_config=config.synthetic_config.copy(), gan_config=config.gan_config.copy())
        config.gan_config.accumulate_grad_batches = 1  # This is manually handled within the model.
        [trainer_dict.update({key: val}) for key, val in config.gan_config.items() if key in trainer_dict]
        trainer_dict.update({'callbacks': get_callbacks(config.gan_config.callbacks)})
    elif config.mode == 'depthnorm':
        split = config.depth_norm_config.training_split if not config.depth_norm_config.training_split_file else \
            config.depth_norm_config.training_split
        dataload_dict = {}
        for i in range(10):
            data_dir = []
            for j in range(len(config.depth_norm_config.data_roles)):
                if str(i) in config.depth_norm_config.data_roles[j]:
                    data_dir.append(config.depth_norm_config.data_directories[j])
            if len(data_dir) == 0:
                break
            data_module = EndoDepthDataModule(batch_size=config.depth_norm_config.batch_size,
                                              data_roles=['color', 'depth', 'normals'],
                                              data_directories=data_dir,
                                              split=split,
                                              image_size=config.image_size,
                                              workers_per_loader=config.num_workers,
                                              depth_scale_factor=1e3,
                                              inverse_depth=config.depth_norm_config.inverse_depth,
                                              memorize_check=config.memorize_check,
                                              add_random_blur=config.depth_norm_config.add_mask_blur,
                                              pin_memory=config.pin_dataloader_memory,
                                              seed=42)
            dataload_dict[i] = data_module
        data_module = DictDataLoaderCombine(dataload_dict)
        model = DepthNormModel(config.depth_norm_config.copy())
        config.depth_norm_config.accumulate_grad_batches = 1  # This is manually handled within the model.
        [trainer_dict.update({key: val}) for key, val in config.depth_norm_config.items() if key in trainer_dict]
        trainer_dict.update({'callbacks': get_callbacks(config.depth_norm_config.callbacks)})
    elif config.mode == 'hail_mary':
        split = config.depth_norm_config.training_split if not config.depth_norm_config.training_split_file else \
            config.depth_norm_config.training_split
        dataload_dict = {}
        for i in range(10):
            data_dir = []
            for j in range(len(config.depth_norm_config.data_roles)):
                if str(i) in config.depth_norm_config.data_roles[j]:
                    data_dir.append(config.depth_norm_config.data_directories[j])
            if len(data_dir) == 0:
                break
            synth_data_module = EndoDepthDataModule(batch_size=config.gan_config.batch_size,
                                                    data_roles=['color', 'depth', 'normals'],
                                                    data_directories=data_dir,
                                                    split=split,
                                                    image_size=config.image_size,
                                                    workers_per_loader=config.num_workers,
                                                    depth_scale_factor=1e3,
                                                    inverse_depth=config.gan_config.inverse_depth,
                                                    memorize_check=config.memorize_check,
                                                    add_random_blur=config.depth_norm_config.add_mask_blur,
                                                    pin_memory=config.pin_dataloader_memory)
            dataload_dict[i] = synth_data_module
        real_data_module = GANDataModule(batch_size=config.synthetic_config.batch_size * len(dataload_dict),
                                         color_image_directories=config.gan_config.source_images,
                                         video_directories=config.gan_config.videos_folder,
                                         generate_output_directory=config.gan_config.image_output_folder,
                                         generate_data=config.gan_config.generate_data,
                                         synth_split=split,
                                         image_size=config.image_size,
                                         workers_per_loader=config.num_workers,
                                         add_random_blur=config.add_mask_blur,
                                         real_only=True,
                                         pin_memory=config.pin_dataloader_memory)
        dataload_dict[len(dataload_dict)] = real_data_module
        data_module = DictDataLoaderCombine(dataload_dict)
        model = HailMary(depth_norm_config=config.depth_norm_config.copy(),
                         gan_config=config.gan_config.copy(),
                         synth_config=config.synthetic_config.copy())
        config.gan_config.accumulate_grad_batches = 1  # This is manually handled within the model.
        [trainer_dict.update({key: val}) for key, val in config.gan_config.items() if key in trainer_dict]
        trainer_dict.update({'callbacks': get_callbacks(config.gan_config.callbacks)})
    logger = pl_loggers.TensorBoardLogger(os.path.join(config.log_directory, config.mode))
    trainer_dict.update({'logger': logger})
    ckpt = trainer_dict.pop('resume_from_checkpoint')
    trainer = pl.Trainer(**trainer_dict)

    if config.split_save_dir:
        save_dir = os.path.join(config.split_save_dir, config.mode)
        os.makedirs(save_dir, exist_ok=True)
        data_module.save_split(os.path.join(save_dir, 'training_split'))

    try:
        if config.training_stage == 'train':
            trainer.fit(model, data_module, ckpt_path=ckpt)
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
