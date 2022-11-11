from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *


@dataclass
class GeneralTrainingConfig:
    """ Configuration options shared between supervised and GAN training"""

    num_workers: int = 6
    """ Number of workers to use during data loading """
    image_size: int = 256
    """ Final square size to make all images """
    training_split: dict = field(default_factory=lambda: {'train': ".*train.*",
                                                          'validate': ".*val.*",
                                                          'test': ".*test.*"})
    """ The entry to control generation of the data split for training. See split option for the FileLoadingDataModule 
    for valid entries """
    training_split_file: str = ''
    """ An existing training split json file """


@dataclass
class CystoDepthConfig:
    """ Configuration for training synthetic depth and/or domain transfer for real cystoscopic videos"""

    data_roles: List[str] = field(default_factory=lambda: ['color', 'depth'])
    """ The names to use for each type of data to be loaded """
    data_directories: List[str] = MISSING
    """ The directories corresponding to the data for each data role in  data_roles """
    mode: str = 'synthetic'
    """ Mode can be one of ['synthetic', 'gan'] """
    training_stage: str = 'train'
    """ Training_stage can be one of ['train', 'test'] """
    gpu: int = -1
    """ Specify single gpu to use. Defaults first one found """
    general_train_config: GeneralTrainingConfig = GeneralTrainingConfig()

