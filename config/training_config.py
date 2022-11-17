from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *


@dataclass
class GeneralTrainingConfig:
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
    data_roles: List[str] = field(default_factory=lambda: ['color', 'depth'])
    """ The names to use for each type of data to be loaded """
    data_directories: List[str] = MISSING
    """ the directories corresponding to the data for each data role in  data_roles """
    mode: str = 'synthetic'
    """ mode can be one of ['synthetic', 'gan'] """
    training_stage: str = 'train'
    """ training_stage can be one of ['train', 'test'] """
    general_train_config: GeneralTrainingConfig = GeneralTrainingConfig()
    """ Training configuration """
