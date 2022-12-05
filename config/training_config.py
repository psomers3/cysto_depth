from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *


@dataclass
class TrainerDictConfig:
    """ Initial settings for PyTorch Lightning Trainer """
    accelerator: str = "auto"
    """ Hardware accelerator to use. Is typically set to one of [gpu, cpu, auto] """
    devices: int = 1
    """ leave as 1 so dataset validation is only checked from one device """
    gpus: List[int] = field(default_factory=lambda: [])
    """ specify list of gpus to use. defaults to none """
    strategy: Union[str, None] = 'ddp'
    """ leave default even for 1 node because matplotlib is used during training """
    log_every_n_steps: int = 50
    """ how often to log during training """


@dataclass
class CallbackConfig:
    """ Configuration for callbacks to be added to the training """

    early_stop_patience: Union[int, None] = None
    """ patience for early stopping. If null, then no early stopping applied. """
    early_stop_metric: str = '${..monitor_metric}'
    """ metric for early stopping"""
    ckpt_metric: str = '${..monitor_metric}'
    """ metric for model checkpoints """
    ckpt_save_top_k: int = 5
    """ keep the top k saved checkpoints """
    ckpt_every_n_epochs: Union[int, None] = None
    """ Number of epochs between checkpoints """
    model_ckpt_save_k: Union[int, None] = None
    """ keep the top k saved checkpoints """


@dataclass
class SyntheticTrainingConfig:
    """ Hyperparameter settings for the supervised synthetic depth training """

    data_roles: List[str] = field(default_factory=lambda: ['color', 'depth'])
    """ The names to use for each type of data to be loaded """
    data_directories: List[str] = MISSING
    """ The directories corresponding to the data for each data role in  data_roles """
    training_split: dict = field(default_factory=lambda: {'train': .6,
                                                          'validate': .3,
                                                          'test': .1})
    """ The entry to control generation of the data split for training. See split option for the FileLoadingDataModule 
    for valid entries """
    training_split_file: str = ''
    """ An existing training split json file. If not empty, will be used instead of training_split """
    lr: float = 1e-3
    """ learning rate for optimizer """
    optimizer: str = 'adam'
    """ Which torch optimizer to use. """
    grad_loss_factor: float = 1.0
    lr_scheduler_patience: int = 10
    lr_scheduler_monitor: str = "val_rmse_log"
    reduce_lr_patience: int = 5
    max_epochs: int = 10
    monitor_metric: str = 'val_rmse'
    """ main metric to track for performance """
    val_check_interval: int = 1
    accumulate_grad_batches: int = 4
    """ how many batches to include before gradient update """
    batch_size: int = 32
    resume_from_checkpoint: Union[str, None] = None
    """ checkpoint to load weights from """
    callbacks: CallbackConfig = CallbackConfig(early_stop_patience=15,
                                               ckpt_save_top_k=5)


@dataclass
class GANTrainingConfig:
    """ Hyperparameter settings for the domain adaptation GAN training """

    source_images: str = MISSING
    """ path to synthetically generated images """
    synth_split: dict = field(default_factory=lambda: {'train': .8,
                                                       'validate': .1,
                                                       'test': .1})
    """ The entry to control generation of the data split for training. """
    training_split_file: str = ''
    """ An existing training split json file. If not empty, will be used instead of training_split """
    generator_lr: float = 5e-6
    """ learning rate for generator """
    discriminator_lr: float = 5e-5
    """ learning rate for discriminator """
    max_epochs: int = 10
    monitor_metric: str = 'g_loss'
    """ main metric to track for performance """
    val_check_interval: int = 1
    """ how many batches before doing validation update """
    accumulate_grad_batches: int = 4
    """ how many batches to include before gradient update """
    batch_size: int = 16
    synthetic_base_model: str = MISSING
    """ The pretrained network to load weights from """
    resume_from_checkpoint: Union[str, None] = None
    """ checkpoint to load weights from """
    generate_data: bool = False
    """ Whether to process the video data folder and generate training images in the image_output_folder """
    videos_folder: str = MISSING
    """ folder with endoscopic videos """
    image_output_folder: str = MISSING
    """ folder containing (or will contain) the generated real image training data """
    callbacks: CallbackConfig = CallbackConfig(ckpt_every_n_epochs=2,
                                               ckpt_save_top_k=1,
                                               model_ckpt_save_k=None)
    beta_1: float = 0.5
    beta_2: float = 0.999
    residual_loss_factor: float = 5
    scale_loss_factor: float = 0
    img_discriminator_factor: float = 0
    residual_transfer: bool = True
    d_max_conf: float = 0.9
    warmup_steps: float = 0


@dataclass
class CystoDepthConfig:
    """ Configuration for training synthetic depth and/or domain transfer for real cystoscopic videos"""

    mode: str = 'synthetic'
    """ Mode can be one of ['synthetic', 'gan'] """
    training_stage: str = 'train'
    """ Training_stage can be one of ['train', 'test'] """
    log_directory: str = './logs'
    """ tensorboard log directory """
    adaptive_gating: bool = False
    """ Whether to turn on adaptive gating for domain adaptation """
    image_gan: bool = False
    """ Whether uses full output for discriminator instead of patches """
    num_workers: int = 6
    """ Number of workers to use during data loading """
    image_size: int = 256
    """ Final square size to make all images """
    print_config: bool = False
    """ Print full Omega Config """
    split_save_dir: str = ''
    """ Directory to save the data split(s) used during training """
    synthetic_config: SyntheticTrainingConfig = SyntheticTrainingConfig()
    gan_config: GANTrainingConfig = GANTrainingConfig()
    trainer_config: TrainerDictConfig = TrainerDictConfig()

