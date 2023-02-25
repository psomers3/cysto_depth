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
    gradient_clip_val: float = 0.0
    """ value for gradient clipping during training. Defaults to no clipping. """
    gradient_clip_algorithm: str = 'norm'
    """ type of gradient clipping to do. Either 'value' or 'norm' """
    num_nodes: int = 1
    """ Number of nodes (for SLURM or whatever). Leave this as 1 and specify the nodes in the SLURM script. """
    overfit_batches: int = 0
    """ Run training on a set number of batches to check a network can learn the task """


@dataclass
class EncoderConfig:
    """ Configuration for encoder """

    backbone: str = 'resnet18'
    """ Base model to use for the encoder [resnet18, resnet34, resnet50] """
    adaptive_gating: bool = False
    """ Whether to turn on adaptive gating for domain adaptation """
    load_imagenet_weights: bool = False
    """ Whether to initialize the encoder with weights from ImageNet """
    residual_learning: bool = False
    """ Whether to use additional residual blocks for the adversarial learning """
    res_layer_norm: str = 'batch'
    """ Type of normalization to use if residual blocks are added [layer, batch, instance]"""
    res_layer_activation: str = 'leaky'
    """ activation function for the added residual layers [leaky, relu, tanh] """


@dataclass
class DiscriminatorConfig:
    activation: str = 'leaky'
    """ activation function for the layers """
    normalization: str = 'layer'
    """ normalization for each layer [batch, layer, instance] """
    in_channels: int = MISSING
    """ number of input channels """
    img_level: bool = False
    """ Whether this discriminator uses the padding/stride for image level discrimination """
    single_out: bool = False
    """ Whether to return just the max of the output """
    output_activation: str = 'sigmoid'
    """ final output activation ['', 'sigmoid', 'tanh'] """
    single_out_reduction: str = 'max'
    """ How to combine outputs to single output [max, min, mean, sum] """


@dataclass
class CallbackConfig:
    """ Configuration for callbacks to be added to the training """

    early_stop_patience: Union[int, None] = None
    """ patience for early stopping. If null, then no early stopping applied. """
    early_stop_metric: str = '${..monitor_metric}'
    """ metric for early stopping"""
    early_stop_check_every: Union[int, None] = 1
    """ check every n validation runs """
    ckpt_metric: Union[str, None] = '${..monitor_metric}'
    """ metric for model checkpoints """
    ckpt_metric_mode: str = 'max'
    """ max or min of metric """
    ckpt_save_top_k: int = 5
    """ keep the top k saved checkpoints """
    ckpt_every_n_epochs: Union[int, None] = None
    """ Number of epochs between checkpoints """
    model_ckpt_save_k: Union[int, None] = None
    """ keep the top k saved checkpoints """
    save_weights_only: bool = False
    """ Only save model weights on checkpoints """


@dataclass
class PhongConfig:
    """ The configuration for the phong dataset/dataloader """

    material_shininess: float = 100.0
    """ how reflective is the material """
    diffusion_color: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 0.25, 0.25))
    """ incoming light color, intensity based on angle to incoming light """
    specular_color: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    """ specular intensity color """
    ambient_color: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """ color that is automatically emitted by the material """
    attenuation: float = .01
    """ rate at which light falls off. intensity = 1/(1+attenuation*distance) """
    camera_intrinsics: List[List[float]] = field(default_factory=lambda: [[1038.1696, 0.0, 0.0],
                                                                          [0.0, 1039.8075, 0.0],
                                                                          [878.9617, 572.9404, 1]])
    """ 3x3 camera intrinsic matrix used for re-projection. Can be path to a numpy file or a numpy matrix. 
        TODO: implement file loading """


@dataclass
class DepthNorm2ImageConfig:
    encoder: EncoderConfig = EncoderConfig(adaptive_gating=False, residual_learning=False)
    data_roles: List[str] = field(default_factory=lambda: ['color', 'depth', 'normals'])
    """ The names to use for each type of data to be loaded """
    data_directories: List[str] = MISSING
    """ The directories corresponding to the data for each data role in  data_roles """
    training_split: dict = field(default_factory=lambda: {'train': .6,
                                                          'validate': .3,
                                                          'test': .1})
    training_split_file: str = ''
    """ An existing training split json file. If not empty, will be used instead of training_split """
    generator_lr: float = 5e-4
    """ learning rate for generator """
    discriminator_lr: float = 5e-5
    """ learning rate for discriminators """
    critic_lr: float = 5e-5
    """ learning rate for critics """
    max_epochs: int = 5000
    depth_scale: float = 1e-3
    add_noise: bool = False
    """ Add a layer of noise to input of the generator """
    use_critic: bool = True
    use_discriminator: bool = True
    critic_loss: str = 'wasserstein_gp'
    """ Which loss to use for training the critics [wasserstein_gp, wasserstein] """
    discriminator_loss: str = 'cross_entropy'
    """ Which loss to use for training the discriminators [cross_entropy, cross_entropy_R1] """
    critic_config: DiscriminatorConfig = DiscriminatorConfig(in_channels=3,
                                                             img_level=True,
                                                             normalization='instance',
                                                             output_activation='')
    discriminator_config: DiscriminatorConfig = DiscriminatorConfig(in_channels=3,
                                                                    img_level=True,
                                                                    normalization='instance',
                                                                    output_activation='sigmoid')
    wasserstein_critic_updates: int = 5
    """ how many gradient updates for critics before another generator update """
    optimizer: str = 'adam'
    """ Which generator optimizer to use. ['adam', 'radam', 'rmsprop'] """
    L_loss: str = ''
    """ L1 or L2 loss for image comparison. empty means don't use. """
    val_check_interval: int = '${..val_check_interval}'
    """ how many steps before checking validation """
    val_plot_interval: int = '${..val_plot_interval}'
    """ how many validation epochs between plotting validation images """
    train_plot_interval: int = '${..train_plot_interval}'
    """ how many steps before plotting train images """
    accumulate_grad_batches: int = 4
    """ how many batches to include before gradient update """
    batch_size: int = 32
    resume_from_checkpoint: Union[str, None] = ""
    """ checkpoint to load weights from """
    image_size: int = '${..image_size}'
    """ Final square size to make all images """
    inverse_depth: bool = "${..inverse_depth}"
    """ Whether to predict the inverse of the depth. NOT IMPLEMENTED YET """
    add_mask_blur: bool = "${..add_mask_blur}"
    """ Whether to add random gaussian blur to the edge of the circular mask """
    monitor_metric: str = 'val_loss'
    """ main metric to track for performance """
    callbacks: CallbackConfig = CallbackConfig(ckpt_every_n_epochs=2,
                                               ckpt_save_top_k=1,
                                               model_ckpt_save_k=1,
                                               save_weights_only=False,
                                               ckpt_metric=None)
    imagenet_norm_output: bool = False
    """ whether to predict normalized images or actual final color values """
    ckpt_metric: Union[str, None] = None


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
    encoder: EncoderConfig = EncoderConfig(adaptive_gating=False, residual_learning=False)
    training_split_file: str = ''
    """ An existing training split json file. If not empty, will be used instead of training_split """
    lr: float = 1e-3
    """ learning rate for optimizer """
    optimizer: str = 'radam'
    """ Which torch optimizer to use. ['adam', 'radam', 'rmsprop'] """
    lr_scheduler_patience: int = 10
    lr_scheduler_monitor: str = "val_rmse_log"
    reduce_lr_patience: int = 5
    max_epochs: int = 10
    monitor_metric: str = 'val_rmse'
    """ main metric to track for performance """
    val_check_interval: int = '${..val_check_interval}'
    """ how many steps before checking validation """
    val_plot_interval: int = '${..val_plot_interval}'
    """ how many validation epochs between plotting validation images """
    train_plot_interval: int = '${..train_plot_interval}'
    """ how many steps before plotting train images """
    accumulate_grad_batches: int = 4
    """ how many batches to include before gradient update """
    batch_size: int = 32
    resume_from_checkpoint: Union[str, None] = ""
    """ checkpoint to load weights from """
    callbacks: CallbackConfig = CallbackConfig(early_stop_patience=15, early_stop_check_every=50, ckpt_save_top_k=5)
    phong_loss_factor: float = 1.0
    """ factor for loss phong shading for depth and normals """
    depth_loss_factor: float = 1.0
    """ factor for loss BerHu on the depth outputs """
    depth_loss_lambda_factor: float = 0.8
    """ factor to scale depth loss by output layer [0 - 1] """
    normals_loss_factor: float = 1.0
    """ factor for loss cosine similarity on the normals outputs """
    depth_grad_loss_factor: float = 0.2
    """ factor for loss gradient between depth output values """
    calculated_normals_loss_factor: float = 1.0
    """ factor for comparing normals calculated from the predicted depth and the ground truth normals """
    phong_config: PhongConfig = '${..phong_config}'
    """ The config for the phong dataloader """
    predict_normals: bool = '${..predict_normals}'
    """ Whether the network should predict normals """
    image_size: int = '${..image_size}'
    """ Final square size to make all images """
    min_depth: float = .5
    """ depth value used to mask normals to zero """
    merged_decoder: bool = True
    """ Whether to use a single decoder when predicting normals """
    inverse_depth: bool = "${..inverse_depth}"
    """ Whether to predict the inverse of the depth. NOT IMPLEMENTED YET """
    add_mask_blur: bool = "${..add_mask_blur}"
    """ Whether to add random gaussian blur to the edge of the circular mask """
    depth_gradient_loss_epochs: List[int] = field(default_factory=lambda: [1, int(1e6)])
    """ Between which epochs to use loss """
    normals_depth_regularization_loss_epochs: List[int] = field(default_factory=lambda: [1, int(1e6)])
    """ Between which epochs to guide the depth using normals """
    normals_loss_epochs: List[int] = field(default_factory=lambda: [1, int(1e6)])
    """ Between which epochs to use supervised normals loss """
    phong_loss_epochs: List[int] = field(default_factory=lambda: [1, int(1e6)])
    """ Between which epochs to use phong loss """
    sync_logging: bool = '${..sync_logging}'
    """ Whether to sync log calls between devices. Can lead to large communication overhead """
    normals_extra_layers: int = 0
    """ Extra convolutions after depth before predicting normals """
    decoder_calculate_norms: bool = False
    """ Whether to calculate the normals from depth before giving to normals extra layers """


@dataclass
class GANTrainingConfig:
    """ Hyperparameter settings for the domain adaptation GAN training """

    source_images: List[str] = MISSING
    """ path to synthetically generated images """
    synth_split: dict = field(default_factory=lambda: {'train': .8,
                                                       'validate': .1,
                                                       'test': .1})
    """ The entry to control generation of the data split for training. """
    encoder: EncoderConfig = EncoderConfig(adaptive_gating=True, residual_learning=True, res_layer_norm='batch')
    training_split_file: str = ''
    """ An existing training split json file. If not empty, will be used instead of training_split """
    generator_lr: float = 5e-4
    """ learning rate for generator """
    discriminator_lr: float = 5e-5
    """ learning rate for discriminators """
    critic_lr: float = 5e-5
    """ learning rate for critics """
    critic_loss: str = 'wasserstein_gp'
    """ Which loss to use for training the critics [wasserstein_gp, wasserstein] """
    discriminator_loss: str = 'cross_entropy'
    """ Which loss to use for training the discriminators [cross_entropy, cross_entropy_R1] """
    wasserstein_lambda: float = 10.0
    """ lambda factor for wasserstein gradient penalty """
    wasserstein_critic_updates: int = 5
    """ How many batches to updated the critics before updating the generator """
    critic_use_variance: bool = False
    """ whether to use the variance of the distributions for wasserstein distance """
    max_epochs: int = 10
    generator_optimizer: str = 'adam'
    """ Which torch optimizer to use. ['adam', 'radam', 'rmsprop'] """
    critic_optimizer: str = 'rmsprop'
    """ Which torch optimizer to use. ['adam', 'radam', 'rmsprop'] """
    discriminator_optimizer: str = 'adam'
    """ Which torch optimizer to use. ['adam', 'radam', 'rmsprop'] """
    monitor_metric: str = 'g_loss'
    """ main metric to track for performance """
    val_check_interval: int = '${..val_check_interval}'
    """ how many steps before checking validation """
    val_plot_interval: int = '${..val_plot_interval}'
    """ how many validation epochs between plotting validation images """
    train_plot_interval: int = '${..train_plot_interval}'
    """ how many steps before plotting train images """
    accumulate_grad_batches: int = 4
    """ how many batches to include before gradient update """
    batch_size: int = 16
    resume_from_checkpoint: Union[str, None] = ""
    """ checkpoint to load weights from """
    generate_data: bool = False
    """ Whether to process the video data folder and generate training images in the image_output_folder """
    videos_folder: List[str] = MISSING
    """ folder with endoscopic videos """
    image_output_folder: str = MISSING
    """ folder containing (or will contain) the generated real image training data """
    callbacks: CallbackConfig = CallbackConfig(ckpt_every_n_epochs=2,
                                               ckpt_save_top_k=1,
                                               model_ckpt_save_k=None,
                                               save_weights_only=False,
                                               ckpt_metric='epoch',
                                               ckpt_metric_mode='max')
    phong_config: PhongConfig = '${..phong_config}'
    """ The config for the phong dataloader """
    predict_normals: bool = '${..predict_normals}'
    """ Whether the network should predict normals """
    image_size: int = '${..image_size}'
    """ Final square size to make all images """
    freeze_batch_norm: bool = True
    """ Whether to freeze the batch norm statistics for the already learned generator """
    use_critic: bool = True
    use_discriminator: bool = True
    depth_discriminator: DiscriminatorConfig = DiscriminatorConfig(in_channels=1,
                                                                   img_level=True,
                                                                   single_out=False,
                                                                   output_activation='sigmoid')
    depth_critic: DiscriminatorConfig = DiscriminatorConfig(in_channels=1,
                                                            img_level=True,
                                                            single_out=False,
                                                            output_activation='')
    normals_critic: DiscriminatorConfig = DiscriminatorConfig(in_channels=3,
                                                              img_level=True,
                                                              single_out=False,
                                                              output_activation='')
    normals_discriminator: DiscriminatorConfig = DiscriminatorConfig(in_channels=3,
                                                                     img_level=True,
                                                                     single_out=False,
                                                                     output_activation='sigmoid')
    phong_discriminator: DiscriminatorConfig = DiscriminatorConfig(in_channels=3,
                                                                   img_level=True,
                                                                   single_out=False,
                                                                   output_activation='sigmoid')
    phong_critic: DiscriminatorConfig = DiscriminatorConfig(in_channels=3,
                                                            img_level=True,
                                                            single_out=False,
                                                            output_activation='')
    feature_level_discriminator: DiscriminatorConfig = DiscriminatorConfig(in_channels=-1,
                                                                           img_level=False,
                                                                           single_out=False,
                                                                           output_activation='sigmoid')
    feature_level_critic: DiscriminatorConfig = DiscriminatorConfig(in_channels=-1,
                                                                    img_level=False,
                                                                    single_out=False,
                                                                    output_activation='')
    img_discriminator_factor: float = 1.0
    phong_discriminator_factor: float = 1.0
    normals_discriminator_factor: float = 1.0
    feature_discriminator_factor: float = 1.0
    """ scaling factor for feature level discriminators """
    d_max_conf: float = 0.9
    """ scaling factor for confidence of discriminators on synthetic data """
    warmup_steps: int = 0
    """ how many steps to train the discriminator before training generator """
    sync_logging: bool = '${..sync_logging}'
    """ Whether to sync log calls between devices. Can lead to large communication overhead """


@dataclass
class CystoDepthConfig:
    """ Configuration for training synthetic depth and/or domain transfer for real cystoscopic videos"""

    mode: str = 'synthetic'
    """ Mode can be one of ['synthetic', 'gan', 'depthnorm'] """
    training_stage: str = 'train'
    """ Training_stage can be one of ['train', 'test'] """
    log_directory: str = './logs'
    """ tensorboard log directory """
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
    depth_norm_config: DepthNorm2ImageConfig = DepthNorm2ImageConfig()
    phong_config: PhongConfig = PhongConfig()
    """ The config for the phong dataloader """
    predict_normals: bool = False
    """ Whether the network should predict normals """
    optimizer: str = 'adam'
    """ What optimizer to use. one of ['adam', 'radam'] """
    inverse_depth: bool = False
    """ Whether to predict the inverse of the depth """
    add_mask_blur: bool = False
    """ Whether to add random gaussian blur to the edge of the circular mask """
    sync_logging: bool = False
    """ Whether to sync log calls between devices. Can lead to large communication overhead """
    memorize_check: bool = False
    """ Whether to memorize a single batch from the training """
    torch_float_precision: str = ""
    """ Sets the internal precision of float32 matrix multiplications. [medium, high, highest] """
    val_check_interval: int = 10
    """ how many steps before checking validation """
    val_plot_interval: int = 30
    """ how many validation epochs between plotting validation images """
    train_plot_interval: int = 500
    """ how many steps before plotting train images """
    slurm_requeue: bool = False
    """ Whether to automatically requeue the training with a checkpoint """
    pin_dataloader_memory: bool = False
    """ Whether to pin_memory when getting the dataloaders """
