import pytorch_lightning as pl
from models.depth_to_image import DepthNorm2Image
from config.training_config import DepthNorm2ImageConfig
import matplotlib.pyplot as plt
from utils.image_utils import generate_img_fig
from data.data_transforms import ImageNetNormalization
from models.discriminator import Discriminator
from argparse import Namespace
from typing import *
from torch import Tensor, optim
import torch
from utils.loss import GANDiscriminatorLoss, GANGeneratorLoss

imagenet_denorm = ImageNetNormalization(inverse=True)
imagenet_norm = ImageNetNormalization()


class DepthNormModel(pl.LightningModule):
    def __init__(self, config: DepthNorm2ImageConfig):
        super(DepthNormModel, self).__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(Namespace(**config))
        self.config = config
        self.model = DepthNorm2Image(config.encoder,
                                     depth_scale=config.depth_scale,
                                     add_noise=config.add_noise,
                                     sigmoid=not config.imagenet_norm_output)
        self.max_num_image_samples = 4
        self.val_denorm_color_images = None
        self.validation_data = None
        self._val_epoch_count = 0
        self.g_losses_log = {'g_loss': 0}
        self.d_losses_log = {}
        self.critic_opt_idx = 0
        sources = list(range(len(config.data_roles) // 3))
        self.sources = sources
        if config.use_critic:
            critics = {str(i): Discriminator(config.critic_config) for i in sources}
            self.critics = torch.nn.ModuleDict(critics)
            self.critic_opt_idx += 1
            self.d_losses_log[f'd_critic_loss'] = 0.0
            self.g_losses_log.update({f'g_critic_loss-{i}': 0.0 for i in sources})
            self.d_losses_log.update({f'd_critic_loss-{i}': 0.0 for i in sources})

        if config.use_discriminator:
            discriminators = {str(i): Discriminator(config.discriminator_config) for i in sources}
            self.discriminators = torch.nn.ModuleDict(discriminators)
            self.discriminators_opt_idx = self.critic_opt_idx + 1
            self.d_losses_log['d_discriminator_loss'] = 0.0
            self.g_losses_log.update({f'g_discriminator_loss-{i}': 0.0 for i in sources})
            self.d_losses_log.update({f'd_discriminator_loss-{i}': 0.0 for i in sources})

        self.generator_global_step = -1
        self.critic_global_step = 0
        self.total_train_step_count = -1
        self.discriminator_critic_loss = GANDiscriminatorLoss['wasserstein_gp']
        self.discriminator_loss = GANDiscriminatorLoss['cross_entropy_R1']
        self.generator_critic_loss = GANGeneratorLoss['wasserstein_gp']
        self.generator_discriminator_loss = GANGeneratorLoss['cross_entropy_R1']
        self.L_loss = None
        if config.L_loss:
            self.L_loss = torch.nn.L1Loss() if '1' in config.L_loss else torch.nn.MSELoss()
            self.g_losses_log.update({'g_img_loss': 0})
        self.val_loss = 0
        self.val_batch_count = 0
        self.batches_accumulated = 0
        self._generator_training = not (self.config.use_critic or self.config.use_discriminator)
        if config.resume_from_checkpoint:
            path_to_ckpt = config.resume_from_checkpoint
            config.resume_from_checkpoint = ""  # set empty or a recursive loading problem occurs
            ckpt = self.load_from_checkpoint(path_to_ckpt,
                                             strict=False,
                                             config=config)
            self.load_state_dict(ckpt.state_dict())

    def forward(self, depth: Tensor, normals: Tensor, source_id: int, **kwargs: Any, ) -> Tensor:
        """

        :param depth: [N, 1, h, w]
        :param normals: [N, 3, h, w]
        :param source_id: id for domain the data came from
        :param kwargs: ain't matter
        :return: color image/s [N, 3, h ,w]
        """
        return self.model(depth, normals, source_id=source_id)

    def __call__(self, *args, **kwargs) -> Tensor:
        return super(DepthNormModel, self).__call__(*args, **kwargs)

    def configure_optimizers(self):
        opts = {'adam': optim.Adam, 'radam': optim.RAdam, 'rmsprop': optim.RMSprop}
        opt = opts[self.config.optimizer.lower()]
        optimizer = opt(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.generator_lr)
        optimizers = [optimizer]
        if self.config.use_critic:
            optimizers.append(optim.RMSprop(filter(lambda p: p.requires_grad, self.critics.parameters()),
                                            lr=self.config.critic_lr))
        if self.config.use_discriminator:
            optimizers.append(optim.Adam(filter(lambda p: p.requires_grad, self.discriminators.parameters()),
                                         lr=self.config.discriminator_lr))
        return optimizers  # , [scheduler]

    def training_step(self, batch, batch_idx):
        self.total_train_step_count += 1
        if self._generator_training:
            # print('generator')
            self.generator_train_step(batch, batch_idx)
        else:
            if self.critic_global_step % self.config.wasserstein_critic_updates == 0 and self.config.use_discriminator:
                # print('discriminator')
                self.discriminator_train_step(batch, batch_idx)  # only update discriminators on first critic update
            # print('critic')
            if self.config.use_critic:
                self.critic_train_step(batch, batch_idx)

    def calculate_generator_loss(self, batch) -> Tensor:
        self.model.train()
        if self.config.use_critic:
            self.critics.eval()
        if self.config.use_discriminator:
            self.discriminators.eval()
        loss: Tensor = 0.0
        for source_id in batch.keys():
            img, depth, normals = batch[source_id]
            out_images = self(depth, normals, source_id=source_id)
            denormed_images = img if self.config.imagenet_norm_output else imagenet_denorm(img)
            if self.L_loss is not None:
                img_loss = self.L_loss(out_images, denormed_images)
                self.g_losses_log['g_img_loss'] += img_loss
                loss += img_loss
            if self.config.use_critic:
                critic_out = self.critics[str(source_id)](out_images)
                critic_loss = self.generator_critic_loss(critic_out)
                self.g_losses_log[f'g_critic_loss-{source_id}'] += critic_loss
                loss += critic_loss
            if self.config.use_discriminator:
                discriminator_out = self.discriminators[str(source_id)](out_images)
                discriminator_loss = self.generator_discriminator_loss(discriminator_out, 1.0)
                self.g_losses_log[f'g_discriminator_loss-{source_id}'] += discriminator_loss
                loss += discriminator_loss
        self.g_losses_log['g_loss'] += loss
        return loss

    def generator_train_step(self, batch: dict, batch_idx: int):
        loss = self.calculate_generator_loss(batch)
        self.manual_backward(loss)
        self.batches_accumulated += 1
        step_optimizers = False
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.generator_global_step += 1
            self.batches_accumulated = 0
            self._generator_training = False if self.config.use_critic or self.config.use_discriminator else True
            step_optimizers = True
        opt = self.optimizers(use_pl_optimizer=True)[0]
        if step_optimizers:
            # print('step generator')
            opt.step()
            opt.zero_grad()
            self.log_dict(self.g_losses_log)
            self.g_losses_log.update({k: 0 for k in self.g_losses_log.keys()})
            self.zero_grad()

    def calculate_discriminator_loss(self, batch) -> Tensor:
        self.model.eval()
        if self.config.use_critic:
            self.critics.eval()
        self.discriminators.train()

        discriminator_loss: Tensor = 0
        for source_id in batch.keys():
            with torch.no_grad():
                original_img, original_depth, original_normals = batch[source_id]
                denormed_images = original_img if self.config.imagenet_norm_output else imagenet_denorm(original_img)
                out_images = self(original_depth, original_normals, source_id=source_id)
            loss_generated = self.discriminator_loss(out_images.detach(), 0.0, self.discriminators[str(source_id)])
            loss_original = self.discriminator_loss(denormed_images.detach(), 1.0, self.discriminators[str(source_id)])
            combined = loss_original + loss_generated
            discriminator_loss += combined
            self.d_losses_log[f'd_discriminator_loss-{source_id}'] += combined
        return discriminator_loss

    def discriminator_train_step(self, batch: dict, batch_idx):
        discriminator_loss = self.calculate_discriminator_loss(batch)

        self.manual_backward(discriminator_loss)
        self.d_losses_log['d_discriminator_loss'] += discriminator_loss

        # +1 because they are stepped in critic update
        if self.batches_accumulated + 1 == self.config.accumulate_grad_batches:
            # print('step discriminators')
            discriminator_opt = self.optimizers(True)[self.discriminators_opt_idx]
            discriminator_opts = [discriminator_opt] if not isinstance(discriminator_opt, list) else discriminator_opt
            [d_opt.step() for d_opt in discriminator_opts]
            [d_opt.zero_grad() for d_opt in discriminator_opts]

    def calculate_critic_loss(self, batch) -> Tensor:
        self.model.eval()
        self.critics.train()
        if self.config.use_discriminator:
            self.discriminators.eval()

        loss: Tensor = 0
        for source_id in batch.keys():
            with torch.no_grad():
                original_img, original_depth, original_normals = batch[source_id]
                denormed_images = original_img if self.config.imagenet_norm_output else imagenet_denorm(original_img)
                out_images = self(original_depth, original_normals, source_id=source_id)
            critic_loss = self.discriminator_critic_loss(denormed_images,
                                                         out_images.detach(),
                                                         self.critics[str(source_id)],
                                                         10,
                                                         self.config.critic_use_variance)
            self.d_losses_log[f'd_critic_loss-{source_id}'] += critic_loss
            loss += critic_loss
        self.d_losses_log[f'd_critic_loss'] += loss
        return loss

    def critic_train_step(self, batch: dict, batch_idx):
        loss = self.calculate_critic_loss(batch)
        self.manual_backward(loss)

        self.batches_accumulated += 1
        step_optimizers = False
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.critic_global_step += 1
            self.batches_accumulated = 0
            step_optimizers = True
            if self.critic_global_step % self.config.wasserstein_critic_updates == 0:
                self._generator_training = not self._generator_training
                self.log_dict(self.d_losses_log)
                self.d_losses_log.update({k: 0.0 for k in self.d_losses_log.keys()})

        if step_optimizers:
            # print('step critics')
            critic_opts = self.optimizers(True)[self.critic_opt_idx]
            critic_opts = [critic_opts] if not isinstance(critic_opts, list) else critic_opts
            [d_opt.step() for d_opt in critic_opts]
            [d_opt.zero_grad() for d_opt in critic_opts]

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            loss = 0
            if self.L_loss is not None:
                for source_id in batch:
                    original_img, original_depth, original_normals = batch[source_id]
                    out_images = self(original_depth, original_normals, source_id=source_id)
                    denormed_images = original_img if self.config.imagenet_norm_output else imagenet_denorm(original_img)
                    loss += self.L_loss(out_images, denormed_images)
                self.val_loss += loss / len(batch)
                self.val_batch_count += 1

            if self.validation_data is None:
                self.validation_data = {}
                original_images_all = []
                original_depths_all = []
                original_normals_all = []
                for source_id in batch:
                    original_img, original_depth, original_normals = batch[source_id]
                    original_images_all.append(original_img[:self.max_num_image_samples])
                    original_depths_all.append(original_depth[:self.max_num_image_samples])
                    original_normals_all.append(original_normals[:self.max_num_image_samples])
                    self.validation_data[source_id] = []
                    self.validation_data[source_id].append(imagenet_denorm(
                        original_img[:self.max_num_image_samples]).detach().cpu())
                    self.validation_data[source_id].append(original_depth[:self.max_num_image_samples].detach())
                    self.validation_data[source_id].append(original_normals[:self.max_num_image_samples].detach())
                self.val_denorm_color_images = torch.cat([self.validation_data[i][0] for i in self.validation_data], dim=0)

    def on_validation_epoch_end(self) -> None:
        if self.val_batch_count > 0:
            self.log('val_loss', self.val_loss / self.val_batch_count)
        self.val_loss = 0
        self.val_batch_count = 0
        if self._val_epoch_count % self.config.val_plot_interval == 0:
            if self.validation_data is not None:
                self.plot()
        self._val_epoch_count += 1
        return super().on_validation_epoch_end()

    def plot(self, step: int = None):
        with torch.no_grad():
            generated_imgs = []
            for source_id in self.validation_data:
                generated_imgs.append(self(self.validation_data[source_id][1],
                                           self.validation_data[source_id][2],
                                           source_id=source_id).detach().cpu())
        labels = ["Source Image", "Generated"]
        generated_imgs = torch.cat(generated_imgs, dim=0)
        generated_imgs = imagenet_denorm(generated_imgs) if self.config.imagenet_norm_output else generated_imgs
        step = self.global_step if step is None else step
        for idx, img_set in enumerate(zip(self.val_denorm_color_images, generated_imgs)):
            fig = generate_img_fig(img_set, labels)
            self.logger.experiment.add_figure(f'generated-image-{idx}', fig, step)
            plt.close(fig)
