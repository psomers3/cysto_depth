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
        self.generator_losses = {'g_loss': 0}
        self.critic_losses = {}
        self.discriminator_losses = {}
        self.critic_opt_idx = 0
        sources = list(range(len(config.data_roles) // 3))
        self.sources = sources
        self.critics = torch.nn.ModuleDict()
        self.discriminators = torch.nn.ModuleDict()
        if config.use_critic:
            self.critics.update({str(i): Discriminator(config.critic_config) for i in sources})
            self.critic_opt_idx += 1
            self.critic_losses[f'd_critic_loss'] = 0.0
            self.generator_losses.update({f'g_critic_loss-{i}': 0.0 for i in sources})
            self.critic_losses.update({f'd_critic_loss-{i}': 0.0 for i in sources})
            self.critic_losses.update({f'd_critic_gp_{i}': 0.0 for i in sources})

        if config.use_discriminator:
            self.discriminators.update({str(i): Discriminator(config.discriminator_config) for i in sources})
            self.discriminators_opt_idx = self.critic_opt_idx + 1
            self.discriminator_losses['d_discriminator_loss'] = 0.0
            self.generator_losses.update({f'g_discriminator_loss-{i}': 0.0 for i in sources})
            self.discriminator_losses.update({f'd_discriminator_loss-{i}': 0.0 for i in sources})
            self.discriminator_losses.update({f'd_discriminator_reg_loss-{i}': 0.0 for i in sources})

        self.generator_global_step = -1
        self.critic_global_step = 0
        self.total_train_step_count = -1
        self.discriminator_critic_loss = GANDiscriminatorLoss[config.critic_loss]
        self.discriminator_loss = GANDiscriminatorLoss[config.discriminator_loss]
        self.generator_critic_loss = GANGeneratorLoss[config.critic_loss]
        self.generator_discriminator_loss = GANGeneratorLoss[config.discriminator_loss]
        self.L_loss = None
        if config.L_loss:
            self.L_loss = torch.nn.L1Loss() if '1' in config.L_loss else torch.nn.MSELoss()
            self.generator_losses.update({'g_img_loss': 0})
        self.val_loss = 0
        self.val_batch_count = 0
        self.batches_accumulated = 0
        self._generator_training = not (self.config.use_critic or self.config.use_discriminator)
        if config.resume_from_checkpoint:
            self._resume_from_checkpoint(config)
        self._full_batch = False

    def _resume_from_checkpoint(self, config):
        path_to_ckpt = config.resume_from_checkpoint
        ckpt = torch.load(path_to_ckpt, map_location=self.device)
        with torch.no_grad():
            # run some data through the network to initial dense layers in discriminators if needed
            temp_out = self(torch.ones(1, 1, config.image_size, config.image_size, device=self.device),
                            torch.ones(1, 3, config.image_size, config.image_size, device=self.device),
                            source_id=0)
            for d in self.discriminators:
                self.discriminators[d](temp_out)
            for c in self.critics:
                self.critics[c](temp_out)

        self.load_state_dict(ckpt, strict=False)

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
        optimizer = opt(self.model.parameters(), lr=self.config.generator_lr)
        optimizers = [optimizer]
        if self.config.use_critic:
            optimizers.append(optim.RMSprop(self.critics.parameters(), lr=self.config.critic_lr))
        if self.config.use_discriminator:
            optimizers.append(optim.Adam(self.discriminators.parameters(), lr=self.config.discriminator_lr))
        return optimizers  # , [scheduler]

    def training_step(self, batch, batch_idx):
        self.total_train_step_count += 1
        self.batches_accumulated += 1
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self._full_batch = True
            self.batches_accumulated = 0

        if self._generator_training:
            # print('generator')
            self.generator_train_step(batch, batch_idx)
        else:
            if (
                    self.critic_global_step % self.config.wasserstein_critic_updates == 0) and self.config.use_discriminator:
                # print('discriminator')
                self.discriminator_train_step(batch, batch_idx)  # only update discriminators on first critic update
            if self.config.use_critic:
                # print('critic')
                self.critic_train_step(batch, batch_idx)
        self._full_batch = False

    def calculate_generator_loss(self, batch) -> Tensor:
        self.model.train()
        self.critics.eval()
        self.discriminators.eval()
        loss: Tensor = 0.0
        for source_id in batch.keys():
            img, depth, normals = batch[source_id]
            out_images = self(depth, normals, source_id=source_id)
            if self.L_loss is not None:
                denormed_images = img if self.config.imagenet_norm_output else imagenet_denorm(img).detach()
                img_loss = self.L_loss(out_images, denormed_images)
                self.generator_losses['g_img_loss'] += img_loss.detach()
                loss += img_loss
            if self.config.use_critic:
                critic_out = self.critics[str(source_id)](out_images)
                critic_loss = self.generator_critic_loss(critic_out)
                self.generator_losses[f'g_critic_loss-{source_id}'] += critic_loss.detach()
                loss += critic_loss
            if self.config.use_discriminator:
                discriminator_loss, penalty = self.generator_discriminator_loss(out_images, 1.0,
                                                                       self.discriminators[str(source_id)])
                self.generator_losses[f'g_discriminator_loss-{source_id}'] += discriminator_loss.detach()
                loss += discriminator_loss
        self.generator_losses['g_loss'] += loss.detach()
        return loss

    def generator_train_step(self, batch: dict, batch_idx: int):
        loss = self.calculate_generator_loss(batch)
        self.manual_backward(loss)
        if self._full_batch:
            self.generator_global_step += 1
            self._generator_training = False
            opt = self.optimizers(use_pl_optimizer=True)[0]
            # print('step generator')
            opt.step()
            opt.zero_grad()
            self.generator_losses.update({k: self.generator_losses[k] / self.config.accumulate_grad_batches
                                          for k in self.generator_losses.keys()})
            self.log_dict(self.generator_losses)
            self.generator_losses.update({k: 0 for k in self.generator_losses.keys()})
            self.zero_grad()

    def calculate_discriminator_loss(self, batch) -> Tensor:
        self.model.eval()
        self.critics.eval()
        self.discriminators.train()

        discriminator_loss: Tensor = 0
        for source_id in batch.keys():
            with torch.no_grad():
                original_img, original_depth, original_normals = batch[source_id]
                denormed_images = original_img if self.config.imagenet_norm_output else imagenet_denorm(original_img)
                out_images = self(original_depth, original_normals, source_id=source_id)
            loss_generated, penalty_generated = self.discriminator_loss(out_images.detach(), 0.0, self.discriminators[str(source_id)])
            loss_original, penalty_original = self.discriminator_loss(denormed_images.detach(), 1.0, self.discriminators[str(source_id)])
            penalty_combined = penalty_original + penalty_generated
            loss_combined = loss_original + loss_generated
            discriminator_loss += penalty_combined + loss_combined
            self.discriminator_losses[f'd_discriminator_loss-{source_id}'] += loss_combined.detach()
            self.discriminator_losses[f'd_discriminator_reg_loss-{source_id}'] += penalty_combined.detach()

        return discriminator_loss

    def discriminator_train_step(self, batch: dict, batch_idx):
        discriminator_loss = self.calculate_discriminator_loss(batch)

        self.manual_backward(discriminator_loss)
        self.discriminator_losses['d_discriminator_loss'] += discriminator_loss.detach()

        if self._full_batch:
            # print('step discriminators')
            discriminator_opt = self.optimizers(True)[self.discriminators_opt_idx]
            discriminator_opt.step()
            discriminator_opt.zero_grad()
            if not self.config.use_critic:
                self._generator_training = True
                self.discriminator_losses.update({k: self.discriminator_losses[k] / self.config.accumulate_grad_batches
                                          for k in self.discriminator_losses.keys()})
                self.log_dict(self.discriminator_losses)
                self.discriminator_losses.update({k: 0.0 for k in self.discriminator_losses.keys()})

    def calculate_critic_loss(self, batch) -> Tensor:
        self.model.eval()
        self.critics.train()
        self.discriminators.eval()

        loss: Tensor = 0
        for source_id in batch.keys():
            with torch.no_grad():
                original_img, original_depth, original_normals = batch[source_id]
                denormed_images = original_img if self.config.imagenet_norm_output else imagenet_denorm(original_img)
                out_images = self(original_depth, original_normals, source_id=source_id)
            critic_loss, penalty = self.discriminator_critic_loss(denormed_images,
                                                                  out_images.detach(),
                                                                  self.critics[str(source_id)],
                                                                  10)

            self.critic_losses[f'd_critic_loss-{source_id}'] += critic_loss.detach()
            self.critic_losses[f'd_critic_gp_{source_id}'] += penalty.detach()
            loss += critic_loss + penalty
        self.critic_losses[f'd_critic_loss'] += loss.detach()
        return loss

    def critic_train_step(self, batch: dict, batch_idx):
        loss = self.calculate_critic_loss(batch)
        self.manual_backward(loss)

        if self._full_batch:
            self.critic_global_step += 1
            self.critic_losses.update({k: self.critic_losses[k] / self.config.accumulate_grad_batches
                                      for k in self.critic_losses.keys()})
            self.log_dict(self.critic_losses)
            self.critic_losses.update({k: 0.0 for k in self.critic_losses.keys()})
            # print('step critics')
            critic_opt = self.optimizers(True)[self.critic_opt_idx]
            critic_opt.step()
            critic_opt.zero_grad()
            if self.critic_global_step % self.config.wasserstein_critic_updates == 0:
                self._generator_training = True

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            loss: Tensor = 0
            l_loss = self.L_loss if self.L_loss is not None else torch.nn.L1Loss()
            for source_id in batch:
                original_img, original_depth, original_normals = batch[source_id]
                out_images = self(original_depth, original_normals, source_id=source_id)
                denormed_images = original_img if self.config.imagenet_norm_output else imagenet_denorm(original_img)
                loss += l_loss(out_images, denormed_images)
            self.val_loss += (loss / len(batch)).detach()
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
                self.val_denorm_color_images = torch.cat([self.validation_data[i][0] for i in self.validation_data],
                                                         dim=0)

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
