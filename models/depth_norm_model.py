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


class DepthNormModel(pl.LightningModule):
    def __init__(self, config: DepthNorm2ImageConfig):
        super(DepthNormModel, self).__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(Namespace(**config))
        self.config = config
        self.model = DepthNorm2Image(config.encoder, depth_scale=config.depth_scale, add_noise=config.add_noise)
        self.loss = torch.nn.L1Loss() if '1' in config.L_loss else torch.nn.MSELoss()
        self.max_num_image_samples = 8
        if config.resume_from_checkpoint:
            path_to_ckpt = config.resume_from_checkpoint
            config.resume_from_checkpoint = ""  # set empty or a recursive loading problem occurs
            ckpt = self.load_from_checkpoint(path_to_ckpt,
                                             strict=False,
                                             config=config)
            self.load_state_dict(ckpt.state_dict())

        self.val_denorm_color_images = None
        self.validation_data = None
        self._val_epoch_count = 0
        self.critic = Discriminator(config.critic_discriminator_config) if config.use_critic else None
        self.generator_global_step = -1
        self.critic_global_step = -1
        self.total_train_step_count = -1
        self.critic_loss = GANDiscriminatorLoss['wasserstein_gp']
        self.generator_loss = GANGeneratorLoss['wasserstein_gp']
        self.check_for_generator_step = self.config.wasserstein_critic_updates + 1
        self.g_losses_log = {'g_img_loss': 0, 'g_critic_loss': 0, 'g_loss': 0}
        self.d_losses_log = {'d_critic_loss': 0}
        self.val_loss = 0
        self.val_batch_count = 0
        self.batches_accumulated = 0
        self._generator_training = False

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
        optimizer = opt(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr)
        optimizers = [optimizer]
        if self.config.use_critic:
            optimizers.append(optim.RMSprop(filter(lambda p: p.requires_grad,
                                                   self.critic.parameters()),
                                            lr=self.config.lr))
        return optimizers  # , [scheduler]

    def training_step(self, batch, batch_idx):
        self._generator_training = True if not self.config.use_critic else self._generator_training
        self.total_train_step_count += 1
        if self._generator_training:
            self.generator_train_step(batch, batch_idx)
        else:
            self.critic_train_step(batch, batch_idx)

    def generator_train_step(self, batch, batch_idx):
        self.model.train()
        if self.config.use_critic:
            self.critic.eval()
        synth_img, synth_depth, synth_normals = batch
        out_images = self(synth_depth, synth_normals, source_id=0)
        denormed_images = imagenet_denorm(synth_img)
        img_loss = self.loss(out_images, denormed_images)
        self.g_losses_log['g_img_loss'] += img_loss
        loss = img_loss
        if self.config.use_critic:
            critic_out = self.critic(out_images)
            critic_loss = self.generator_loss(critic_out)
            self.g_losses_log['g_critic_loss'] += critic_loss
            loss += critic_loss
        self.g_losses_log['g_loss'] += loss
        self.manual_backward(loss)
        self.batches_accumulated += 1
        step_optimizers = False
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.generator_global_step += 1
            self.batches_accumulated = 0
            self._generator_training = not self._generator_training
            step_optimizers = True
        opt = self.optimizers(use_pl_optimizer=True)[0]
        if step_optimizers:
            opt.step()
            opt.zero_grad()
            self.log_dict(self.g_losses_log)
            self.g_losses_log.update({k: 0 for k in self.g_losses_log.keys()})

    def critic_train_step(self, batch, batch_idx):
        self.model.eval()
        self.critic.train()
        with torch.no_grad():
            synth_img, synth_depth, synth_normals = batch
            denormed_images = imagenet_denorm(synth_img)
            out_images = self(synth_depth, synth_normals, source_id=0)

        critic_loss = self.critic_loss(denormed_images, out_images, self.critic, 10)
        self.d_losses_log['d_critic_loss'] += critic_loss
        self.manual_backward(critic_loss)

        self.batches_accumulated += 1
        step_optimizers = False
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.critic_global_step += 1
            self.batches_accumulated = 0
            step_optimizers = True
            if self.critic_global_step % self.config.wasserstein_critic_updates == 0:
                self._generator_training = not self._generator_training

        if step_optimizers:
            critic_opts = self.optimizers(True)[1:]
            critic_opts = [critic_opts] if not isinstance(critic_opts, list) else critic_opts
            [d_opt.step() for d_opt in critic_opts]
            [d_opt.zero_grad() for d_opt in critic_opts]
            self.log_dict(self.d_losses_log)
            self.d_losses_log.update({k: 0 for k in self.d_losses_log.keys()})

    def validation_step(self, batch, batch_idx):
        self.eval()
        synth_img, synth_depth, synth_normals = batch
        out_images = self(synth_depth, synth_normals, source_id=0)
        loss = self.loss(out_images, imagenet_denorm(synth_img))
        self.val_loss += loss
        self.val_batch_count += 1

        if self.validation_data is None:
            synth_img, synth_depth, synth_normals = batch
            self.val_denorm_color_images = imagenet_denorm(synth_img[:self.max_num_image_samples]).detach().cpu()
            self.validation_data = synth_depth[:self.max_num_image_samples].detach(), \
                                   synth_normals[:self.max_num_image_samples].detach()

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', self.val_loss/self.val_batch_count)
        self.val_loss = 0
        self.val_batch_count = 0
        if self._val_epoch_count % self.config.val_plot_interval == 0:
            if self.validation_data is not None:
                self.plot()
        self._val_epoch_count += 1
        return super().on_validation_epoch_end()

    def plot(self):
        with torch.no_grad():
            generated_img = self(*self.validation_data, source_id=0).detach().cpu()
        labels = ["Source Image", "Generated"]

        for idx, img_set in enumerate(zip(self.val_denorm_color_images, generated_img)):
            fig = generate_img_fig(img_set, labels)
            self.logger.experiment.add_figure(f'generated-image-{idx}', fig, self.global_step)
            plt.close(fig)
