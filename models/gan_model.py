from matplotlib import pyplot as plt
import torch
from torch import Tensor
from models.base_model import BaseModel
from models.adaptive_encoder import AdaptiveEncoder
from models.discriminator import Discriminator
from models.depth_model import DepthEstimationModel
from data.data_transforms import ImageNetNormalization
from utils.image_utils import generate_heatmap_fig, freeze_batchnorm, generate_final_imgs, generate_img_fig
from config.training_config import SyntheticTrainingConfig, GANTrainingConfig, DiscriminatorConfig
from argparse import Namespace
from utils.rendering import PhongRender, depth_to_normals
from utils.loss import CosineSimilarity, GANDiscriminatorLoss, GANGeneratorLoss
from typing import *


class GAN(BaseModel):
    def __init__(self, synth_config: SyntheticTrainingConfig, gan_config: GANTrainingConfig):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(Namespace(**gan_config))
        synth_config.resume_from_checkpoint = ''
        self.depth_model: DepthEstimationModel = DepthEstimationModel.load_from_checkpoint(
            gan_config.synthetic_base_model,
            strict=False,
            config=synth_config)
        self.config = gan_config
        self.generator = AdaptiveEncoder(gan_config.encoder)

        self.generator.load_state_dict(self.depth_model.encoder.state_dict(), strict=False)
        self.depth_model.requires_grad = False

        d_in_shapes = self.generator.feature_levels[::-1]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:-1]:
            d_config: DiscriminatorConfig = gan_config.feature_level_discriminator.copy()
            d_config.in_channels = d_in_shape
            d = Discriminator(d_config)
            d_feat_list.append(d)
        self.d_img = Discriminator(gan_config.depth_discriminator)
        self.d_feat_modules = torch.nn.ModuleList(modules=d_feat_list)
        self.imagenet_denorm = ImageNetNormalization(inverse=True)
        self.phong_renderer: PhongRender = None
        self.phong_discriminator = Discriminator(gan_config.phong_discriminator)
        self.depth_phong_discriminator = Discriminator(gan_config.phong_discriminator)
        self.cosine_sim: CosineSimilarity = None
        self.feat_idx_start: int = 0
        # TODO: make the log dictionaries TypedDicts and define them elsewhere with comments
        self.d_losses_log = {'d_loss': 0, 'd_loss_depth_img': 0, 'd_loss_phong': 0, 'd_loss_depth_phong': 0}
        self.d_losses_log.update({f'd_loss_feature_{i}': 0 for i in range(len(d_feat_list))})
        self.g_losses_log = {'g_loss': 0, 'g_loss_depth_img': 0, 'g_loss_phong': 0, 'g_feat_loss': 0, 'g_res_loss': 0,
                             'g_loss_depth_phong': 0}
        self.g_losses_log.update({f'g_loss_feature_{i}': 0 for i in range(len(d_feat_list))})
        self.generator_global_step = -1
        self.discriminators_global_step = -1
        self.total_train_step_count = -1
        self.unadapted_images_for_plotting = None
        self.discriminator_loss = GANDiscriminatorLoss[gan_config.loss]
        self.generator_loss = GANGeneratorLoss[gan_config.loss]
        self.check_for_generator_step = self.config.wasserstein_critic_updates + 1 if \
            self.config.loss != 'cross_entropy' else 2

    def forward(self, x, generator: bool = True) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
        return self.get_predictions(x, generator=generator)

    def __call__(self, *args, **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
        return super(GAN, self).__call__(*args, **kwargs)

    def on_validation_epoch_start(self) -> None:
        if self.config.predict_normals:
            self.phong_renderer = PhongRender(config=self.config.phong_config,
                                              image_size=self.config.image_size,
                                              device=self.device)
            self.cosine_sim = CosineSimilarity(ignore_direction=True, device=self.device)

    def get_predictions(self, x: torch.Tensor, generator: bool) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
        """ helper function to clean up the training_step function.

        :param x: batch of images
        :param generator: whether to use the generator encoder
        :return: encoder_outs, encoder_mare_outs, decoder_outs, normals
                Note: everything is a list of tensors for each Unet level except normals.
        """
        if generator:
            encoder_outs, encoder_mare_outs = self.generator(x)
        else:
            encoder_outs, encoder_mare_outs = self.depth_model.encoder(x)
        if self.config.predict_normals:
            if self.depth_model.config.merged_decoder:
                output = self.depth_model.decoder(encoder_outs)
                decoder_outs = [self.depth_model.decoder(encoder_outs)[i][:, 0, ...].unsqueeze(1) for i in
                                range(len(output))]
                normals = output[-1][:, 1:, ...]
            else:
                normals = self.depth_model.normals_decoder(encoder_outs)
                decoder_outs = self.depth_model.decoder(encoder_outs)
            normals = torch.where(decoder_outs[-1] > self.depth_model.config.min_depth,
                                  normals, torch.zeros([1], device=self.device))
        else:
            decoder_outs = self.depth_model.decoder(encoder_outs)
            normals = None

        return encoder_outs, encoder_mare_outs, decoder_outs, normals

    def training_step(self, batch, batch_idx):
        # start by freezing all batchnorm layers throughout the networks that shouldn't update statistics
        self.train()
        self.depth_model.apply(freeze_batchnorm)
        if self.config.freeze_batch_norm:
            self.generator.apply(freeze_batchnorm)

        generator_step = self.total_train_step_count % self.check_for_generator_step == 0
        self.total_train_step_count += 1
        if generator_step and self.global_step >= self.config.warmup_steps:
            self.generator_train_step(batch, batch_idx)
        else:
            self.discriminators_train_step(batch, batch_idx)

    def generator_train_step(self, batch, batch_idx) -> None:
        # x = synthetic image, z = real image
        _, z = batch

        # set discriminators to eval so that any normalization statistics don't get updated
        self.d_img.eval()
        self.d_feat_modules.eval()
        self.depth_phong_discriminator.eval()
        self.phong_discriminator.eval()
        if self.config.encoder.residual_learning:
            self.generator.set_residuals_train()

        optimizers: List[torch.optim.Optimizer] = self.optimizers(use_pl_optimizer=True)
        schedulers: List[torch.optim.lr_scheduler.CyclicLR] = self.lr_schedulers()
        generator_opt = optimizers[0]
        generator_sched = schedulers[0]
        # output of encoder when evaluating a real image
        encoder_outs_real, encoder_mare_outs_real, decoder_outs_real, normals_real = self(z, generator=True)
        depth_out = decoder_outs_real[-1]

        # compare output levels to make sure they produce roughly the same output
        residual_loss: Tensor = 0
        if self.config.encoder.residual_learning:
            residual_loss = torch.mean(torch.stack(encoder_mare_outs_real)) * self.config.residual_loss_factor
            self.g_losses_log['g_res_loss'] += residual_loss

        g_loss: Tensor = 0
        feat_outs = encoder_outs_real[::-1][:len(self.d_feat_modules)]
        for idx, feature_out in enumerate(feat_outs):
            real_predicted = self.d_feat_modules[idx](feature_out).type_as(feature_out)
            g_loss += self._apply_generator_loss(real_predicted, f'feature_{idx}',
                                                 apply_uncertainty=False) * self.config.feature_discriminator_factor

        valid_predicted_depth = self.d_img(depth_out)
        g_loss += self._apply_generator_loss(valid_predicted_depth, 'depth_img',
                                             apply_uncertainty=False) * self.config.img_discriminator_factor

        if self.config.predict_normals:
            synth_phong_rendering = self.phong_renderer((depth_out, normals_real))
            phong_discrimination = self.phong_discriminator(synth_phong_rendering)
            g_loss += self._apply_generator_loss(phong_discrimination, 'phong',
                                                 apply_uncertainty=False) * self.config.phong_discriminator_factor

            calculated_norms = depth_to_normals(depth_out, self.phong_renderer.camera_intrinsics[None],
                                                self.phong_renderer.resized_pixel_locations)
            depth_phong = self.phong_renderer((depth_out, calculated_norms))
            g_loss += self._apply_generator_loss(self.depth_phong_discriminator(depth_phong), 'depth_phong',
                                                 apply_uncertainty=False) * self.config.phong_discriminator_factor

        g_loss += residual_loss

        self.manual_backward(g_loss)
        self.g_losses_log['g_loss'] += g_loss
        self.generator_global_step += 1
        step_optimizers = self.generator_global_step % self.config.accumulate_grad_batches == 0
        if step_optimizers:
            generator_opt.step()
            generator_sched.step()
            generator_opt.zero_grad()
            self.log_dict(self.g_losses_log)
            self.g_losses_log.update({k: 0 for k in self.g_losses_log.keys()})

    def discriminators_train_step(self, batch, batch_idx) -> None:
        # x = synthetic image, z = real image
        x, z = batch
        self.generator.eval()
        # set discriminators to train because idk if they were set back after generator step
        self.d_img.train()
        self.d_feat_modules.train()
        self.depth_phong_discriminator.train()
        self.phong_discriminator.train()

        optimizers: List[torch.optim.Optimizer] = self.optimizers(use_pl_optimizer=True)
        schedulers: List[torch.optim.lr_scheduler.CyclicLR] = self.lr_schedulers()
        discriminator_opts = [optimizers[i] for i in range(1, len(optimizers))]
        discriminator_sched = [schedulers[i] for i in range(1, len(schedulers))]

        with torch.no_grad():
            # predictions with real images through generator
            encoder_outs_real, encoder_mare_outs_real, decoder_outs_real, normals_real = self(z, generator=True)
            # predictions with synthetic images through frozen network
            encoder_outs_synth, encoder_mare_outs_synth, decoder_outs_synth, normals_synth = self(x, generator=False)

            depth_real = decoder_outs_real[-1]
            depth_synth = decoder_outs_synth[-1]

            if self.config.predict_normals:
                phong_synth = self.phong_renderer((depth_synth, normals_synth))
                phong_real = self.phong_renderer((depth_real, normals_real))
                calculated_phong_synth = self.phong_renderer((depth_synth,
                                                              depth_to_normals(depth_synth,
                                                                               self.phong_renderer.camera_intrinsics[
                                                                                   None],
                                                                               self.phong_renderer.resized_pixel_locations)))
                calculated_phong_real = self.phong_renderer((depth_real,
                                                             depth_to_normals(depth_real,
                                                                              self.phong_renderer.camera_intrinsics[
                                                                                  None],
                                                                              self.phong_renderer.resized_pixel_locations)))

        d_loss: Tensor = 0
        d_loss += self._apply_discriminator_loss(depth_real.detach(), depth_synth.detach(), self.d_img, 'depth_img')
        feat_outs = zip(encoder_outs_real[::-1], encoder_outs_synth[::-1])
        for idx, d_feat in enumerate(self.d_feat_modules):
            feature_out_r, feature_out_s = next(feat_outs)
            d_loss += self._apply_discriminator_loss(feature_out_r.detach(), feature_out_s.detach(), d_feat,
                                                     f'feature_{idx}')

        if self.config.predict_normals:
            d_loss += self._apply_discriminator_loss(phong_real.detach(), phong_synth.detach(),
                                                     self.phong_discriminator, 'phong')
            d_loss += self._apply_discriminator_loss(calculated_phong_real.detach(),
                                                     calculated_phong_synth.detach(),
                                                     self.depth_phong_discriminator,
                                                     'depth_phong')

        self.manual_backward(d_loss)
        self.d_losses_log['d_loss'] += d_loss

        self.discriminators_global_step += 1
        step_optimizers = self.discriminators_global_step % self.config.accumulate_grad_batches == 0
        if step_optimizers:
            [d_opt.step() for d_opt in discriminator_opts]
            [d_opt.zero_grad() for d_opt in discriminator_opts]
            [d_sched.step() for d_sched in discriminator_sched]
            self.log_dict(self.d_losses_log)
            self.d_losses_log.update({k: 0 for k in self.d_losses_log.keys()})

    def _apply_generator_loss(self, discriminator_out: Tensor, name: str, label_as_good: bool = True,
                              apply_uncertainty: bool = True) -> Tensor:
        if self.config.loss == 'cross_entropy':
            label_func = torch.ones_like if label_as_good else torch.zeros_like
            label = label_func(discriminator_out, device=discriminator_out.device, dtype=discriminator_out.dtype)
            confidence = self.config.d_max_conf if apply_uncertainty else 1.0
            loss = self.generator_loss(discriminator_out, label * confidence)
        else:
            loss = self.generator_loss(discriminator_out)
        self.g_losses_log[f'g_loss_{name}'] += loss
        return loss

    def _apply_discriminator_loss(self, real: Tensor, synth: Tensor, discriminator: torch.nn.Module,
                                  name: str) -> Tensor:
        synth_depth_discriminated = discriminator(synth)
        real_depth_discriminated = discriminator(real)

        if self.config.loss == 'cross_entropy':
            # label the real data as zeros and synth as ones
            real_label = torch.zeros_like(real_depth_discriminated,
                                          device=self.device,
                                          dtype=synth_depth_discriminated.dtype) * self.config.d_max_conf
            synth_label = torch.ones_like(synth_depth_discriminated,
                                          device=self.device,
                                          dtype=real_depth_discriminated.dtype)

            synth_loss = self.discriminator_loss(synth_depth_discriminated, synth_label)
            real_loss = self.discriminator_loss(real_depth_discriminated, real_label)
            # discriminator loss is the average of these
            d_loss = (real_loss + synth_loss) / 2
        else:
            d_loss = self.discriminator_loss(synth,
                                             real,
                                             discriminator,
                                             self.config.wasserstein_lambda)

        self.d_losses_log[f'd_loss_{name}'] += d_loss
        return d_loss

    def validation_step(self, batch, batch_idx):
        """
        TODO: This function only does plotting... We need some sort of metric
        :param batch:
        :param batch_idx:
        :return:
        """
        self.eval()
        if batch_idx != 0:
            # just plot one batch worth of images. In case there are a lot...
            return
        x, z = batch
        # predictions with real images through generator
        self.generator.apply(freeze_batchnorm)
        _, _, decoder_outs_adapted, normals_adapted = self(z, generator=True)
        depth_adapted = decoder_outs_adapted[-1]

        if self.unadapted_images_for_plotting is None:
            _, _, decoder_outs_unadapted, normals_unadapted = self(z, generator=False)
            depth_unadapted = decoder_outs_unadapted[-1].detach()
            if normals_unadapted is not None:
                phong_unadapted = self.phong_renderer((depth_unadapted, normals_unadapted)).cpu()
            else:
                phong_unadapted = None
            self.unadapted_images_for_plotting = (depth_unadapted, normals_unadapted.detach(), phong_unadapted.detach())

        depth_unadapted, normals_unadapted, phong_unadapted = self.unadapted_images_for_plotting
        denormed_images = self.imagenet_denorm(z).cpu()
        plot_tensors = [denormed_images]
        labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
        centers = [None, None, None, 0]
        minmax = []
        plot_tensors.append(depth_adapted)
        plot_tensors.append(depth_unadapted.cpu())
        plot_tensors.append((depth_adapted - depth_unadapted).cpu())

        self.log_gate_coefficients(step=self.global_step)
        for idx, imgs in enumerate(zip(*plot_tensors)):
            fig = generate_heatmap_fig(imgs, labels=labels, centers=centers, minmax=minmax,
                                       align_scales=False)
            self.logger.experiment.add_figure(f"GAN Prediction Result-{idx}", fig, self.global_step)
            plt.close(fig)

        if normals_adapted is not None:
            phong_adapted = self.phong_renderer((depth_adapted, normals_adapted)).cpu()

            labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted"]
            for idx, img_set in enumerate(zip(denormed_images, phong_adapted, phong_unadapted)):
                fig = generate_img_fig(img_set, labels)
                self.logger.experiment.add_figure(f'GAN-phong-{idx}', fig, self.global_step)
                plt.close(fig)

    def test_step(self, batch, batch_idx):
        x, z = batch
        y_hat = self.depth_model.decoder(self.generator(z)[0])
        img_unapdated = self.depth_model(z)[-1][:, 0, ...].unsqueeze(1)
        img_adapted = y_hat[-1][:, 0, ...].unsqueeze(1)
        plot_tensors = [self.imagenet_denorm(z)]
        # no labels for test step
        labels = ["", "", "", ""]
        centers = [None, None, None, 0]
        minmax = []

        plot_tensors.append(img_adapted)
        plot_tensors.append(img_unapdated)
        plot_tensors.append(img_adapted - img_unapdated)

        for idx, imgs in enumerate(zip(*plot_tensors)):
            fig = generate_final_imgs(imgs,
                                      labels=labels,
                                      centers=centers,
                                      minmax=minmax,
                                      colorbars=[False, False, False, True, True],
                                      align_scales=True,
                                      savefigs=True,
                                      figname=str(batch_idx) + str(idx))
            self.logger.experiment.add_figure("GAN Test Result-{}-{}".format(batch_idx, idx), fig, self.global_step)
            plt.close(fig)
            fig = generate_heatmap_fig(imgs, labels=labels, centers=centers, minmax=minmax,
                                       align_scales=True)
            self.logger.experiment.add_figure("GAN Prediction Result-{}-{}".format(batch_idx, idx), fig,
                                              self.global_step)
            plt.close(fig)

    def configure_optimizers(self):
        lr_d = self.config.discriminator_lr
        lr_g = self.config.generator_lr
        b1 = self.config.beta_1
        b2 = self.config.beta_2

        opts = {'adam': torch.optim.Adam, 'radam': torch.optim.RAdam, 'rmsprop': torch.optim.RMSprop}
        opt = opts[self.config.optimizer.lower()]
        if self.config.optimizer.lower() == 'rmsprop':
            opt_g = opt(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=lr_g)
            opt_d_feature = [opt(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr_d)
                             for discriminator in self.d_feat_modules]
            opt_d_img = opt(filter(lambda p: p.requires_grad, self.d_img.parameters()), lr=lr_d)
        else:
            opt_g = opt(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=lr_g, betas=(b1, b2))
            opt_d_feature = [opt(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr_d,
                                 betas=(b1, b2)) for discriminator in self.d_feat_modules]
            opt_d_img = opt(filter(lambda p: p.requires_grad, self.d_img.parameters()), lr=lr_d, betas=(b1, b2))

        up_steps = self.config.cyclic_step_period // 2
        lr_scheduler_g = torch.optim.lr_scheduler.CyclicLR(opt_g, base_lr=lr_g, max_lr=lr_g * 10, gamma=.1,
                                                           cycle_momentum=False, step_size_up=up_steps)
        lr_scheduler_d_img = torch.optim.lr_scheduler.CyclicLR(opt_d_img, base_lr=lr_g, max_lr=lr_g * 10, gamma=.1,
                                                               cycle_momentum=False, step_size_up=up_steps)
        lr_schedulers_d_feat = [torch.optim.lr_scheduler.CyclicLR(opt,
                                                                  base_lr=lr_g,
                                                                  max_lr=lr_g * 10,
                                                                  gamma=.1,
                                                                  cycle_momentum=False, step_size_up=up_steps)
                                for opt in
                                opt_d_feature]
        optimizers, schedulers = [opt_g, opt_d_img, *opt_d_feature], \
                                 [lr_scheduler_g, lr_scheduler_d_img, *lr_schedulers_d_feat]
        self.feat_idx_start = 2
        if self.config.predict_normals:
            opt_d_phong = torch.optim.Adam(filter(lambda p: p.requires_grad, self.phong_discriminator.parameters()),
                                           lr=lr_d, betas=(b1, b2))
            lr_scheduler_d_phong = torch.optim.lr_scheduler.CyclicLR(opt_d_phong,
                                                                     base_lr=lr_g,
                                                                     max_lr=lr_g * 10,
                                                                     gamma=.1,
                                                                     cycle_momentum=False, step_size_up=up_steps)
            optimizers.insert(2, opt_d_phong)
            schedulers.insert(2, lr_scheduler_d_phong)
            self.feat_idx_start += 1

        return optimizers, schedulers

    def _on_epoch_end(self):
        if self.lr_schedulers() is not None:
            self.log("generator_lr", self.lr_schedulers()[0].get_last_lr()[0])
            self.log("discriminator_lr", self.lr_schedulers()[1].get_last_lr()[0])

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end()

    def log_gate_coefficients(self, step=None):
        if step is None:
            step = self.current_epoch
        # iterating through all parameters
        for name, params in self.generator.named_parameters():
            if 'gate_coefficients' in name:
                scalars = {str(i): params[i] for i in range(len(params))}
                self.logger.experiment.add_scalars(name, scalars, step)
