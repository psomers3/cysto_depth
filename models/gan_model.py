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

opts = {'adam': torch.optim.Adam, 'radam': torch.optim.RAdam, 'rmsprop': torch.optim.RMSprop}


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
        self.imagenet_denorm = ImageNetNormalization(inverse=True)
        self.phong_renderer: PhongRender = None
        self.cosine_sim: CosineSimilarity = None
        self.d_losses_log = {'d_loss': 0.0}
        self.g_losses_log = {'g_loss': 0.0}
        self.discriminators: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.critics: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.feat_idx_start: int = 0
        self.critic_opt_idx = 0
        self.discriminators_opt_idx = 0
        self._unwrapped_optimizers = []
        self.setup_generator_optimizer()
        if self.config.use_critic:
            self.setup_critics()
        if self.config.use_discriminator:
            self.setup_discriminators()

        self.generator_global_step = -1
        self.critic_global_step = 0
        self.total_train_step_count = -1
        self.batches_accumulated = 0
        self._generator_training = False
        self.unadapted_images_for_plotting = None
        self.discriminator_loss = GANDiscriminatorLoss[gan_config.discriminator_loss]
        self.critic_loss = GANDiscriminatorLoss[gan_config.critic_loss]
        self.generator_critic_loss = GANGeneratorLoss[gan_config.critic_loss]
        self.generator_discriminator_loss = GANGeneratorLoss[gan_config.discriminator_loss]
        self.check_for_generator_step = self.config.wasserstein_critic_updates + 1

        if gan_config.resume_from_checkpoint:
            path_to_ckpt = gan_config.resume_from_checkpoint
            gan_config.resume_from_checkpoint = ""  # set empty or a recursive loading problem occurs
            ckpt = self.load_from_checkpoint(path_to_ckpt,
                                             strict=False,
                                             synth_config=synth_config,
                                             gan_config=gan_config)
            self.load_state_dict(ckpt.state_dict())

    def setup_generator_optimizer(self):
        opt = opts[self.config.generator_optimizer.lower()]
        self._unwrapped_optimizers.append(opt(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                              lr=self.config.generator_lr))

    def setup_discriminators(self):
        d_in_shapes = self.generator.feature_levels[::-1]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:-1]:
            d_config: DiscriminatorConfig = self.config.feature_level_discriminator.copy()
            d_config.in_channels = d_in_shape
            d = Discriminator(d_config)
            d_feat_list.append(d)
        self.d_losses_log['d_discriminators_loss'] = 0.0
        self.discriminators['features'] = torch.nn.ModuleList(modules=d_feat_list)
        self.d_losses_log.update({f'd_loss_discriminator_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        self.g_losses_log.update({f'g_loss_discriminator_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        if self.config.predict_normals:
            self.discriminators['phong'] = Discriminator(self.config.phong_discriminator)
            self.discriminators['depth_phong'] = Discriminator(self.config.phong_discriminator)
            self.d_losses_log.update({'d_loss_discriminator_phong': 0.0, 'd_loss_discriminator_depth_phong': 0.0})
            self.g_losses_log.update({'g_loss_discriminator_phong': 0.0, 'g_loss_discriminator_depth_phong': 0.0})
        self.discriminators['depth_image'] = Discriminator(self.config.depth_discriminator)
        self.d_losses_log.update({'d_loss_discriminator_depth_img': 0.0})
        self.g_losses_log.update({'g_loss_discriminator_depth_img': 0.0})
        opt = opts[self.config.discriminator_optimizer.lower()]
        self._unwrapped_optimizers.append(opt(filter(lambda p: p.requires_grad, self.discriminators.parameters()),
                                              lr=self.config.discriminator_lr))
        self.discriminators_opt_idx = self.critic_opt_idx + 1

    def setup_critics(self):
        d_in_shapes = self.generator.feature_levels[::-1]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:-1]:
            d_config: DiscriminatorConfig = self.config.feature_level_critic.copy()
            d_config.in_channels = d_in_shape
            d = Discriminator(d_config)
            d_feat_list.append(d)
        self.d_losses_log['d_critics_loss'] = 0.0
        self.critics['features'] = torch.nn.ModuleList(modules=d_feat_list)
        self.d_losses_log.update({f'd_loss_critic_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        self.g_losses_log.update({f'g_loss_critic_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        if self.config.predict_normals:
            self.critics['phong'] = Discriminator(self.config.phong_critic)
            self.critics['depth_phong'] = Discriminator(self.config.phong_critic)
            self.d_losses_log.update({'d_loss_critic_phong': 0.0, 'd_loss_critic_depth_phong': 0.0})
            self.g_losses_log.update({'g_loss_critic_phong': 0.0, 'g_loss_critic_depth_phong': 0.0})
        self.critics['depth_image'] = Discriminator(self.config.depth_critic)
        self.d_losses_log.update({'d_loss_critic_depth_img': 0.0})
        self.g_losses_log.update({'g_loss_critic_depth_img': 0.0})
        opt = opts[self.config.critic_optimizer.lower()]
        self._unwrapped_optimizers.append(opt(filter(lambda p: p.requires_grad, self.critics.parameters()),
                                              lr=self.config.critic_lr))
        self.critic_opt_idx += 1

    @staticmethod
    def reset_log_dict(log_dict: dict):
        log_dict.update({k: 0.0 for k in log_dict.keys()})

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

        self.total_train_step_count += 1
        if self._generator_training:
            self.generator_train_step(batch, batch_idx)
        else:
            self.discriminator_critic_train_step(batch, batch_idx)

    def generator_train_step(self, batch, batch_idx) -> None:
        # x = synthetic image, z = real image
        _, z = batch
        # set discriminators to eval so that any normalization statistics don't get updated
        self.discriminators.eval()
        self.critics.eval()
        if self.config.encoder.residual_learning:
            self.generator.set_residuals_train()

        # output of encoder when evaluating a real image
        encoder_outs_real, encoder_mare_outs_real, decoder_outs_real, normals_real = self(z, generator=True)
        depth_out = decoder_outs_real[-1]

        g_loss: Tensor = 0.0

        if self.config.predict_normals:
            synth_phong_rendering = self.phong_renderer((depth_out, normals_real))
            calculated_norms = depth_to_normals(depth_out, self.phong_renderer.camera_intrinsics[None],
                                                self.phong_renderer.resized_pixel_locations)
            depth_phong = self.phong_renderer((depth_out, calculated_norms))

        if self.config.use_discriminator:
            feat_outs = encoder_outs_real[::-1][:len(self.discriminators['features'])]
            for idx, feature_out in enumerate(feat_outs):
                real_predicted = self.discriminators['features'][idx](feature_out).type_as(feature_out)
                g_loss += self._apply_generator_discriminator_loss(real_predicted, f'discriminator_feature_{idx}') \
                          * self.config.feature_discriminator_factor
            valid_predicted_depth = self.discriminators['depth_image'](depth_out)
            g_loss += self._apply_generator_discriminator_loss(valid_predicted_depth, 'discriminator_depth_img') \
                      * self.config.img_discriminator_factor
            if self.config.predict_normals:
                phong_discrimination = self.discriminators['phong'](synth_phong_rendering)
                g_loss += self._apply_generator_discriminator_loss(phong_discrimination, 'discriminator_phong') \
                          * self.config.phong_discriminator_factor
                g_loss += self._apply_generator_discriminator_loss(self.discriminators['depth_phong'](depth_phong),
                                                                   'discriminator_depth_phong') * self.config.phong_discriminator_factor

        if self.config.use_critic:
            feat_outs = encoder_outs_real[::-1][:len(self.critics['features'])]
            for idx, feature_out in enumerate(feat_outs):
                real_predicted = self.critics['features'][idx](feature_out).type_as(feature_out)
                g_loss += self._apply_generator_critic_loss(real_predicted, f'critic_feature_{idx}') \
                          * self.config.feature_discriminator_factor
            valid_predicted_depth = self.critics['depth_image'](depth_out)
            g_loss += self._apply_generator_critic_loss(valid_predicted_depth, 'critic_depth_img') \
                      * self.config.img_discriminator_factor

            if self.config.predict_normals:
                phong_discrimination = self.critics['phong'](synth_phong_rendering)
                g_loss += self._apply_generator_critic_loss(phong_discrimination, 'critic_phong') \
                          * self.config.phong_discriminator_factor
                g_loss += self._apply_generator_critic_loss(self.critics['depth_phong'](depth_phong), 'critic_depth_phong') \
                          * self.config.phong_discriminator_factor

        self.manual_backward(g_loss)
        self.g_losses_log['g_loss'] += g_loss
        self.batches_accumulated += 1
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.generator_global_step += 1
            self.batches_accumulated = 0
            self._generator_training = False
            generator_opt = self.optimizers(True)[0]
            generator_opt.step()
            generator_opt.zero_grad()
            self.log_dict(self.g_losses_log)
            self.reset_log_dict(self.g_losses_log)

    def discriminator_critic_train_step(self, batch, batch_idx) -> None:
        self.generator.eval()
        # set discriminators to train because idk if they were set back after generator step
        self.critics.train()
        self.discriminators.train()

        predictions = self.get_discriminator_critic_inputs(batch, batch_idx)
        self.batches_accumulated += 1
        full_batch = self.batches_accumulated == self.config.accumulate_grad_batches
        if full_batch:
            self.batches_accumulated = 0
        first_of_mini_batches = self.critic_global_step % self.config.wasserstein_critic_updates == 0
        last_mini_batch = (self.critic_global_step + 1) % self.config.wasserstein_critic_updates == 0 \
            if self.config.use_critic else True

        if self.config.use_discriminator and first_of_mini_batches:
            discriminator_loss = self._discriminators(predictions)
            self.manual_backward(discriminator_loss)
            self.d_losses_log['d_discriminators_loss'] += discriminator_loss
            # +1 because they are stepped after critic update
            if full_batch:
                discriminator_opt = self.optimizers(True)[self.discriminators_opt_idx]
                discriminator_opt.step()
                discriminator_opt.zero_grad()

        if self.config.use_critic:
            critic_loss = self._critics(predictions)
            self.manual_backward(critic_loss)
            self.d_losses_log['d_critics_loss'] += critic_loss
            if full_batch:
                critic_opt = self.optimizers(True)[self.critic_opt_idx]
                critic_opt.step()
                critic_opt.zero_grad()
                self.critic_global_step += 1

        if last_mini_batch and full_batch:
            self._generator_training = True
            self.log_dict(self.d_losses_log)
            self.reset_log_dict(self.d_losses_log)

    def get_discriminator_critic_inputs(self, batch, batch_idx) -> Dict[str, Tensor]:
        """

        :param batch:
        :param batch_idx:
        :return: dict
        """
        x, z = batch
        results = {}
        with torch.no_grad():
            # predictions with real images through generator
            encoder_outs_real, encoder_mare_outs_real, decoder_outs_real, normals_real = self(z, generator=True)
            # predictions with synthetic images through frozen network
            encoder_outs_synth, encoder_mare_outs_synth, decoder_outs_synth, normals_synth = self(x, generator=False)

            depth_real = decoder_outs_real[-1]
            depth_synth = decoder_outs_synth[-1]
            results['encoder_outs_real'] = [e.detach() for e in encoder_outs_real]
            results['encoder_outs_synth'] = [e.detach() for e in encoder_outs_synth]
            results['depth_real'] = depth_real.detach()
            results['depth_synth'] = depth_synth.detach()
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
                results['phong_synth'] = phong_synth.detach()
                results['phong_real'] = phong_real.detach()
                results['calculated_phong_synth'] = calculated_phong_synth.detach()
                results['calculated_phong_real'] = calculated_phong_real.detach()
        return results

    def _discriminators(self, predictions: Dict[str, Tensor]) -> Tensor:
        depth_real = predictions['depth_real']
        depth_synth = predictions['depth_synth']
        encoder_outs_real = predictions['encoder_outs_real']
        encoder_outs_synth = predictions['encoder_outs_synth']
        loss: Tensor = 0.0
        loss += self._apply_discriminator_loss(depth_real,
                                               depth_synth,
                                               self.discriminators['depth_image'],
                                               'discriminator_depth_img')
        feat_outs = zip(encoder_outs_real[::-1], encoder_outs_synth[::-1])
        for idx, d_feat in enumerate(self.discriminators['features']):
            feature_out_r, feature_out_s = next(feat_outs)
            loss += self._apply_discriminator_loss(feature_out_r,
                                                   feature_out_s,
                                                   d_feat,
                                                   f'discriminator_feature_{idx}')
        if self.config.predict_normals:
            phong_real = predictions['phong_real']
            phong_synth = predictions['phong_synth']
            calculated_phong_real = predictions['calculated_phong_real']
            calculated_phong_synth = predictions['calculated_phong_synth']
            loss += self._apply_discriminator_loss(phong_real,
                                                   phong_synth,
                                                   self.discriminators['phong'],
                                                   'discriminator_phong')
            loss += self._apply_discriminator_loss(calculated_phong_real,
                                                   calculated_phong_synth,
                                                   self.discriminators['depth_phong'],
                                                   'discriminator_depth_phong')
        return loss

    def _critics(self, predictions: Dict[str, Tensor]) -> Tensor:
        depth_real = predictions['depth_real']
        depth_synth = predictions['depth_synth']
        encoder_outs_real = predictions['encoder_outs_real']
        encoder_outs_synth = predictions['encoder_outs_synth']
        loss: Tensor = 0.0
        loss += self._apply_critic_loss(depth_real, depth_synth, self.critics['depth_image'],
                                        self.config.wasserstein_lambda, 'critic_depth_img')
        feat_outs = zip(encoder_outs_real[::-1], encoder_outs_synth[::-1])
        for idx, feature_critic in enumerate(self.critics['features']):
            feature_out_r, feature_out_s = next(feat_outs)
            loss += self._apply_critic_loss(feature_out_r, feature_out_s, feature_critic,
                                            self.config.wasserstein_lambda, f'critic_feature_{idx}')
        if self.config.predict_normals:
            phong_real = predictions['phong_real']
            phong_synth = predictions['phong_synth']
            calculated_phong_real = predictions['calculated_phong_real']
            calculated_phong_synth = predictions['calculated_phong_synth']
            loss += self._apply_critic_loss(phong_real, phong_synth, self.critics['phong'],
                                            self.config.wasserstein_lambda, 'critic_phong')
            loss += self._apply_critic_loss(calculated_phong_real, calculated_phong_synth, self.critics['depth_phong'],
                                            self.config.wasserstein_lambda, 'critic_depth_phong')
        return loss

    def _apply_generator_discriminator_loss(self, discriminator_out: Tensor, name: str, label: float = 1.0) -> Tensor:
        loss = self.generator_discriminator_loss(discriminator_out, label)
        self.g_losses_log[f'g_loss_{name}'] += loss
        return loss

    def _apply_generator_critic_loss(self, discriminator_out: Tensor, name: str,) -> Tensor:
        loss = self.generator_critic_loss(discriminator_out)
        self.g_losses_log[f'g_loss_{name}'] += loss
        return loss

    def _apply_discriminator_loss(self, generated: Tensor, original: Tensor, discriminator: torch.nn.Module,
                                  name: str) -> Tensor:
        original_depth_discriminated = discriminator(original)
        generated_depth_discriminated = discriminator(generated)

        loss_generated = self.discriminator_loss(generated_depth_discriminated, 0.0)
        loss_original = self.discriminator_loss(original_depth_discriminated, 1.0)
        combined = (loss_original + loss_generated) / 2
        self.d_losses_log[f'd_loss_{name}'] += combined
        return combined

    def _apply_critic_loss(self, original: Tensor, generated: Tensor, critic: torch.nn.Module,
                           wasserstein_lambda: float, name: str):
        critic_loss = self.critic_loss(original, generated, critic, wasserstein_lambda)
        self.d_losses_log[f'd_loss_{name}'] += critic_loss
        return critic_loss

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
        return self._unwrapped_optimizers

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
        if self.config.encoder.adaptive_gating and self.config.encoder.residual_learning:
            for name, params in self.generator.named_parameters():
                if 'gate_coefficients' in name:
                    scalars = {str(i): params[i] for i in range(len(params))}
                    self.logger.experiment.add_scalars(name, scalars, step)
