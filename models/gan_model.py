from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from models.base_model import BaseModel
from models.discriminator_img import ImgDiscriminator
from models.adaptive_encoder import AdaptiveEncoder
from models.discriminator import Discriminator
from models.depth_model import DepthEstimationModel
from data.data_transforms import ImageNetNormalization
import socket
from utils.image_utils import generate_heatmap_fig, freeze_batchnorm, generate_final_imgs, generate_img_fig
from config.training_config import SyntheticTrainingConfig, GANTrainingConfig
from argparse import Namespace
from utils.rendering import PhongRender, depth_to_normals
from utils.loss import CosineSimilarity
from typing import *


class GAN(BaseModel):
    def __init__(
            self,
            synth_config: SyntheticTrainingConfig,
            gan_config: GANTrainingConfig,
            image_gan: bool = False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(Namespace(**gan_config))
        synth_config.resume_from_checkpoint = ''
        self.depth_model: DepthEstimationModel = DepthEstimationModel.load_from_checkpoint(
            gan_config.synthetic_base_model,
            strict=False,
            config=synth_config)
        self.config = gan_config
        self.generator = AdaptiveEncoder(adaptive_gating=gan_config.adaptive_gating,
                                         backbone=synth_config.backbone,
                                         residual_learning=gan_config.residual_learning)

        self.generator.load_state_dict(self.depth_model.encoder.state_dict(), strict=False)
        self.depth_model.requires_grad = False

        d_in_shapes = self.generator.feature_levels[::-1]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:-1]:
            d = Discriminator(in_channels=d_in_shape, single_out=image_gan)
            d_feat_list.append(d)
        self.d_img = ImgDiscriminator(in_channels=1)
        self.d_feat_modules = torch.nn.ModuleList(modules=d_feat_list)
        self.gan = True
        self.imagenet_denorm = ImageNetNormalization(inverse=True)
        self.phong_renderer: PhongRender = None
        self.phong_discriminator = ImgDiscriminator(in_channels=3)
        self.depth_phong_discriminator = ImgDiscriminator(in_channels=3)
        self.cosine_sim: CosineSimilarity =  None
        self.feat_idx_start: int = 0
        # TODO: make the log dictionaries TypedDicts and define them elsewhere with comments
        self.d_losses_log = {'d_loss': 0, 'd_loss_img': 0, 'd_loss_phong': 0, 'd_loss_depth_phong': 0}
        self.d_losses_log.update({f'd_loss_feature_{i}': 0 for i in range(len(d_feat_list))})
        self.g_losses_log = {'g_loss': 0, 'g_loss_img': 0, 'g_loss_phong': 0, 'g_feat_loss': 0, 'g_res_loss': 0,
                             'g_neg_loss': 0, 'g_loss_depth_phong': 0, 'g_cosine_sim': 0}
        self.g_losses_log.update({f'g_loss_feature_{i}': 0 for i in range(len(d_feat_list))})
        self.generator_global_step = -1
        self.discriminators_global_step = -1
        self.unadapted_images_for_plotting = None

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

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

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

    def generator_train_step(self, batch, batch_idx) -> None:
        # x = synthetic image, z = real image
        _, z = batch

        # set discriminators to eval so that any normalization statistics don't get updated
        self.d_img.eval()
        self.d_feat_modules.eval()
        self.depth_phong_discriminator.eval()
        self.phong_discriminator.eval()

        optimizers: List[torch.optim.Optimizer] = self.optimizers(use_pl_optimizer=True)
        schedulers: List[torch.optim.lr_scheduler.CyclicLR] = self.lr_schedulers()
        generator_opt = optimizers[0]
        generator_sched = schedulers[0]
        # output of encoder when evaluating a real image
        encoder_outs_real, encoder_mare_outs_real, decoder_outs_real, normals_real = self(z, generator=True)
        depth_out = decoder_outs_real[-1]
        # compare output levels to make sure they produce roughly the same output
        residual_loss = torch.Tensor([0]).to(z.device)
        if self.config.residual_learning:
            residual_loss = torch.mean(torch.stack(encoder_mare_outs_real))

        # actual output of the discriminator
        g_losses_feat = []
        feat_outs = encoder_outs_real[::-1][:len(self.d_feat_modules)]
        for idx, feature_out in enumerate(feat_outs):
            real_predicted = self.d_feat_modules[idx](feature_out).type_as(feature_out)
            real = torch.ones_like(real_predicted,
                                   device=self.device,
                                   dtype=feature_out.dtype) * self.config.d_max_conf
            loss = self.adversarial_loss(real_predicted, real)
            self.g_losses_log[f'g_loss_feature_{idx}'] += loss.detach()
            g_losses_feat.append(loss)

        # Use ones (="synth image") as ground truth to get generator to figure out how to trick discriminator
        valid_predicted_depth = self.d_img(depth_out)
        g_img_label = torch.ones_like(valid_predicted_depth,
                                      device=self.device,
                                      dtype=valid_predicted_depth.dtype)
        g_loss_img = self.adversarial_loss(valid_predicted_depth, g_img_label)
        self.g_losses_log[f'g_loss_img'] += g_loss_img.detach()

        phong_loss = 0
        cosine_loss = 0
        if self.config.predict_normals:
            synth_phong_rendering = self.phong_renderer((depth_out, normals_real))
            phong_discrimination = self.phong_discriminator(synth_phong_rendering)
            phong_loss = self.adversarial_loss(phong_discrimination, g_img_label)
            self.g_losses_log[f'g_loss_phong'] += phong_loss.detach()

            calculated_norms = depth_to_normals(depth_out, self.phong_renderer.camera_intrinsics[None],
                                                self.phong_renderer.resized_pixel_locations)
            depth_phong = self.phong_renderer((depth_out, calculated_norms))
            depth_phong_loss = self.adversarial_loss(self.depth_phong_discriminator(depth_phong), g_img_label)
            self.g_losses_log[f'g_loss_depth_phong'] += depth_phong_loss.detach()
            phong_loss += depth_phong_loss
            cosine_loss = self.cosine_sim(calculated_norms, normals_real)
            self.g_losses_log['g_cosine_sim'] += cosine_loss.detach()

        g_loss_feat = torch.sum(torch.stack(g_losses_feat))
        g_loss = + self.config.feature_discriminator_factor * g_loss_feat \
                 + self.config.residual_loss_factor * residual_loss \
                 + self.config.img_discriminator_factor * g_loss_img \
                 + self.config.phong_discriminator_factor * phong_loss \
                 + cosine_loss

        self.g_losses_log['g_loss'] += g_loss.detach()
        self.g_losses_log['g_feat_loss'] += g_loss_feat.detach()
        self.g_losses_log['g_res_loss'] += residual_loss.detach()

        self.manual_backward(g_loss)
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

        d_loss = 0
        d_loss += self._apply_discriminator_loss(depth_real, depth_synth, self.d_img, 'img')
        feat_outs = zip(encoder_outs_real[::-1], encoder_outs_synth[::-1])
        for idx, d_feat in enumerate(self.d_feat_modules):
            feature_out_r, feature_out_s = next(feat_outs)
            d_loss += self._apply_discriminator_loss(feature_out_r, feature_out_s, d_feat, f'feature_{idx}')

        if self.config.predict_normals:
            d_loss += self._apply_discriminator_loss(phong_real, phong_synth, self.phong_discriminator, 'phong')
            d_loss += self._apply_discriminator_loss(calculated_phong_real,
                                                     calculated_phong_synth,
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

    def training_step(self, batch, batch_idx):
        # start by freezing all batchnorm layers throughout the networks that shouldn't update statistics
        self.depth_model.apply(freeze_batchnorm)
        if self.config.freeze_batch_norm:
            self.generator.apply(freeze_batchnorm)
            if self.config.residual_learning:
                self.generator.set_residuals_train()

        generator_step = batch_idx % 2 == 0 if self.global_step >= self.config.warmup_steps else False
        if generator_step:
            self.generator_train_step(batch, batch_idx)
        else:
            self.discriminators_train_step(batch, batch_idx)

    def _apply_discriminator_loss(self, real: torch.Tensor, synth: torch.Tensor, discriminator: Callable,
                                  name: str) -> torch.Tensor:
        synth_depth_discriminated = discriminator(synth)
        real_depth_discriminated = discriminator(real)

        # label the real data as zeros and synth as ones
        real_label = torch.zeros_like(real_depth_discriminated,
                                      device=self.device,
                                      dtype=synth_depth_discriminated.dtype) * self.config.d_max_conf
        synth_label = torch.ones_like(synth_depth_discriminated,
                                      device=self.device,
                                      dtype=real_depth_discriminated.dtype)

        synth_loss = self.adversarial_loss(synth_depth_discriminated, synth_label)
        real_loss = self.adversarial_loss(real_depth_discriminated, real_label)
        # discriminator loss is the average of these
        d_loss = (real_loss + synth_loss) / 2
        self.d_losses_log[f'd_loss_{name}'] += d_loss.detach()
        return d_loss

    def validation_step(self, batch, batch_idx):
        """
        TODO: This function only does plotting... We need some sort of metric
        :param batch:
        :param batch_idx:
        :return:
        """
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
            depth_unadapted = decoder_outs_unadapted[-1]
            if normals_unadapted is not None:
                phong_unadapted = self.phong_renderer((depth_unadapted, normals_unadapted)).cpu()
            else:
                phong_unadapted = None
            self.unadapted_images_for_plotting = (depth_unadapted, normals_unadapted.cpu(), phong_unadapted.cpu())

        depth_unadapted, normals_unadapted, phong_unadapted = self.unadapted_images_for_plotting
        denormed_images = self.imagenet_denorm(z).cpu()
        plot_tensors = [denormed_images]
        labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
        centers = [None, None, None, 0]
        minmax = []
        plot_tensors.append(depth_adapted)
        plot_tensors.append(depth_unadapted.cpu())
        plot_tensors.append((depth_adapted - depth_unadapted).cpu())

        self.add_histograms(step=self.global_step)
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

        opt_g = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=lr_g,
                                 betas=(b1, b2))
        opt_d_feature = [torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                          lr=lr_d,
                                          betas=(b1, b2)) for discriminator in self.d_feat_modules]
        opt_d_img = torch.optim.Adam(filter(lambda p: p.requires_grad, self.d_img.parameters()), lr=lr_d,
                                     betas=(b1, b2))
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

        schedulers = [{'scheduler': s, 'interval': 'step', 'frequency': 1, 'strict': False} for s in schedulers]
        return optimizers, schedulers

    def _on_epoch_end(self):
        if self.lr_schedulers() is not None:
            self.log("generator_lr", self.lr_schedulers()[0].get_last_lr()[0])
            self.log("discriminator_lr", self.lr_schedulers()[1].get_last_lr()[0])
        # self.log(socket.gethostname(), True)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_training_end(self):
        self.log(socket.gethostname(), False)

    def add_histograms(self, step=None):
        if step is None:
            step = self.current_epoch
        # iterating through all parameters
        for name, params in self.generator.named_parameters():
            if 'gate_coefficients' in name:
                scalars = {str(i): params[i] for i in range(len(params))}
                self.logger.experiment.add_scalars(name, scalars, step)
