from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from models.base_model import BaseModel
from models.vanillaencoder import VanillaEncoder
from models.discriminator_img import ImgDiscriminator
from models.adaptive_encoder import AdaptiveEncoder
from models.discriminator import Discriminator
from models.depth_model import DepthEstimationModel
from data.data_transforms import ImageNetNormalization
import socket
from utils.image_utils import generate_heatmap_fig, freeze_batchnorm, generate_final_imgs
from config.training_config import SyntheticTrainingConfig, GANTrainingConfig
from argparse import Namespace
from utils.rendering import PhongRender


class GAN(BaseModel):
    def __init__(
            self,
            synth_config: SyntheticTrainingConfig,
            gan_config: GANTrainingConfig,
            image_gan: bool = False,
    ):
        super().__init__()
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
        self.feat_idx_start: int = 0

    def forward(self, z, full_prediction=False):
        if full_prediction:
            encoder_outs, encoder_mare_outs = self.generator(z)
            return self.depth_model.decoder(encoder_outs)
        else:
            return self.generator(z)

    def on_validation_epoch_start(self) -> None:
        if self.config.predict_normals:
            self.phong_renderer = PhongRender(config=self.config.phong_config,
                                              image_size=self.config.image_size,
                                              device=self.device)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.depth_model.apply(freeze_batchnorm)
        if self.config.freeze_batch_norm:
            self.generator.apply(freeze_batchnorm)

        # x = synthetic image, z = real image
        x, z = batch

        # generator optimizer index = 0
        # depth map discriminator optimizer index = 1
        # feature level optimizers indexes = 2 - 5
        # phong img optimizer index = 6
        if optimizer_idx == 0 and self.global_step >= self.config.warmup_steps:
            # output of encoder when evaluating a real image
            encoder_outs, encoder_mare_outs = self.generator(z)
            if self.depth_model.config.predict_normals:
                if self.depth_model.config.merged_decoder:
                    decoder_outs_synth = self.depth_model.decoder(encoder_outs)
                    depth_out = decoder_outs_synth[-1][:, 0, ...].unsqueeze(1)
                    normals_out = decoder_outs_synth[-1][:, 1:, ...]
                else:
                    decoder_outs_synth, normals_out = self.depth_model.decoder(encoder_outs)
                    depth_out = decoder_outs_synth[-1]
            else:
                decoder_outs_synth = self.depth_model.decoder(encoder_outs)
                depth_out = decoder_outs_synth[-1]

            # compare output levels to make sure they produce roughly the same output
            if self.config.residual_transfer:
                residual_loss = torch.mean(torch.stack(encoder_mare_outs))
            else:
                residual_loss = torch.tensor([0]).type_as(z)

            scale_loss = 0
            # actual output of the discriminator
            g_losses_feat = []
            feat_outs = encoder_outs[::-1][:len(self.d_feat_modules)]
            for idx, feature_out in enumerate(feat_outs):
                real_predicted = self.d_feat_modules[idx](feature_out).type_as(feature_out)
                real = torch.ones(real_predicted.size(),
                                  device=self.device,
                                  dtype=feature_out.dtype) * self.config.d_max_conf
                loss = self.adversarial_loss(real_predicted, real)
                self.log("g_loss_feature_{}".format(idx), loss)
                g_losses_feat.append(loss)

            # Use ones as ground truth to get generator to figure out how to trick discriminator
            valid_predicted_depth = self.d_img(depth_out)
            g_img_label = torch.ones_like(valid_predicted_depth,
                                                            device=self.device,
                                                            dtype=valid_predicted_depth.dtype)
            g_loss_img = self.adversarial_loss(valid_predicted_depth, g_img_label)
            self.log("g_loss_img", g_loss_img)

            phong_loss = 0
            if self.config.predict_normals:
                synth_phong_rendering = self.phong_renderer((depth_out, normals_out))
                phong_discrimination = self.phong_discriminator(synth_phong_rendering)
                phong_loss = self.adversarial_loss(phong_discrimination, g_img_label)
                self.log("g_phong_loss", phong_loss)

            g_loss_feat = torch.sum(torch.stack(g_losses_feat))
            g_loss_img = g_loss_img
            g_loss = g_loss_feat \
                     + self.config.residual_loss_factor * residual_loss \
                     + self.config.img_discriminator_factor * g_loss_img \
                     + self.config.phong_discriminator_factor * phong_loss
            self.log("g_loss", g_loss)
            self.log("g_skip_loss", g_loss_feat)
            self.log("g_res_loss", residual_loss)
            self.log("g_scale_loss", scale_loss)
            return g_loss
        elif optimizer_idx > 0:  # discriminators
            # image level discriminator
            with torch.no_grad():
                if optimizer_idx == 1:
                    if self.config.predict_normals:
                        prediction_from_synth = self.depth_model(x)[0][-1].detach()
                        if self.depth_model.config.merged_decoder:
                            prediction_from_real = self.depth_model.decoder(self.generator(z)[0])[-1][:, 0, ...].unsqueeze(1).detach()
                        else:
                            prediction_from_real = self.depth_model.decoder(self.generator(z)[0])[-1].detach()
                    else:
                        prediction_from_synth, _ = self.depth_model(x)[-1].detach()
                        prediction_from_real = self.depth_model.decoder(self.generator(z)[0])[-1].detach()
                    d = self.d_img
                    name = "img"
                # phong discriminator
                elif optimizer_idx == 2 and self.config.predict_normals:
                    if self.depth_model.config.merged_decoder:
                        depth_out, normals_out = self.depth_model(x)
                        prediction_from_synth = self.phong_renderer((depth_out[-1], normals_out))
                        output = self.depth_model.decoder(self.generator(z)[0])
                        depth_out = output[-1][:, 0, ...].unsqueeze(1).detach()
                        normals_out = output[-1][:, 1:, ...].detach()
                        prediction_from_real = self.phong_renderer((depth_out, normals_out)).detach()
                    else:
                        decoder_outs_synth, normals_out = self.depth_model(x)
                        depth_out = decoder_outs_synth[-1]
                        prediction_from_synth = self.phong_renderer((depth_out, normals_out)).detach()
                        decoder_outs_synth = self.depth_model.decoder(self.generator(z)[0])
                        normals_out = self.depth_model.normals_decoder(self.generator(z)[0])
                        depth_out = decoder_outs_synth[-1]
                        prediction_from_real = self.phong_renderer((depth_out, normals_out)).detach()
                    d = self.phong_discriminator
                    name = "phong"
                # feature discriminators
                else:
                    decoder_outs_synth = self.depth_model.encoder(x)[0][::-1]
                    prediction_from_synth = decoder_outs_synth[optimizer_idx - self.feat_idx_start].detach()
                    decoder_outs_real = self.generator(z)[0][::-1]
                    # evaluate current generator with a real image and take bottleneck output
                    prediction_from_real = decoder_outs_real[optimizer_idx - self.feat_idx_start].detach()
                    d = self.d_feat_modules[optimizer_idx - self.feat_idx_start]
                    name = str(optimizer_idx - self.feat_idx_start)

            real_predicted = d(prediction_from_synth)
            fake_predicted = d(prediction_from_real)

            # how well can it label as real
            # TODO: move this allocation of ground truth labels somewhere else
            real = torch.ones_like(real_predicted,
                                   device=self.device,
                                   dtype=real_predicted.dtype) * self.config.d_max_conf
            fake = torch.zeros_like(fake_predicted,
                                    device=self.device,
                                    dtype=prediction_from_real.dtype)

            real_loss = self.adversarial_loss(real_predicted, real)
            fake_loss = self.adversarial_loss(fake_predicted, fake)
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log(f"d_loss_{name}", d_loss)
            if optimizer_idx == 1:
                self.accum_dlosses = 0
            self.accum_dlosses += d_loss.detach()
            if optimizer_idx == 5:
                self.log("d_loss", self.accum_dlosses)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, z = batch
        if self.config.predict_normals:
            depth_unadapted = self.depth_model(z)[0][-1].detach()
            if self.depth_model.config.merged_decoder:
                depth_adapted = self.depth_model.decoder(self.generator(z)[0])[-1][:, 0, ...].unsqueeze(1).detach()
            else:
                depth_adapted = self.depth_model.decoder(self.generator(z)[0])[-1]
        else:
            y_hat = self.depth_model.decoder(self.generator(z)[0])
            depth_unadapted = self.depth_model(z)[-1]
            depth_adapted = y_hat[-1]
        plot_tensors = [self.imagenet_denorm(z)]
        labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
        centers = [None, None, None, 0]
        minmax = []
        plot_tensors.append(depth_adapted)
        plot_tensors.append(depth_unadapted)
        plot_tensors.append(depth_adapted - depth_unadapted)

        self.add_histograms(step=self.global_step)
        for idx, imgs in enumerate(zip(*plot_tensors)):
            fig = generate_heatmap_fig(imgs, labels=labels, centers=centers, minmax=minmax,
                                       align_scales=True)
            self.logger.experiment.add_figure("GAN Prediction Result-{}-{}".format(batch_idx, idx), fig,
                                              self.global_step)
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
        lr_scheduler_g = torch.optim.lr_scheduler.CyclicLR(opt_g, base_lr=lr_g, max_lr=lr_g*10, gamma=.1)
        lr_scheduler_d_img = torch.optim.lr_scheduler.CyclicLR(opt_d_img, base_lr=lr_g,max_lr=lr_g*10,  gamma=.1)
        lr_schedulers_d_feat = [torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_g, max_lr=lr_g*10, gamma=.1) for opt in
                                opt_d_feature]
        optimizers, schedulers = [opt_g, opt_d_img, *opt_d_feature], \
                                 [lr_scheduler_g, lr_scheduler_d_img, *lr_schedulers_d_feat]
        self.feat_idx_start = 2
        if self.config.predict_normals:
            opt_d_phong = torch.optim.Adam(filter(lambda p: p.requires_grad, self.phong_discriminator.parameters()),
                                           lr=lr_d, betas=(b1, b2))
            lr_scheduler_d_phong = torch.optim.lr_scheduler.CyclicLR(opt_d_phong, base_lr=lr_g, max_lr=lr_g*10, gamma=.1)
            optimizers.insert(2, opt_d_phong)
            schedulers.insert(2, lr_scheduler_d_phong)
            self.feat_idx_start += 1
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
