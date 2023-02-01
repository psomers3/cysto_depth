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
        self.depth_model = DepthEstimationModel.load_from_checkpoint(gan_config.synthetic_base_model,
                                                                     strict=False,
                                                                     config=synth_config)
        self.config = gan_config
        self.generator = AdaptiveEncoder(gan_config.adaptive_gating, backbone=synth_config.backbone) \
            if gan_config.residual_transfer else VanillaEncoder(backbone=synth_config.backbone)

        self.generator.load_state_dict(self.depth_model.encoder.state_dict(), strict=False)
        self.depth_model.requires_grad = False

        d_in_shapes = [512, 256, 128, 64, 64]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:3]:
            d = Discriminator(in_channels=d_in_shape, single_out=image_gan)
            d_feat_list.append(d)
        self.d_img = ImgDiscriminator(in_shape=1)
        self.d_feat_modules = torch.nn.ModuleList(modules=d_feat_list)
        self.gan = True
        self.imagenet_denorm = ImageNetNormalization(inverse=True)

    def forward(self, z, full_prediction=False):
        if full_prediction:
            encoder_outs, encoder_mare_outs = self.generator(z)
            return self.depth_model.decoder(encoder_outs)
        else:
            return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.depth_model.apply(freeze_batchnorm)
        if self.config.freeze_batch_norm:
            self.generator.apply(freeze_batchnorm)

        # x = synthetic image, z = real image
        x, z = batch
        if optimizer_idx == 0 and self.global_step >= self.config.warmup_steps:
            # output of encoder when evaluating a real image
            encoder_outs, encoder_mare_outs = self.generator(z)
            decoder_outs_synth = self.depth_model.decoder(encoder_outs)

            img_out = decoder_outs_synth[3]

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
                valid_predicted = self.d_feat_modules[idx](feature_out).type_as(feature_out)
                valid = torch.ones(valid_predicted.size()).type_as(feature_out) * self.config.d_max_conf
                loss = self.adversarial_loss(valid_predicted, valid)
                self.log("g_loss_feature_{}".format(idx), loss)
                g_losses_feat.append(loss)

            valid_predicted_img = self.d_img(img_out).type_as(img_out)
            # use type as to transfer to GPU
            valid_img = torch.ones(valid_predicted_img.size()).type_as(img_out) * self.config.d_max_conf
            # binary cross entropy (patch gan)

            g_loss_img = self.adversarial_loss(valid_predicted_img, valid_img)
            self.log("g_loss_img", g_loss_img)

            g_loss_feat = torch.sum(torch.stack(g_losses_feat))
            g_loss_img = g_loss_img
            g_loss = g_loss_feat \
                     + self.config.residual_loss_factor * residual_loss \
                     + self.config.img_discriminator_factor * g_loss_img
            self.log("g_loss", g_loss)
            self.log("g_skip_loss", g_loss_feat)
            self.log("g_res_loss", residual_loss)
            self.log("g_scale_loss", scale_loss)
            return g_loss
        elif optimizer_idx > 0:
            if optimizer_idx == 1:
                y = self.depth_model(x)[-1].detach()
                y_hat = self.depth_model.decoder(self.generator(z)[0])[-1].detach()
                d = self.d_img
                name = "img"
            else:
                decoder_outs_synth = self.depth_model.encoder(x)[0][::-1]
                y = decoder_outs_synth[optimizer_idx - 2].detach()
                decoder_outs_real = self.generator(z)[0][::-1]
                # evaluate current generator with a real image and take bottleneck output
                y_hat = decoder_outs_real[optimizer_idx - 2].detach()
                d = self.d_feat_modules[optimizer_idx - 2]
                name = str(optimizer_idx - 2)

            valid_predicted = d(y)
            fake_predicted = d(y_hat)

            # how well can it label as real?
            valid = (torch.ones(valid_predicted.size()) * self.config.d_max_conf).type_as(y_hat)

            real_loss = self.adversarial_loss(valid_predicted, valid)

            # how well can it label as fake?
            fake = torch.zeros(fake_predicted.size()).type_as(y_hat)

            fake_loss = self.adversarial_loss(fake_predicted, fake)
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss_{}".format(name), d_loss)
            if optimizer_idx == 1:
                self.accum_dlosses = 0
            self.accum_dlosses += d_loss.detach()
            if optimizer_idx == 4:
                self.log("d_loss", self.accum_dlosses)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, z = batch
        y_hat = self.depth_model.decoder(self.generator(z)[0])
        img_unapdated = self.depth_model(z)[-1]
        img_adapted = y_hat[-1]
        plot_tensors = [self.imagenet_denorm(z)]
        labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
        centers = [None, None, None, 0]
        minmax = []
        plot_tensors.append(img_adapted)
        plot_tensors.append(img_unapdated)
        plot_tensors.append(img_adapted - img_unapdated)

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
        img_unapdated = self.depth_model(z)[-1]
        img_adapted = y_hat[-1]
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
        if self.config.residual_transfer:
            milestones = [10, 20, 30]
        else:
            milestones = [10, 20, 30]
        lr_scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=milestones, gamma=.1)
        lr_scheduler_d_img = torch.optim.lr_scheduler.MultiStepLR(opt_d_img, milestones=milestones, gamma=.1)
        lr_schedulers_d_feat = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=.1) for opt in
                                opt_d_feature]
        return [opt_g, opt_d_img, *opt_d_feature], [lr_scheduler_g, lr_scheduler_d_img, *lr_schedulers_d_feat]

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
