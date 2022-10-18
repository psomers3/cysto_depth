from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from models.base_model import BaseModel
from models.vanillaencoder import VanillaEncoder
from models.discriminator_img import ImgDiscriminator
from models.adaptive_encoder import AdaptiveEncoder
from models.discriminator import Discriminator
from models.depth_model import DepthEstimationModel
import socket
from utils.image_utils import generate_heatmap_fig, set_bn_eval, generate_final_imgs


class GAN(BaseModel):
    def __init__(
            self,
            depth_model=None,
            preadapted_model=None,
            image_gan: bool = False,
            lr_d: float = 5e-5,
            lr_g: float = 5e-6,
            b1: float = 0.5,
            b2: float = 0.999,
            res_loss_factor=5,
            scale_loss_factor=0,
            img_discriminator_factor=0,
            res_transfer=True,
            adaptive_gating=False,
            warmup_steps=0,
            d_max_conf=0.9,
            accum_grad_batches=None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters("lr_d", "lr_g", "b1", "b2", "res_loss_factor", "scale_loss_factor",
                                  "img_discriminator_factor", "res_transfer", "warmup_steps", "d_max_conf",
                                  "accum_grad_batches", "adaptive_gating", "image_gan")

        if res_transfer:
            self.generator = AdaptiveEncoder(adaptive_gating)
            if preadapted_model:
                self.generator.load_state_dict(preadapted_model.encoder.state_dict())
            elif depth_model is not None:
                self.generator.load_state_dict(depth_model.encoder.state_dict(), strict=False)
        else:
            self.generator = VanillaEncoder()
            if depth_model is not None:
                self.generator.load_state_dict(depth_model.encoder.state_dict(), strict=False)
        # self.generator.grad_upper_layers(False)
        if depth_model is not None:
            self.depth_model = depth_model
        else:
            self.depth_model = DepthEstimationModel()
        self.depth_model.requires_grad_(False)

        d_in_shapes = [512, 256, 128, 64, 64]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:3]:
            d = Discriminator(in_channels=d_in_shape, single_out=image_gan)
            d_feat_list.append(d)
        self.d_img = ImgDiscriminator(in_shape=1)
        self.d_feat_modules = torch.nn.ModuleList(modules=d_feat_list)
        self.gan = True

    def forward(self, z, full_prediction=False):
        if full_prediction:
            encoder_outs, encoder_mare_outs = self.generator(z)
            return self.depth_model.decoder(encoder_outs)
        else:
            return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.depth_model.apply(set_bn_eval)
        self.generator.apply(set_bn_eval)

        # x = synthetic image, z = real image, l = labels
        x, z, _, _ = batch
        if self.global_step == 0:
            self.logger.experiment.add_graph(self.generator, z)
        # train generator
        if optimizer_idx == 0 and self.global_step >= self.hparams.warmup_steps:
            # output of encoder when evaluating a real image
            encoder_outs, encoder_mare_outs = self.generator(z)
            decoder_outs_synth = self.depth_model.decoder(encoder_outs)

            img_out = decoder_outs_synth[3]

            # compare output levels to make sure they produce roughly the same output
            if self.hparams.res_transfer:
                residual_loss = torch.mean(torch.stack(
                    encoder_mare_outs))  # torch.mean(torch.stack([torch.mean(torch.abs(tensor)) for tensor in encoder_res_outs]))
            else:
                residual_loss = torch.tensor([0]).type_as(z)
            scale_loss = 0  # torch.sum(torch.stack([F.l1_loss(out,scaled_target) for out,scaled_target in zip(l_x,decoder_outs_synth[0:4])]))
            # actual output of the discriminator
            g_losses_feat = []
            feat_outs = encoder_outs[::-1][:len(self.d_feat_modules)]
            for idx, feature_out in enumerate(feat_outs):
                valid_predicted = self.d_feat_modules[idx](feature_out).type_as(feature_out)
                valid = torch.ones(valid_predicted.size()).type_as(feature_out) * self.hparams.d_max_conf
                loss = self.adversarial_loss(valid_predicted, valid)
                self.log("g_loss_feature_{}".format(idx), loss)
                g_losses_feat.append(loss)

            valid_predicted_img = self.d_img(img_out).type_as(img_out)
            # use type as to transfer to GPU
            valid_img = torch.ones(valid_predicted_img.size()).type_as(img_out) * self.hparams.d_max_conf
            # binary cross entropy (patch gan)

            g_loss_img = self.adversarial_loss(valid_predicted_img, valid_img)
            self.log("g_loss_img", g_loss_img)

            g_loss_feat = torch.sum(torch.stack(g_losses_feat))
            g_loss_img = g_loss_img
            g_loss = g_loss_feat + self.hparams.res_loss_factor * residual_loss + self.hparams.img_discriminator_factor * g_loss_img
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
            valid = (torch.ones(valid_predicted.size()) * self.hparams.d_max_conf).type_as(y_hat)

            real_loss = self.adversarial_loss(valid_predicted, valid)

            # how well can it label as fake?
            fake = torch.zeros(fake_predicted.size()).type_as(y_hat)

            fake_loss = self.adversarial_loss(fake_predicted, fake)
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss_{}".format(name), d_loss)
            if optimizer_idx == 1:
                self.accum_dlosses = 0
            self.accum_dlosses += d_loss.item()
            if optimizer_idx == 4:
                self.log("d_loss", self.accum_dlosses)
            return d_loss

    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # if self.global_step == 600 or self.global_step == 1200:
    #     #     for g in optimizer.param_groups:
    #     #         g['lr'] = g['lr']/10

    def validation_step(self, batch, batch_idx):
        x, z, l_x, l_z = batch
        y_hat = self.depth_model.decoder(self.generator(z)[0])
        img_unapdated = self.depth_model(z)[-1]
        img_adapted = y_hat[-1]
        plot_tensors = [z]  # img_adapted, img_unapdated, diff]
        labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
        centers = [None, None, None, 0]

        minmax = []
        if len(l_z) > 0:
            metrics_adapted, img_adapted = self.calculate_metrics("gan", img_adapted, l_z)
            metrics_unadapted, img_unapdated = self.calculate_metrics("gan", img_unapdated, l_z)
            # additionally log rmse as monitorable metric for early stopping etc.
            self.log("val_rmse_monitor", metrics_adapted['gan_rmse'])
            self.log("val_silog_monitor", metrics_adapted['gan_silog'])
            for k in metrics_adapted.keys():
                adapted_metric = metrics_adapted[k]
                unadapted_metric = metrics_unadapted[k]
                logdict = {"adapted": adapted_metric.item(), "unadapted": unadapted_metric.item()}
                self.log(str(k), logdict)
        plot_tensors.append(img_adapted)
        plot_tensors.append(img_unapdated)
        plot_tensors.append(img_adapted - img_unapdated)
        if len(l_z) > 0:
            plot_tensors.append(l_z)
            labels.append("Ground Truth")
            centers.append(None)
            # minmax.append(minmaxtuple)

        self.add_histograms(step=self.global_step)
        for idx, imgs in enumerate(zip(*plot_tensors)):
            fig = generate_heatmap_fig(imgs, labels=labels, centers=centers, minmax=minmax,
                                       align_scales=True)
            self.logger.experiment.add_figure("GAN Prediction Result-{}-{}".format(batch_idx, idx), fig,
                                              self.global_step)
            plt.close(fig)

    def test_step(self, batch, batch_idx):
        x, z, l_x, l_z = batch
        y_hat = self.depth_model.decoder(self.generator(z)[0])
        img_unapdated = self.depth_model(z)[-1]
        img_adapted = y_hat[-1]
        plot_tensors = [z]  # img_adapted, img_unapdated, diff]
        # no labels for test step for the thesis
        labels = ["", "", "", ""]
        centers = [None, None, None, 0]
        minmax = []

        if len(l_z) > 0:
            metrics_adapted, img_adapted = self.calculate_metrics("gan", img_adapted, l_z)
            metrics_unadapted, img_unapdated = self.calculate_metrics("gan", img_unapdated, l_z)
            for k in metrics_adapted.keys():
                adapted_metric = metrics_adapted[k]
                unadapted_metric = metrics_unadapted[k]
                logdict = {"adapted": adapted_metric.item(), "unadapted": unadapted_metric.item()}
                self.log(str(k), logdict)

        plot_tensors.append(img_adapted)
        plot_tensors.append(img_unapdated)
        plot_tensors.append(img_adapted - img_unapdated)

        if len(l_z) > 0:
            plot_tensors.append(l_z)
            labels.append("Ground Truth")
            centers.append(None)
            # minmax.append(minmaxtuple)

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
        lr_d = self.hparams.lr_d
        lr_g = self.hparams.lr_g
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=lr_g,
                                 betas=(b1, b2))
        opt_d_feature = [torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                          lr=lr_d,
                                          betas=(b1, b2)) for discriminator in self.d_feat_modules]
        opt_d_img = torch.optim.Adam(filter(lambda p: p.requires_grad, self.d_img.parameters()), lr=lr_d,
                                     betas=(b1, b2))
        if self.hparams.res_transfer:
            milestones = [10, 20, 30]
        else:
            milestones = [10, 20, 30]
        lr_scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=milestones, gamma=.1)
        lr_scheduler_d_img = torch.optim.lr_scheduler.MultiStepLR(opt_d_img, milestones=milestones, gamma=.1)
        lr_schedulers_d_feat = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=.1) for opt in
                                opt_d_feature]
        return [opt_g, opt_d_img, *opt_d_feature], [lr_scheduler_g, lr_scheduler_d_img, *lr_schedulers_d_feat]

    def on_epoch_end(self):
        # if self.lr_schedulers() is not None:
        # self.log("g_lr",self.lr_schedulers()[0].get_last_lr()[0])
        # self.log("d_lr",self.lr_schedulers()[1].get_last_lr()[0])
        self.log(socket.gethostname(), True)

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
