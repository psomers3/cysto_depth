from matplotlib import pyplot as plt
from torch import optim
import torch
from models.base_model import BaseModel
from config.training_config import SyntheticTrainingConfig
from typing import *
from utils.loss import BerHu, GradientLoss, CosineSimilarity, PhongLoss
from models.adaptive_encoder import AdaptiveEncoder
from utils.image_utils import generate_heatmap_fig, generate_normals_fig, generate_img_fig
from models.decoder import Decoder
from argparse import Namespace


class DepthEstimationModel(BaseModel):
    def __init__(self, config: SyntheticTrainingConfig):
        super().__init__()
        # automatic learning rate finder sets lr to self.lr, else default
        self.save_hyperparameters(Namespace(**config))
        self.config = config
        self.depth_decoder = Decoder(output_each_level=True)
        self.normals_decoder = Decoder(3, output_each_level=False) if config.predict_normals else None
        self.berhu = BerHu()
        self.gradient_loss = GradientLoss()
        self.normals_loss: CosineSimilarity = None
        self.phong_loss: PhongLoss = None
        self.validation_images = None
        self.max_num_image_samples = 7
        """ number of images to track and plot during training """
        self.encoder = AdaptiveEncoder(config.adaptive_gating)
        if config.resume_from_checkpoint:
            path_to_ckpt = config.resume_from_checkpoint
            config.resume_from_checkpoint = ""  # set empty or a recursive loading problem occurs
            ckpt = self.load_from_checkpoint(path_to_ckpt,
                                             strict=False,
                                             config=config)
            self.load_state_dict(ckpt.state_dict())

    def forward(self, _input):
        skip_outs, _ = self.encoder(_input)
        depth_out = self.depth_decoder(skip_outs)
        if self.config.predict_normals:
            normals_out = torch.nn.functional.normalize(self.normals_decoder(skip_outs), dim=1)
            return depth_out, torch.where(depth_out[-1] > self.config.min_depth, normals_out,
                                          torch.zeros((1), device=self.device))
        return depth_out

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.lr_scheduler_monitor,
                "frequency": self.hparams.lr_scheduler_patience
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def setup_losses(self):
        """ set up the custom losses here because the device isn't set yet by pytorch lightning when __init__ is run. """
        if self.phong_loss is None:
            self.phong_loss = PhongLoss(image_size=self.config.image_size, config=self.config.phong_config,
                                        device=self.device)
            self.normals_loss = CosineSimilarity(device=self.device)

    def training_step(self, batch, batch_idx):
        if self.config.predict_normals:
            synth_img, synth_phong, synth_depth, synth_normals = batch
            y_hat_depth, y_hat_normals = self(synth_img)
        else:
            synth_img, synth_phong, synth_depth = batch
            y_hat_depth = self(synth_img)

        self.setup_losses()
        depth_loss = 0
        normals_loss = 0
        grad_loss = 0
        phong_loss = 0

        # iterate through outputs at each level of decoder from output to bottleneck
        for idx, predicted in enumerate(y_hat_depth[::-1]):
            depth_loss += self.berhu(predicted, synth_depth)

            # apply gradient loss after first epoch
            if self.current_epoch > 0:
                # apply only to high resolution prediction
                if idx == 0:
                    grad_loss = self.gradient_loss(predicted, synth_depth)
                    self.log("depth_gradient_loss", grad_loss)
            self.log("depth_berhu_loss", depth_loss)
        if self.config.predict_normals:
            normals_loss = self.normals_loss(y_hat_normals, synth_normals)
            self.log("normals_cosine_similarity_loss", normals_loss)
            phong_loss = self.phong_loss((y_hat_depth[-1], y_hat_normals), synth_phong)[0]
            self.log("phong_loss", phong_loss)

        loss = depth_loss * self.config.depth_loss_factor + \
               grad_loss * self.config.depth_grad_loss_factor + \
               normals_loss * self.config.normals_loss_factor + \
               phong_loss * self.config.phong_loss_factor
        self.log("training_loss", loss)
        return loss

    def shared_val_test_step(self, batch: List[torch.Tensor], batch_idx: int, prefix: str):
        self.setup_losses()
        if self.config.predict_normals:
            synth_img, synth_phong, synth_depth, synth_normals = batch
            y_hat_depth, y_hat_normals = self(synth_img)
        else:
            synth_img, synth_depth = batch
            y_hat_depth = self(synth_img)

        metric_dict, _ = self.calculate_metrics(prefix, y_hat_depth[-1], synth_depth)
        self.log_dict(metric_dict)
        if batch_idx == 0:
            # do plot on the same images without differing augmentations
            if self.validation_images is None:
                self.plot_minmax = [[None, (0, img.max().cpu()), (0, img.max().cpu())] for img in synth_depth]
                self.validation_images = (synth_img.clone()[:self.max_num_image_samples].cpu(),
                                          synth_depth.clone()[:self.max_num_image_samples].cpu(),
                                          synth_normals.clone()[:self.max_num_image_samples].cpu() if
                                          self.config.predict_normals else None,
                                          synth_phong.clone()[:self.max_num_image_samples].cpu() if
                                          self.config.predict_normals else None,
                                          )
            self.plot(prefix)
        return metric_dict

    def test_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "val")

    def plot(self, prefix) -> None:
        """
        plot all the images to tensorboard

        :param prefix: a string to prepend to the image tags. Usually "test" or "train"
        """
        with torch.no_grad():
            synth_imgs, synth_depths, synth_normals, synth_phong = self.validation_images
            if self.config.predict_normals:
                y_hat_depth, y_hat_normals = self(synth_imgs.to(self.device))
                y_hat_depth, y_hat_normals = y_hat_depth[-1].to(self.phong_loss.light.device), \
                                             y_hat_normals.to(self.phong_loss.light.device)
                y_phong = self.phong_loss((y_hat_depth, y_hat_normals), synth_phong.to(self.phong_loss.light.device))[
                    1].cpu()
                print(f'y_hat_normals: {y_hat_normals.max()}, y_phong: {y_phong.max()}')
                self.gen_normal_plots(zip(synth_imgs, y_hat_normals.cpu(), synth_normals),
                                      prefix=f'{prefix}-synth-normals',
                                      labels=["Synth Image", "Predicted", "Ground Truth"])
                self.gen_phong_plots(zip(synth_imgs, y_phong, synth_phong),
                                     prefix=f'{prefix}-synth-phong',
                                     labels=["Synth Image", "Predicted", "Ground Truth"])
            else:
                y_hat_depth = self(synth_imgs.to(self.device))[-1]

            self.gen_depth_plots(zip(synth_imgs, y_hat_depth.cpu(), synth_depths),
                                 f"{prefix}-synth-depth",
                                 labels=["Synth Image", "Predicted", "Ground Truth"],
                                 minmax=self.plot_minmax)

    def gen_phong_plots(self, images, prefix, labels):
        for idx, img_set in enumerate(images):
            fig = generate_img_fig(img_set, labels)
            self.logger.experiment.add_figure(f'{prefix}-{idx}', fig, self.global_step)
            plt.close(fig)

    def gen_normal_plots(self, images, prefix, labels):
        for idx, img_set in enumerate(images):
            fig = generate_normals_fig(img_set, labels)
            self.logger.experiment.add_figure(f'{prefix}-{idx}', fig, self.global_step)
            plt.close(fig)

    def gen_depth_plots(self, images, prefix, labels=None, centers=None, minmax=None):
        if minmax is None:
            minmax = [[] for _ in range(len(images))]
        for idx, img in enumerate(images):
            fig = generate_heatmap_fig(img, labels, centers, minmax=minmax[idx])
            self.logger.experiment.add_figure(f"{prefix}-{idx}", fig, self.global_step)
            plt.close(fig)
