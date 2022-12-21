from matplotlib import pyplot as plt
from torch import optim
import torch
from models.base_model import BaseModel
from typing import *
from utils.loss import BerHu, GradientLoss, CosineSimilarity
from models.adaptive_encoder import AdaptiveEncoder
from utils.image_utils import generate_heatmap_fig
from models.decoder import Decoder


class DepthEstimationModel(BaseModel):
    def __init__(self, adaptive_gating=True, include_normals=False, **kwargs):
        super().__init__()
        # automatic learning rate finder sets lr to self.lr, else default
        self.save_hyperparameters()  # saves all keywords and their values passed to init function
        self.depth_decoder = Decoder()
        self.normals_decoder = Decoder(3) if include_normals else None
        self.berhu = BerHu()
        self.gradient_loss = GradientLoss()
        self.normals_loss = CosineSimilarity()
        self.regularized_normals_loss = torch.nn.MSELoss()
        self.validation_images = None
        self.include_normals = include_normals
        if kwargs.get('resume_from_checkpoint', None):
            self.encoder = AdaptiveEncoder(adaptive_gating)
            ckpt = self.load_from_checkpoint(kwargs['resume_from_checkpoint'], strict=False)
            self.load_state_dict(ckpt.state_dict())
        else:
            self.encoder = AdaptiveEncoder(adaptive_gating)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
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

    def training_step(self, batch, batch_idx):
        if self.include_normals:
            synth_img, synth_phong, synth_depth, synth_normals = batch
            y_hat_depth, y_hat_normals = self(synth_img)
        else:
            synth_img, synth_phong, synth_depth = batch
            y_hat_depth = self(synth_img)
        depth_loss = 0
        normals_loss = 0
        regularized_normals_loss = 0
        grad_loss = 0

        # iterate through outputs at each level of decoder
        for idx, predicted in enumerate(y_hat_depth[::-1]):
            depth_loss += self.berhu(predicted, synth_depth)
            # apply gradient loss after first epoch
            if self.current_epoch > 0:
                # apply only to high resolution prediction
                if idx == len(y_hat_depth[::-1]) - 1:
                    grad_loss += self.hparams.grad_loss_factor * self.gradient_loss(predicted, synth_depth)
        if self.include_normals:
            for idx, predicted in enumerate(y_hat_normals[::-1]):
                normals_loss += self.normals_loss(predicted, synth_normals)
                norm = torch.linalg.norm(predicted, dim=1)
                regularized_normals_loss += self.regularized_normals_loss(norm, torch.ones_like(norm))

        loss = depth_loss + grad_loss + normals_loss + regularized_normals_loss
        self.log("training_loss", loss)
        return loss

    def shared_val_test_step(self, batch: List[torch.Tensor], batch_idx: int, prefix: str):
        if self.include_normals:
            synth_img, synth_phong, synth_depth, synth_normals = batch
            y_hat_depth, y_hat_normals = self(synth_img)
        else:
            synth_img, synth_phong, synth_depth = batch
            y_hat_depth = self(synth_img)

        metric_dict, _ = self.calculate_metrics(prefix, y_hat_depth[-1], synth_depth)
        self.log_dict(metric_dict)
        if batch_idx == 0:
            # do plot on the same images without differing augmentations
            if self.validation_images is None:
                self.plot_minmax = [[None, (0, img.max().cpu()), (0, img.max().cpu())] for img in synth_depth]
                self.validation_images = (synth_img.clone(),
                                          synth_depth.clone(),
                                          synth_normals.clone() if self.include_normals else None)
            synth_img, synth_depth, synth_normals = self.validation_images
            if self.include_normals:
                y_hat_depth, y_hat_normals = self(synth_img)
            else:
                y_hat_depth = self(synth_img)
            self.plot(prefix, synth_img, y_hat_depth[-1].cpu(), synth_depth)
        return metric_dict

    def test_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "val")

    def plot(self, prefix, synth_img, prediction, label):
        max_num_samples = 7
        self.gen_plots(zip(synth_img[:max_num_samples], prediction[:max_num_samples], label[:max_num_samples]),
                       "{}-synth-prediction".format(prefix), labels=["Synth Image", "Depth Predicted", "Depth GT"],
                       minmax=self.plot_minmax)

    def forward(self, _input):
        skip_outs, _ = self.encoder(_input)
        depth_out = self.depth_decoder(skip_outs)
        if self.include_normals:
            normals_out = self.normals_decoder(skip_outs)
            return depth_out, normals_out
        return depth_out

    def gen_plots(self, imgs, prefix, labels=None, centers=None, minmax=None):
        if minmax is None:
            minmax = [[] for _ in range(len(imgs))]
        for idx, imgs in enumerate(imgs):
            fig = generate_heatmap_fig(imgs, labels, centers, minmax=minmax[idx])
            self.logger.experiment.add_figure("{}-{}".format(prefix, idx), fig, self.global_step)
            plt.close(fig)
