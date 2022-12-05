from matplotlib import pyplot as plt
from torch import optim
import torch
from models.base_model import BaseModel
from typing import *
from utils.loss import BerHu, GradientLoss
from models.adaptive_encoder import AdaptiveEncoder
from utils.image_utils import generate_heatmap_fig
from models.vanillaencoder import VanillaEncoder
from models.decoder import Decoder


class DepthEstimationModel(BaseModel):
    def __init__(self, adaptive_gating=False, **kwargs):
        super().__init__()
        # automatic learning rate finder sets lr to self.lr, else default
        self.save_hyperparameters()  # saves all keywords and their values passed to init function
        self.decoder = Decoder()
        self.berhu = BerHu()
        self.gradloss = GradientLoss()
        self.validation_images = None
        if kwargs.get('resume_from_checkpoint', None):
            self.encoder = AdaptiveEncoder(adaptive_gating)
            ckpt = self.load_from_checkpoint(kwargs['resume_from_checkpoint'], strict=False)
            self.load_state_dict(ckpt.state_dict())
            for param in self.decoder.parameters():
                param.requires_grad_ = False
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
        synth_img, synth_label = batch
        y_hat = self(synth_img)
        # iterate through scales
        depth_loss = 0
        grad_loss = 0
        for idx, predicted in enumerate(y_hat[::-1]):
            depth_loss += self.berhu(predicted, synth_label)
            # apply gradient loss after first epoch
            if self.current_epoch > 0:
                # apply only to high resolution prediction
                grad_loss += self.hparams.grad_loss_factor * self.gradloss(predicted, synth_label)
        loss = depth_loss + grad_loss
        self.log("train_loss", loss)
        return loss

    def shared_val_test_step(self, batch: List[torch.Tensor], batch_idx: int, prefix: str):
        synth_img, synth_label = batch
        # only final depth map is of interest during validation
        y_hat = self(synth_img)[-1]
        metric_dict, _ = self.calculate_metrics(prefix, y_hat, synth_label)
        self.log_dict(metric_dict)
        if batch_idx == 0:
            # do plot on the same images without differing augmentations
            if self.validation_images is None:
                self.plot_minmax = [[None, (0, img.max()), (0, img.max())] for img in synth_label]
                self.validation_images = (synth_img.clone(), synth_label.clone())
            synth_img, synth_label = self.validation_images
            y_hat = self(synth_img)[-1]
            self.plot(prefix, synth_img, y_hat, synth_label)
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
        out = self.decoder(skip_outs)
        return out

    def gen_plots(self, imgs, prefix, labels=None, centers=None, minmax=None):
        if minmax is None:
            minmax = [[] for _ in range(len(imgs))]
        for idx, imgs in enumerate(imgs):
            fig = generate_heatmap_fig(imgs, labels, centers, minmax=minmax[idx])
            self.logger.experiment.add_figure("{}-{}".format(prefix, idx), fig, self.global_step)
            plt.close(fig)
