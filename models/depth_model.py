from matplotlib import pyplot as plt
from torch import optim
import torch
from models.base_model import BaseModel

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
        if kwargs.get('resume_from_checkpoint', None):
            self.encoder = AdaptiveEncoder(adaptive_gating)
            self.load_state_dict(kwargs['resume_from_checkpoint'], strict=False)
            for param in self.decoder.parameters():
                param.requires_grad_ = False
        else:
            self.encoder = VanillaEncoder()

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

    def shared_val_test_step(self, batch, batch_idx, prefix):
        synth_img, synth_label = batch
        # only final depth map is of interest during validation
        y_hat = self(synth_img)[-1]
        label = synth_label
        metric_dict, _ = self.calculate_metrics(prefix, y_hat, label)
        self.log_dict(metric_dict)
        # if batch_idx == 0:
        #     self.plot(prefix, synth_img, y_hat, label, real_img, real_hat)
        return metric_dict

    def test_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "val")

    def plot(self, prefix, synth_img, prediction, label, real_img, real_hat):
        kwargs1 = {}
        kwargs2 = {}
        max_num_samples = 7

        kwargs1["minmax"] = [None, (0, 20), (0, 20)]
        kwargs2["minmax"] = [None, (0, 20)]
        self.gen_plots(zip(synth_img[:max_num_samples], prediction[:max_num_samples], label[:max_num_samples]),
                       "{}-synth-prediction".format(prefix), labels=["Synth Image", "Depth Predicted", "Depth GT"],
                       **kwargs1)
        self.gen_plots(zip(real_img[:max_num_samples], real_hat[:max_num_samples]), "{}-real-prediction".format(prefix),
                       labels=["Real Image", "Real Predicted"], **kwargs2)

    def forward(self, _input):
        skip_outs, _ = self.encoder(_input)
        out = self.decoder(skip_outs)
        return out

    def gen_plots(self, imgs, prefix, labels=None, centers=None, minmax=None):
        if minmax is None:
            minmax = []
        for idx, imgs in enumerate(imgs):
            fig = generate_heatmap_fig(imgs, labels, centers, minmax=minmax)
            self.logger.experiment.add_figure("{}-{}".format(prefix, idx), fig, self.global_step)
            plt.close(fig)
