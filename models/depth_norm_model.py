import pytorch_lightning as pl
from models.depth_to_image import DepthNorm2Image
from config.training_config import DepthNorm2ImageConfig
import matplotlib.pyplot as plt
from utils.image_utils import generate_img_fig
from data.data_transforms import ImageNetNormalization
from argparse import Namespace
from typing import *
from torch import Tensor, optim
import torch

imagenet_denorm = ImageNetNormalization(inverse=True)


class DepthNormModel(pl.LightningModule):
    def __init__(self, config: DepthNorm2ImageConfig):
        super(DepthNormModel, self).__init__()
        self.save_hyperparameters(Namespace(**config))
        self.config = config
        self.model = DepthNorm2Image(config.encoder, depth_scale=config.depth_scale, add_noise=config.add_noise)
        self.loss = torch.nn.L1Loss() if '1' in config.L_loss else torch.nn.MSELoss()
        self.max_num_image_samples = 8
        if config.resume_from_checkpoint:
            path_to_ckpt = config.resume_from_checkpoint
            config.resume_from_checkpoint = ""  # set empty or a recursive loading problem occurs
            ckpt = self.load_from_checkpoint(path_to_ckpt,
                                             strict=False,
                                             config=config)
            self.load_state_dict(ckpt.state_dict())

        self.val_denorm_color_images = None
        self.validation_data = None
        self._val_epoch_count = 0

    def forward(self, depth: Tensor, normals: Tensor, source_id: int, **kwargs: Any, ) -> Tensor:
        """

        :param depth: [N, 1, h, w]
        :param normals: [N, 3, h, w]
        :param source_id: id for domain the data came from
        :param kwargs: ain't matter
        :return: color image/s [N, 3, h ,w]
        """
        return self.model(depth, normals, source_id=source_id)

    def __call__(self, *args, **kwargs) -> Tensor:
        return super(DepthNormModel, self).__call__(*args, **kwargs)

    def configure_optimizers(self):
        opts = {'adam': optim.Adam, 'radam': optim.RAdam, 'rmsprop': optim.RMSprop}
        opt = opts[self.config.optimizer.lower()]
        optimizer = opt(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer  # , [scheduler]

    def training_step(self, batch, batch_idx):
        synth_img, synth_depth, synth_normals = batch
        out_images = self(synth_depth, synth_normals, source_id=0)
        loss = self.loss(out_images, imagenet_denorm(synth_img))
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.validation_data is None:
            synth_img, synth_depth, synth_normals = batch
            self.val_denorm_color_images = imagenet_denorm(synth_img[:self.max_num_image_samples]).detach().cpu()
            self.validation_data = synth_depth[:self.max_num_image_samples].detach(), \
                                   synth_normals[:self.max_num_image_samples].detach()

    def on_validation_epoch_end(self) -> None:
        if self._val_epoch_count % self.config.val_plot_interval == 0:
            if self.validation_data is not None:
                self.plot()
        self._val_epoch_count += 1
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "test")

    def plot(self):
        with torch.no_grad():
            generated_img = self(*self.validation_data, source_id=0).detach().cpu()
        labels = ["Source Image", "Generated"]

        for idx, img_set in enumerate(zip(self.val_denorm_color_images, generated_img)):
            fig = generate_img_fig(img_set, labels)
            self.logger.experiment.add_figure(f'generated-image-{idx}', fig, self.global_step)
            plt.close(fig)
