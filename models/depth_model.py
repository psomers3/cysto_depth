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
from data.data_transforms import ImageNetNormalization
from argparse import Namespace
from utils.rendering import depth_to_normals, get_pixel_locations, PhongRender

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger to attach... ", end='', flush=True)
# debugpy.wait_for_client()
# print("done!")

imagenet_denorm = ImageNetNormalization(inverse=True)


class DepthEstimationModel(BaseModel):
    def __init__(self, config: SyntheticTrainingConfig):
        super().__init__()
        # automatic learning rate finder sets lr to self.lr, else default
        ckpt = None
        if config.resume_from_checkpoint:
            ckpt = torch.load(config.resume_from_checkpoint, map_location=self.device)
            hparams = ckpt['hyper_parameters']
            needed_params = ['encoder', 'merged_decoder', 'normals_extra_layers', 'phong_config', 'image_size',
                             'decoder_calculate_norms']
            [setattr(config, key, val) for key, val in hparams.items() if key in needed_params]

        self.save_hyperparameters(Namespace(**config))
        self.config = config
        self.encoder = AdaptiveEncoder(config.encoder)

        num_output_layers = 4 if config.merged_decoder and config.predict_normals else 1
        self.decoder = Decoder(feature_levels=self.encoder.feature_levels[::-1],
                               num_output_channels=num_output_layers,
                               output_each_level=True,
                               extra_normals_layers=config.normals_extra_layers,
                               phong_renderer=PhongRender(config=config.phong_config,
                                                          image_size=config.image_size,
                                                          device=self.device,)
                               if config.decoder_calculate_norms else None,
                               use_skip_connections=config.use_skip_connections)
        if config.predict_normals and not config.merged_decoder:
            self.normals_decoder = Decoder(feature_levels=self.encoder.feature_levels,
                                           num_output_channels=3,
                                           output_each_level=False,
                                           use_skip_connections=config.use_skip_connections)
        else:
            self.normals_decoder = None
        self.pixel_locations = None
        self.berhu = BerHu()
        self.gradient_loss = GradientLoss()
        self.normals_loss: CosineSimilarity = None
        self.calculated_normals_loss: CosineSimilarity = None
        self.phong_loss: PhongLoss = None
        self.validation_images = None
        self.train_images = None
        self.test_images = None
        self.train_denorm_color_images = None
        self.val_denorm_color_images = None
        self.test_denorm_color_images =None
        self.plot_minmax_train = None
        self.plot_minmax_test = None
        self.plot_minmax_val = None
        self.val_plottable_norms = None
        self.train_plottable_norms = None
        self.test_plottable_norms = None
        self._val_epoch_count = 0
        self.max_num_image_samples = 7
        """ number of images to track and plot during training """
        if ckpt is not None:
            self._resume_from_checkpoint(ckpt)

    def _resume_from_checkpoint(self, ckpt: dict):
        self.load_state_dict(ckpt['state_dict'], strict=False)

    def forward(self, _input):
        skip_outs, _ = self.encoder(_input)
        depth_out = self.decoder(skip_outs)
        if self.config.predict_normals:
            if self.config.merged_decoder:
                depth_out, normals_out = [layer[:, 0, ...].unsqueeze(1) for layer in depth_out], depth_out[-1][:, 1:,
                                                                                                 ...]
            else:
                normals_out = self.normals_decoder(skip_outs)
            return depth_out, torch.where(depth_out[-1] > self.config.min_depth, normals_out,
                                          torch.zeros((1), device=self.device))
        return depth_out

    def configure_optimizers(self):
        optimizer_cls = optim.Adam if self.config.optimizer.lower() == 'adam' else optim.RAdam
        optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
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
        """
        set up the custom losses here because the device isn't set yet by pytorch lightning when __init__ is run.
        """
        if self.phong_loss is None:
            self.phong_loss = PhongLoss(image_size=self.config.image_size, config=self.config.phong_config,
                                        device=self.device)
            self.normals_loss = CosineSimilarity(device=self.device)
            self.calculated_normals_loss = CosineSimilarity(device=self.device, ignore_direction=True)
            self.pixel_locations = get_pixel_locations(self.config.image_size,
                                                       self.config.image_size,
                                                       device=self.device)

    def training_step(self, batch, batch_idx):
        real_images_included = False
        if isinstance(batch, dict):
            real_images_included = True
            # run real images through network to update any batch norm statistics
            real_images = batch['real'][0]
            batch = batch['synth']
            batch[0] = torch.cat([batch[0], real_images], dim=0)

        if self.config.predict_normals:
            synth_img, synth_phong, synth_depth, synth_normals = batch
            y_hat_depth, y_hat_normals = self(synth_img)
            if real_images_included:
                y_hat_depth, y_hat_normals = [d[:self.config.batch_size] for d in y_hat_depth], \
                                             y_hat_normals[:self.config.batch_size]
        else:
            synth_img, synth_depth = batch
            y_hat_depth = self(synth_img)
            if real_images_included:
                y_hat_depth = [d[:self.config.batch_size] for d in y_hat_depth]

        self.setup_losses()
        depth_loss = 0
        normals_loss = 0
        normals_regularization_loss = 0
        grad_loss = 0
        phong_loss = 0

        # iterate through outputs at each level of decoder from output to bottleneck
        for idx, predicted in enumerate(y_hat_depth[::-1]):
            lambda_factor = self.config.depth_loss_lambda_factor ** idx
            depth_loss += self.berhu(predicted, synth_depth) * lambda_factor

            if self.config.depth_gradient_loss_epochs[0] \
                    <= self.current_epoch \
                    <= self.config.depth_gradient_loss_epochs[1]:
                # apply only to high resolution prediction
                if idx == 0:
                    grad_loss = self.gradient_loss(predicted, synth_depth)
                    self.log("depth_gradient_loss", grad_loss, sync_dist=self.config.sync_logging)
            self.log("depth_berhu_loss", depth_loss)
        if self.config.predict_normals:
            if self.config.normals_loss_epochs[0] <= self.current_epoch <= self.config.normals_loss_epochs[1]:
                normals_loss = self.normals_loss(y_hat_normals, synth_normals)
                self.log("normals_cosine_similarity_loss", normals_loss, sync_dist=self.config.sync_logging)

            if self.config.normals_depth_regularization_loss_epochs[0] \
                    <= self.current_epoch \
                    <= self.config.normals_depth_regularization_loss_epochs[1]:
                calculated_normals = depth_to_normals(y_hat_depth[-1],
                                                      self.phong_loss.phong_renderer.camera_intrinsics[None],
                                                      self.pixel_locations)
                calculated_synthetic_normals = depth_to_normals(synth_depth,
                                                                self.phong_loss.phong_renderer.camera_intrinsics[None],
                                                                self.pixel_locations)
                normals_regularization_loss = self.calculated_normals_loss(calculated_normals,
                                                                           calculated_synthetic_normals)
                self.log('depth_to_normals_loss', normals_regularization_loss, sync_dist=self.config.sync_logging)
            if self.config.phong_loss_epochs[0] \
                    <= self.current_epoch \
                    <= self.config.phong_loss_epochs[1]:
                phong_loss = self.phong_loss((y_hat_depth[-1], y_hat_normals), synth_phong)[0]
                self.log("phong_loss", phong_loss, sync_dist=self.config.sync_logging)

        loss = depth_loss * self.config.depth_loss_factor + \
               grad_loss * self.config.depth_grad_loss_factor + \
               normals_loss * self.config.normals_loss_factor + \
               phong_loss * self.config.phong_loss_factor + \
               normals_regularization_loss * self.config.calculated_normals_loss_factor
        self.log("training_loss", loss, sync_dist=self.config.sync_logging)

        if batch_idx % self.config.train_plot_interval == 0:
            if self.train_images is None:
                self.plot_minmax_train, self.train_images = self.prepare_images(batch, self.max_num_image_samples,
                                                                               self.config.predict_normals)
                self.train_denorm_color_images = torch.clamp(imagenet_denorm(self.train_images[0]), 0, 1)
                self.train_plottable_norms = (torch.nn.functional.normalize(self.train_images[2], dim=1) + 1) / 2 \
                    if self.config.predict_normals else None
            self.plot(prefix="train")
        return loss

    @staticmethod
    def prepare_images(batch, max_num_image_samples: int = 7, predict_normals: bool = False) \
            -> Tuple[List[Any], Tuple[torch.Tensor, ...]]:
        """
        Helper function to save a sample of images for plotting during training

        :param batch:
        :param max_num_image_samples:
        :param predict_normals:
        :return:
        """
        if predict_normals:
            synth_img, synth_phong, synth_depth, synth_normals = batch
        else:
            synth_img, synth_depth = batch
        plot_minmax = [[None, (0, img.max().detach().cpu()), (0, img.max().detach().cpu())] for img in synth_depth]
        images = (synth_img.detach()[:max_num_image_samples].cpu(),
                  synth_depth.detach()[:max_num_image_samples].cpu(),
                  synth_normals.detach()[:max_num_image_samples].cpu() if
                  predict_normals else None,
                  synth_phong.detach()[:max_num_image_samples].cpu() if
                  predict_normals else None)
        return plot_minmax, images

    def shared_val_test_step(self, batch: List[torch.Tensor], batch_idx: int, prefix: str):
        self.setup_losses()
        if isinstance(batch, dict):
            batch = batch['synth']
        if self.config.predict_normals:
            synth_img, _, synth_depth, _ = batch
            y_hat_depth, _ = self(synth_img)
        else:
            synth_img, synth_depth = batch
            y_hat_depth = self(synth_img)

        metric_dict, _ = self.calculate_metrics(prefix, y_hat_depth[-1], synth_depth)
        self.log_dict(metric_dict)
        if self.validation_images is None:
            if prefix == 'val':
                self.plot_minmax_val, self.validation_images = self.prepare_images(batch, self.max_num_image_samples,
                                                                                   self.config.predict_normals)
                self.val_denorm_color_images = torch.clamp(imagenet_denorm(self.validation_images[0]), 0, 1)
                self.val_plottable_norms = (torch.nn.functional.normalize(self.validation_images[2], dim=1) + 1) / 2 \
                    if self.config.predict_normals else None
            else:
                self.plot_minmax_test, self.test_images = self.prepare_images(batch, self.max_num_image_samples,
                                                                                   self.config.predict_normals)
                self.test_denorm_color_images = torch.clamp(imagenet_denorm(self.test_images[0]), 0, 1)
                self.test_plottable_norms = (torch.nn.functional.normalize(self.test_images[2], dim=1) + 1) / 2 \
                    if self.config.predict_normals else None
        return metric_dict

    def on_validation_epoch_end(self) -> None:
        if self._val_epoch_count % self.config.val_plot_interval == 0:
            self.plot('val')
        self._val_epoch_count += 1

    def test_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "test")

    def on_test_epoch_end(self) -> None:
        self.plot('test')

    def validation_step(self, batch, batch_idx):
        return self.shared_val_test_step(batch, batch_idx, "val")

    def plot(self, prefix) -> None:
        """
        plot all the images to tensorboard

        :param prefix: a string to prepend to the image tags. Usually "test" or "train"
        """
        if (self.validation_images is None) and (prefix == 'val'):
            return
        with torch.no_grad():
            if prefix == 'val':
                synth_imgs, synth_depths, synth_normals, synth_phong = self.validation_images
                minmax = self.plot_minmax_val.copy()
                denormed_synth_imgs = self.val_denorm_color_images
                plottable_norms = self.val_plottable_norms
            elif prefix == 'train':
                synth_imgs, synth_depths, synth_normals, synth_phong = self.train_images
                minmax = self.plot_minmax_train.copy()
                denormed_synth_imgs = self.train_denorm_color_images
                plottable_norms = self.train_plottable_norms
            else:
                synth_imgs, synth_depths, synth_normals, synth_phong = self.test_images
                minmax = self.plot_minmax_test.copy()
                denormed_synth_imgs = self.test_denorm_color_images
                plottable_norms = self.test_plottable_norms
            if self.config.predict_normals:
                y_hat_depth, y_hat_normals = self(synth_imgs.to(self.device))
                y_hat_depth, y_hat_normals = y_hat_depth[-1].detach().to(self.phong_loss.phong_renderer.device), \
                                             y_hat_normals.detach().to(self.phong_loss.phong_renderer.device)
                y_hat_normals = torch.nn.functional.normalize(y_hat_normals, dim=1)
                y_phong = self.phong_loss((y_hat_depth, y_hat_normals),
                                          synth_phong.to(self.phong_loss.phong_renderer.device))[1].cpu()
                y_hat_normals = (y_hat_normals + 1) / 2
                self.gen_normal_plots(zip(denormed_synth_imgs, y_hat_normals.detach().cpu(), plottable_norms),
                                      prefix=f'{prefix}-synth-normals',
                                      labels=["Synth Image", "Predicted", "Ground Truth"])
                self.gen_phong_plots(zip(denormed_synth_imgs, y_phong, synth_phong),
                                     prefix=f'{prefix}-synth-phong',
                                     labels=["Synth Image", "Predicted", "Ground Truth"])
            else:
                y_hat_depth = self(synth_imgs.to(self.device))[-1]

            y_hat_depth = y_hat_depth.detach().cpu()[:self.max_num_image_samples]
            synth_depths = synth_depths[:self.max_num_image_samples]
            differences = synth_depths - y_hat_depth

            # TODO: tie this to a flag of synchronized gradient plots
            minmax = [[None] for _ in range(self.max_num_image_samples)]
            maximums = [max(x.max().cpu(), y.max().cpu()) for x, y in zip(y_hat_depth, synth_depths)]
            [minmax[i].extend([(0, m), (0, m)]) for i, m in enumerate(maximums)]
            [minmax[i].append((differences[i].min(), differences[i].max())) for i in range(self.max_num_image_samples)]
            self.gen_depth_plots(zip(denormed_synth_imgs, y_hat_depth, synth_depths, differences),
                                 f"{prefix}-synth-depth",
                                 labels=["Synth Image", "Predicted", "Ground Truth", "Error"],
                                 minmax=minmax,
                                 centers=[None, None, None, 0])

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
