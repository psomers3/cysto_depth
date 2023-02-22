from matplotlib import pyplot as plt
import torch
from torch import Tensor
from models.base_model import BaseModel
from models.adaptive_encoder import AdaptiveEncoder
from models.discriminator import Discriminator
from models.depth_model import DepthEstimationModel
from models.depth_norm_model import DepthNormModel
from data.data_transforms import ImageNetNormalization
from utils.image_utils import generate_heatmap_fig, freeze_batchnorm, generate_img_fig
from config.training_config import SyntheticTrainingConfig, GANTrainingConfig, DiscriminatorConfig, \
    DepthNorm2ImageConfig
from argparse import Namespace
from utils.rendering import PhongRender, depth_to_normals
from utils.loss import GANDiscriminatorLoss, GANGeneratorLoss
from typing import *

opts = {'adam': torch.optim.Adam, 'radam': torch.optim.RAdam, 'rmsprop': torch.optim.RMSprop}


class DiscriminatorCriticInputs(TypedDict):
    color: Tensor
    encoder_outs: List[Tensor]
    depth: Tensor
    normals: Tensor
    phong: Tensor
    calculated_phong: Tensor


class HailMary(BaseModel):
    def __init__(self,
                 synth_config: SyntheticTrainingConfig,
                 gan_config: GANTrainingConfig,
                 depth_norm_config: DepthNorm2ImageConfig):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(Namespace(**gan_config))
        self.depth_model = DepthEstimationModel(synth_config)
        self.config = gan_config
        self.generator = AdaptiveEncoder(gan_config.encoder)
        self.generator.load_state_dict(self.depth_model.encoder.state_dict(), strict=False)
        self.texture_generator = DepthNormModel(depth_norm_config)
        self.depth_model.requires_grad = False
        self.imagenet_denorm = ImageNetNormalization(inverse=True)
        self.phong_renderer: PhongRender = None
        self.d_losses_log = {}
        self.g_losses_log = {'g_loss': 0.0}
        self.discriminators: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.critics: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.generated_source_id: int = len(self.texture_generator.sources)
        self.critic_opt_idx = 0
        self.discriminators_opt_idx = 0
        self._unwrapped_optimizers = []
        self.setup_generator_optimizer()
        if self.config.use_critic:
            self.setup_critics()
        if self.config.use_discriminator:
            self.setup_discriminators()
        self.setup_texture_generator()
        self.texture_generator_opt_idx = len(self._unwrapped_optimizers)
        self.texture_critic_opt_idx = self.texture_generator_opt_idx + 1 if \
            self.texture_generator.config.use_critic else self.texture_generator_opt_idx

        self._unwrapped_optimizers.extend(self.texture_generator.configure_optimizers())
        self.depth_model.requires_grad = True
        self.generator.requires_grad = True
        self.texture_generator.requires_grad =True
        self.texture_discriminator_opt_idx = self.texture_critic_opt_idx + 1
        self.validation_epoch = 0
        self.generator_global_step = -1
        self.critic_global_step = 0
        self.total_train_step_count = -1
        self.batches_accumulated = 0
        self._generator_training = False
        self.unadapted_images_for_plotting = None
        self.validation_data = None
        self.discriminator_loss = GANDiscriminatorLoss[gan_config.discriminator_loss]
        self.critic_loss = GANDiscriminatorLoss[gan_config.critic_loss]
        self.generator_critic_loss = GANGeneratorLoss[gan_config.critic_loss]
        self.generator_discriminator_loss = GANGeneratorLoss[gan_config.discriminator_loss]

        if gan_config.resume_from_checkpoint:
            path_to_ckpt = gan_config.resume_from_checkpoint
            gan_config.resume_from_checkpoint = ""  # set empty or a recursive loading problem occurs
            ckpt = self.load_from_checkpoint(path_to_ckpt,
                                             strict=False,
                                             synth_config=synth_config,
                                             gan_config=gan_config)
            self.load_state_dict(ckpt.state_dict())

    def setup_generator_optimizer(self):
        opt = opts[self.config.generator_optimizer.lower()]
        self._unwrapped_optimizers.append(opt(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                              lr=self.config.generator_lr))

    def setup_discriminators(self):
        d_in_shapes = self.generator.feature_levels[::-1]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:-1]:
            d_config: DiscriminatorConfig = self.config.feature_level_discriminator.copy()
            d_config.in_channels = d_in_shape
            d = Discriminator(d_config)
            d_feat_list.append(d)
        self.d_losses_log['d_discriminators_loss'] = 0.0
        self.discriminators['features'] = torch.nn.ModuleList(modules=d_feat_list)
        self.d_losses_log.update({f'd_loss_discriminator_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        self.g_losses_log.update({f'g_loss_discriminator_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        if self.config.predict_normals:
            self.discriminators['phong'] = Discriminator(self.config.phong_discriminator)
            self.discriminators['depth_phong'] = Discriminator(self.config.phong_discriminator)
            self.d_losses_log.update({'d_loss_discriminator_phong': 0.0, 'd_loss_discriminator_depth_phong': 0.0})
            self.g_losses_log.update({'g_loss_discriminator_phong': 0.0, 'g_loss_discriminator_depth_phong': 0.0})
        self.discriminators['depth_image'] = Discriminator(self.config.depth_discriminator)
        self.d_losses_log.update({'d_loss_discriminator_depth_img': 0.0})
        self.g_losses_log.update({'g_loss_discriminator_depth_img': 0.0})
        opt = opts[self.config.discriminator_optimizer.lower()]
        self._unwrapped_optimizers.append(opt(filter(lambda p: p.requires_grad, self.discriminators.parameters()),
                                              lr=self.config.discriminator_lr))
        self.discriminators_opt_idx = self.critic_opt_idx + 1

    def setup_texture_generator(self):
        if self.texture_generator.config.use_discriminator:
            self.texture_generator.discriminators[str(self.generated_source_id)] = \
                Discriminator(self.texture_generator.config.discriminator_config)
            self.texture_generator.g_losses_log[f'g_discriminator_loss-{self.generated_source_id}'] = 0.0
            self.texture_generator.d_losses_log[f'd_discriminator_loss-{self.generated_source_id}'] = 0.0

        if self.texture_generator.config.use_critic:
            self.texture_generator.critics[str(self.generated_source_id)] = \
                Discriminator(self.texture_generator.config.critic_config)
            self.texture_generator.g_losses_log[f'g_critic_loss-{self.generated_source_id}'] = 0.0
            self.texture_generator.d_losses_log[f'd_critic_loss-{self.generated_source_id}'] = 0.0

    def setup_critics(self):
        d_in_shapes = self.generator.feature_levels[::-1]
        d_feat_list = []
        for d_in_shape in d_in_shapes[:-1]:
            d_config: DiscriminatorConfig = self.config.feature_level_critic.copy()
            d_config.in_channels = d_in_shape
            d = Discriminator(d_config)
            d_feat_list.append(d)
        self.d_losses_log['d_critics_loss'] = 0.0
        self.critics['features'] = torch.nn.ModuleList(modules=d_feat_list)
        self.d_losses_log.update({f'd_loss_critic_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        self.g_losses_log.update({f'g_loss_critic_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        if self.config.predict_normals:
            self.critics['phong'] = Discriminator(self.config.phong_critic)
            self.critics['depth_phong'] = Discriminator(self.config.phong_critic)
            self.d_losses_log.update({'d_loss_critic_phong': 0.0, 'd_loss_critic_depth_phong': 0.0})
            self.g_losses_log.update({'g_loss_critic_phong': 0.0, 'g_loss_critic_depth_phong': 0.0})
        self.critics['depth_image'] = Discriminator(self.config.depth_critic)
        self.d_losses_log.update({'d_loss_critic_depth_img': 0.0})
        self.g_losses_log.update({'g_loss_critic_depth_img': 0.0})
        opt = opts[self.config.critic_optimizer.lower()]
        self._unwrapped_optimizers.append(opt(filter(lambda p: p.requires_grad, self.critics.parameters()),
                                              lr=self.config.critic_lr))
        self.critic_opt_idx += 1

    @staticmethod
    def reset_log_dict(log_dict: dict):
        log_dict.update({k: 0.0 for k in log_dict.keys()})

    def forward(self, x, generator: bool = True) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
        return self.get_predictions(x, generator=generator)

    def __call__(self, *args, **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
        return super(HailMary, self).__call__(*args, **kwargs)

    def on_validation_epoch_start(self) -> None:
        if self.config.predict_normals:
            self.phong_renderer = PhongRender(config=self.config.phong_config,
                                              image_size=self.config.image_size,
                                              device=self.device)

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

        return encoder_outs, encoder_mare_outs, decoder_outs, normals

    def training_step(self, batch, batch_idx):
        # start by freezing all batchnorm layers throughout the networks that shouldn't update statistics
        self.train()
        self.depth_model.apply(freeze_batchnorm)
        if self.config.freeze_batch_norm:
            self.generator.apply(freeze_batchnorm)

        self.total_train_step_count += 1
        self.batches_accumulated += 1
        if self._generator_training:
            # print('generator')
            self.generator_train_step(batch, batch_idx)
        self.discriminator_critic_train_step(batch, batch_idx)
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.batches_accumulated = 0

    def generator_train_step(self, batch: Dict[int, List[Tensor]], batch_idx) -> None:
        z = batch[self.generated_source_id][0]  # real images
        # set discriminators to eval so that any normalization statistics don't get updated
        self.discriminators.eval()
        self.critics.eval()
        if self.config.encoder.residual_learning:
            self.generator.set_residuals_train()

        # output of encoder when evaluating a real image
        encoder_outs_generated, encoder_mare_outs_generated, decoder_outs_generated, normals_generated = self(z, generator=True)
        depth_out = decoder_outs_generated[-1]

        g_loss: Tensor = 0.0

        original_phong_rendering = self.phong_renderer((depth_out, normals_generated))
        calculated_norms = depth_to_normals(depth_out, self.phong_renderer.camera_intrinsics[None],
                                            self.phong_renderer.resized_pixel_locations)
        depth_phong = self.phong_renderer((depth_out, calculated_norms))
        if self.config.use_discriminator:
            feat_outs = encoder_outs_generated[::-1][:len(self.discriminators['features'])]
            for idx, feature_out in enumerate(feat_outs):
                g_loss += self._apply_generator_discriminator_loss(feature_out, self.discriminators['features'][idx], f'discriminator_feature_{idx}') \
                          * self.config.feature_discriminator_factor
            g_loss += self._apply_generator_discriminator_loss(depth_out, self.discriminators['depth_image'], 'discriminator_depth_img') \
                      * self.config.img_discriminator_factor
            g_loss += self._apply_generator_discriminator_loss(original_phong_rendering, self.discriminators['phong'], 'discriminator_phong') \
                      * self.config.phong_discriminator_factor
            g_loss += self._apply_generator_discriminator_loss(depth_phong, self.discriminators['depth_phong'],
                                                               'discriminator_depth_phong') * self.config.phong_discriminator_factor

        if self.config.use_critic:
            feat_outs = encoder_outs_generated[::-1][:len(self.critics['features'])]
            for idx, feature_out in enumerate(feat_outs):
                generated_predicted = self.critics['features'][idx](feature_out).type_as(feature_out)
                g_loss += self._apply_generator_critic_loss(generated_predicted, f'critic_feature_{idx}') \
                          * self.config.feature_discriminator_factor
            valid_predicted_depth = self.critics['depth_image'](depth_out)
            g_loss += self._apply_generator_critic_loss(valid_predicted_depth, 'critic_depth_img') \
                      * self.config.img_discriminator_factor

            phong_discrimination = self.critics['phong'](original_phong_rendering)
            g_loss += self._apply_generator_critic_loss(phong_discrimination, 'critic_phong') \
                      * self.config.phong_discriminator_factor
            g_loss += self._apply_generator_critic_loss(self.critics['depth_phong'](depth_phong),
                                                        'critic_depth_phong') \
                      * self.config.phong_discriminator_factor

        batch[self.generated_source_id].extend([depth_out, normals_generated])
        texture_generator_loss = self.texture_generator.calculate_generator_loss(batch)
        g_loss += texture_generator_loss

        self.manual_backward(g_loss)
        self.g_losses_log['g_loss'] += g_loss.detach()
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.generator_global_step += 1
            self._generator_training = False
            optimizers = self.optimizers(True)
            optimizers = [optimizers[i] for i in [0, self.texture_generator_opt_idx]]
            [o.step() for o in optimizers]
            [o.zero_grad() for o in optimizers]
            self.log_dict(self.g_losses_log)
            self.log_dict(self.texture_generator.g_losses_log)
            self.reset_log_dict(self.g_losses_log)
            self.reset_log_dict(self.texture_generator.g_losses_log)
            self.zero_grad()

    def discriminator_critic_train_step(self, batch: Dict[int, List[Tensor]], batch_idx) -> None:
        self.generator.eval()
        self.critics.train()
        self.discriminators.train()
        optimizers = self.optimizers(True)

        predictions = self.get_discriminator_critic_inputs(batch, batch_idx)
        full_batch = self.batches_accumulated == self.config.accumulate_grad_batches
        first_of_mini_batches = self.critic_global_step % self.config.wasserstein_critic_updates == 0
        last_mini_batch = (self.critic_global_step + 1) % self.config.wasserstein_critic_updates == 0 \
            if self.config.use_critic else True

        if self.config.use_discriminator and first_of_mini_batches:
            # print('discriminator')
            discriminator_loss = self._discriminators(predictions)
            self.manual_backward(discriminator_loss)
            self.d_losses_log['d_discriminators_loss'] += discriminator_loss.detach()
            if full_batch:
                # print('discriminator step')
                discriminator_opt = optimizers[self.discriminators_opt_idx]
                discriminator_opt.step()
                discriminator_opt.zero_grad()

        if self.config.use_critic:
            # print('critic')
            critic_loss = self._critics(predictions)
            self.manual_backward(critic_loss)
            self.d_losses_log['d_critics_loss'] += critic_loss.detach()
            if full_batch:
                # print('critic step')
                o = optimizers[self.critic_opt_idx]
                o.step()
                o.zero_grad()
                self.critic_global_step += 1

        batch[self.generated_source_id].extend([predictions[self.generated_source_id]['depth'],
                                           predictions[self.generated_source_id]['normals']])

        if self.texture_generator.config.use_critic:
            critic_loss = self.texture_generator.calculate_critic_loss(batch)
            self.manual_backward(critic_loss)
            if full_batch:
                o = optimizers[self.texture_critic_opt_idx]
                o.step()
                o.zero_grad()
                if not self.config.use_critic:
                    self.critic_global_step += 1

        if self.texture_generator.config.use_discriminator and first_of_mini_batches:
            discriminator_loss = self.texture_generator.calculate_discriminator_loss(batch)
            self.manual_backward(discriminator_loss)
            if full_batch:
                o = optimizers[self.texture_discriminator_opt_idx]
                o.step()
                o.zero_grad()

        if last_mini_batch and full_batch:
            self._generator_training = True
            self.log_dict(self.d_losses_log)
            self.reset_log_dict(self.d_losses_log)
            self.log_dict(self.texture_generator.d_losses_log)
            self.reset_log_dict(self.texture_generator.d_losses_log)

    def get_discriminator_critic_inputs(self, batch, batch_idx) -> Dict[int, DiscriminatorCriticInputs]:
        """

        :param batch:
        :param batch_idx:
        :return: dict
        """
        with torch.no_grad():
            results = {}
            for source_id in batch:
                results[source_id] = {}
                z = batch[source_id][0]
                encoder_outs, encoder_mare_outs, decoder_outs, normals = self(z,
                                                                              generator=source_id == self.generated_source_id)
                depth = decoder_outs[-1]
                results[source_id]['color'] = z
                results[source_id]['encoder_outs'] = [e.detach() for e in encoder_outs]
                results[source_id]['depth'] = depth.detach()
                results[source_id]['normals'] = normals.detach()
                phong = self.phong_renderer((depth, normals))
                calculated_phong = self.phong_renderer(
                    (depth, depth_to_normals(depth, self.phong_renderer.camera_intrinsics[None],
                                             self.phong_renderer.resized_pixel_locations)))
                results[source_id]['phong'] = phong.detach()
                results[source_id]['calculated_phong'] = calculated_phong.detach()
        return results

    def _discriminators(self, predictions: Dict[int, DiscriminatorCriticInputs]) -> Tensor:
        depth_generated = predictions[self.generated_source_id]['depth']
        encoder_outs_generated = predictions[self.generated_source_id]['encoder_outs']
        phong_generated = predictions[self.generated_source_id]['phong']
        calculated_phong_generated = predictions[self.generated_source_id]['calculated_phong']

        depth_original = []
        encoder_outs_original = []
        phong_original = []
        calculated_phong_original = []
        for source_id in predictions:
            if source_id == self.generated_source_id:
                break
            depth_original.append(predictions[source_id]['depth'])
            encoder_outs_original.append(predictions[source_id]['encoder_outs'])
            phong_original.append(predictions[source_id]['phong'])
            calculated_phong_original.append(predictions[source_id]['calculated_phong'])

        depth_original = torch.cat(depth_original, dim=0)
        encoder_outs_original = [torch.cat([s[i] for s in encoder_outs_original], dim=0) for i in
                              range(len(encoder_outs_original[0]))]
        phong_original = torch.cat(phong_original, dim=0)
        calculated_phong_original = torch.cat(calculated_phong_original, dim=0)

        loss: Tensor = 0.0
        loss += self._apply_discriminator_loss(depth_generated,
                                               depth_original,
                                               self.discriminators['depth_image'],
                                               'discriminator_depth_img')
        feat_outs = zip(encoder_outs_generated[::-1], encoder_outs_original[::-1])
        for idx, d_feat in enumerate(self.discriminators['features']):
            feature_out_r, feature_out_s = next(feat_outs)
            loss += self._apply_discriminator_loss(feature_out_r,
                                                   feature_out_s,
                                                   d_feat,
                                                   f'discriminator_feature_{idx}')

        loss += self._apply_discriminator_loss(phong_generated,
                                               phong_original,
                                               self.discriminators['phong'],
                                               'discriminator_phong')
        loss += self._apply_discriminator_loss(calculated_phong_generated,
                                               calculated_phong_original,
                                               self.discriminators['depth_phong'],
                                               'discriminator_depth_phong')
        return loss

    def _critics(self, predictions: Dict[int, DiscriminatorCriticInputs]) -> Tensor:
        depth_generated = predictions[self.generated_source_id]['depth']
        encoder_outs_generated = predictions[self.generated_source_id]['encoder_outs']
        phong_generated = predictions[self.generated_source_id]['phong']
        calculated_phong_generated = predictions[self.generated_source_id]['calculated_phong']

        depth_original = []
        encoder_outs_original = []
        phong_original = []
        calculated_phong_original = []
        for source_id in predictions:
            if source_id == self.generated_source_id:
                break
            depth_original.append(predictions[source_id]['depth'])
            encoder_outs_original.append(predictions[source_id]['encoder_outs'])
            phong_original.append(predictions[source_id]['phong'])
            calculated_phong_original.append(predictions[source_id]['calculated_phong'])

        depth_original = torch.cat(depth_original, dim=0)
        encoder_outs_original = [torch.cat([s[i] for s in encoder_outs_original], dim=0) for i in
                              range(len(encoder_outs_original[0]))]
        phong_original = torch.cat(phong_original, dim=0)
        calculated_phong_original = torch.cat(calculated_phong_original, dim=0)
        loss: Tensor = 0.0
        loss += self._apply_critic_loss(depth_generated, depth_original, self.critics['depth_image'],
                                        self.config.wasserstein_lambda, 'critic_depth_img')
        feat_outs = zip(encoder_outs_generated[::-1], encoder_outs_original[::-1])
        for idx, feature_critic in enumerate(self.critics['features']):
            feature_out_r, feature_out_s = next(feat_outs)
            loss += self._apply_critic_loss(feature_out_r, feature_out_s, feature_critic,
                                            self.config.wasserstein_lambda, f'critic_feature_{idx}')

        loss += self._apply_critic_loss(phong_generated, phong_original, self.critics['phong'],
                                        self.config.wasserstein_lambda, 'critic_phong')
        loss += self._apply_critic_loss(calculated_phong_generated, calculated_phong_original, self.critics['depth_phong'],
                                        self.config.wasserstein_lambda, 'critic_depth_phong')
        return loss

    def _apply_generator_discriminator_loss(self, discriminator_in: Tensor, discriminator: torch.nn.Module, name: str,
                                            label: float = 1.0) -> Tensor:
        loss = self.generator_discriminator_loss(discriminator_in, label, discriminator)
        self.g_losses_log[f'g_loss_{name}'] += loss.detach()
        return loss

    def _apply_generator_critic_loss(self, discriminator_out: Tensor, name: str, ) -> Tensor:
        loss = self.generator_critic_loss(discriminator_out)
        self.g_losses_log[f'g_loss_{name}'] += loss.detach()
        return loss

    def _apply_discriminator_loss(self, generated: Tensor, original: Tensor, discriminator: torch.nn.Module,
                                  name: str) -> Tensor:
        loss_generated = self.discriminator_loss(generated, 0.0, discriminator)
        loss_original = self.discriminator_loss(original, 1.0, discriminator)
        combined = (loss_original + loss_generated) / 2
        self.d_losses_log[f'd_loss_{name}'] += combined.detach()
        return combined

    def _apply_critic_loss(self, generated: Tensor, original: Tensor, critic: torch.nn.Module,
                           wasserstein_lambda: float, name: str):
        critic_loss = self.critic_loss(generated, original, critic, wasserstein_lambda)
        self.d_losses_log[f'd_loss_{name}'] += critic_loss.detach()
        return critic_loss

    def validation_step(self, batch, batch_idx):
        """
        TODO: This function only does plotting... We need some sort of metric
        :param batch:
        :param batch_idx:
        :return:
        """
        self.eval()
        if batch_idx != 0:
            return

        with torch.no_grad():
            if batch_idx != 0 or self.validation_epoch % self.config.val_plot_interval != 0:
                # just plot one batch worth of images. In case there are a lot...
                return

            if self.validation_data is None:
                self.validation_data = {}
                for source_id in batch:
                    if source_id < self.generated_source_id:
                        self.validation_data[source_id] = [x[:2].detach() for x in batch[source_id]]
                        self.validation_data[source_id][0] = self.imagenet_denorm(self.validation_data[source_id][0]).detach()
                    else:
                        self.validation_data[source_id] = batch[source_id][0][:2].detach()
            self.plot()
            self.log_gate_coefficients(step=self.global_step)

    def test_step(self, *args: Any, **kwargs: Any):
        pass

    def configure_optimizers(self):
        return self._unwrapped_optimizers

    def _on_epoch_end(self):
        if self.lr_schedulers() is not None:
            self.log("generator_lr", self.lr_schedulers()[0].get_last_lr()[0])
            self.log("discriminator_lr", self.lr_schedulers()[1].get_last_lr()[0])

    def on_validation_epoch_end(self) -> None:
        self.validation_epoch += 1
        self._on_epoch_end()

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end()

    def log_gate_coefficients(self, step=None):
        if step is None:
            step = self.current_epoch
        # iterating through all parameters
        if self.config.encoder.adaptive_gating and self.config.encoder.residual_learning:
            for name, params in self.generator.named_parameters():
                if 'gate_coefficients' in name:
                    scalars = {str(i): params[i] for i in range(len(params))}
                    self.logger.experiment.add_scalars(name, scalars, step)

    def plot(self):
        with torch.no_grad():
            z = self.validation_data[self.generated_source_id]
            _, _, decoder_outs_adapted, normals_adapted = self(z, generator=True)
            depth_adapted = decoder_outs_adapted[-1].detach()
            denormed_images = self.imagenet_denorm(z).detach()
            self.validation_data[self.generated_source_id] = [denormed_images, depth_adapted, normals_adapted.detach()]
            self.texture_generator.validation_data = self.validation_data
            self.texture_generator.val_denorm_color_images = torch.cat([self.validation_data[i][0].detach().cpu() for i in self.validation_data], dim=0)
            self.texture_generator.plot(self.global_step)
            self.validation_data[self.generated_source_id] = z

            if self.unadapted_images_for_plotting is None:
                _, _, decoder_outs_unadapted, normals_unadapted = self(z, generator=False)
                depth_unadapted = decoder_outs_unadapted[-1].detach()
                phong_unadapted = self.phong_renderer((depth_unadapted, normals_unadapted)).detach().cpu()

                self.unadapted_images_for_plotting = (depth_unadapted.detach(), normals_unadapted.detach().cpu(), phong_unadapted)

            depth_unadapted, normals_unadapted, phong_unadapted = self.unadapted_images_for_plotting
            denormed_images = denormed_images.cpu()
            plot_tensors = [denormed_images]
            labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
            centers = [None, None, None, 0]
            minmax = []
            plot_tensors.append(depth_adapted.cpu())
            plot_tensors.append(depth_unadapted.cpu())
            plot_tensors.append((depth_adapted - depth_unadapted).cpu())

            for idx, imgs in enumerate(zip(*plot_tensors)):
                fig = generate_heatmap_fig(imgs, labels=labels, centers=centers, minmax=minmax,
                                           align_scales=True)
                self.logger.experiment.add_figure(f"GAN Prediction Result-{idx}", fig, self.global_step)
                plt.close(fig)
            phong_adapted = self.phong_renderer((depth_adapted, normals_adapted)).detach().cpu()

            labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted"]
            for idx, img_set in enumerate(zip(denormed_images, phong_adapted, phong_unadapted)):
                fig = generate_img_fig(img_set, labels)
                self.logger.experiment.add_figure(f'GAN-phong-{idx}', fig, self.global_step)
                plt.close(fig)
