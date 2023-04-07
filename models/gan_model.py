from matplotlib import pyplot as plt
import torch
from torch import Tensor
from models.base_model import BaseModel
from models.adaptive_encoder import AdaptiveEncoder
from models.discriminator import Discriminator
from models.depth_model import DepthEstimationModel
from data.data_transforms import ImageNetNormalization
from utils.image_utils import generate_heatmap_fig, freeze_batchnorm, generate_img_fig
from config.training_config import SyntheticTrainingConfig, GANTrainingConfig, DiscriminatorConfig
from argparse import Namespace
from utils.rendering import PhongRender, depth_to_normals
from utils.loss import GANDiscriminatorLoss, GANGeneratorLoss, CosineSimilarity
from typing import *

opts = {'adam': torch.optim.Adam, 'radam': torch.optim.RAdam, 'rmsprop': torch.optim.RMSprop}


class Predictions(TypedDict):
    """ Prediction dictionary collected before calculating discriminator and critic losses """
    encoder_outs_generated: List[Tensor]
    encoder_outs_original: List[Tensor]
    depth_generated: Tensor
    depth_original: Tensor
    phong_original: Tensor
    phong_generated: Tensor
    calculated_phong_original: Tensor
    calculated_phong_generated: Tensor
    normals_generated: Tensor
    normals_original: Tensor


class GAN(BaseModel):
    def __init__(self, synth_config: SyntheticTrainingConfig, gan_config: GANTrainingConfig):
        super().__init__()
        self.automatic_optimization = False
        ckpt = None
        self._start_step = -1
        if gan_config.resume_from_checkpoint:
            ckpt = torch.load(gan_config.resume_from_checkpoint, map_location=self.device)
            self._start_step = ckpt['global_step']
            hparams = ckpt['hyper_parameters']
            hparams['resume_from_checkpoint'] = gan_config.resume_from_checkpoint
            [setattr(gan_config, key, val) for key, val in hparams.items() if key in gan_config]

        self.save_hyperparameters(Namespace(**gan_config))
        self.depth_model: DepthEstimationModel = DepthEstimationModel(synth_config)
        gan_config.encoder.backbone = self.depth_model.config.encoder.backbone
        self.config = gan_config
        self.generator = AdaptiveEncoder(gan_config.encoder)
        self.generator.load_state_dict(self.depth_model.encoder.state_dict(), strict=False)
        self.depth_model.requires_grad = False
        self.imagenet_denorm = ImageNetNormalization(inverse=True)
        self.phong_renderer: PhongRender = None
        self._num_feature_level_disc: int = 3
        self.discriminator_losses: Dict[str, Union[float, Tensor]] = {}
        self.generator_losses = {'g_loss': 0.0}
        if gan_config.predict_normals:
            self.generator_losses['g_normals_similarity'] = 0.0
        self.critic_losses: Dict[str, Union[float, Tensor]] = {}
        self.discriminators: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.critics: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.critic_opt_idx = 0
        self.discriminators_opt_idx = 0
        self._unwrapped_optimizers = []
        self._unwrapped_optimizer_sched = []
        self.setup_generator_optimizer()
        if self.config.use_critic:
            self.setup_critics()
        if self.config.use_discriminator:
            self.setup_discriminators()
        self.calculated_normals_loss: CosineSimilarity = None
        self.validation_epoch = 0
        self.generator_global_step = -1
        self.critic_global_step = 0
        self.total_train_step_count = -1
        self.batches_accumulated = 0
        self._generator_training = False
        self.unadapted_images_for_plotting = None
        self.discriminator_loss = GANDiscriminatorLoss[gan_config.discriminator_loss.lower()]
        self.critic_loss = GANDiscriminatorLoss[gan_config.critic_loss.lower()]
        self.generator_critic_loss = GANGeneratorLoss[gan_config.critic_loss.lower()]
        self.generator_discriminator_loss = GANGeneratorLoss[gan_config.discriminator_loss.lower()]
        if gan_config.resume_from_checkpoint:
            self._resume_from_checkpoint(ckpt)

    def _resume_from_checkpoint(self, ckpt: dict):
        with torch.no_grad():
            # run some data through the network to initial dense layers in discriminators if needed
            encoder_outs, encoder_mare_outs, decoder_outs, normals = self(
                torch.ones(1, 3, self.config.image_size, self.config.image_size, device=self.device))

            for k in range(self.config.discriminator_ensemble_size):
                feat_outs = encoder_outs[-len(self.discriminators['features-0']):]

                if self.config.use_discriminator:
                    if self.config.use_feature_level:
                        for idx, feature_out in enumerate(feat_outs):
                            self.discriminators[f'features-{k}'][idx](feature_out)
                    self.discriminators[f'depth_image-{k}'](decoder_outs[-1])
                    if self.config.predict_normals:
                        self.discriminators[f'phong-{k}'](normals)
                        self.discriminators[f'depth_phong-{k}'](normals)
                        self.discriminators[f'normals-{k}'](normals)

            if self.config.use_critic:
                feat_outs = encoder_outs[-len(self.critic_losses['features']):]
                if self.config.use_feature_level:
                    for idx, feature_out in enumerate(feat_outs):
                        self.critics['features'][idx](feature_out)
                self.critics['depth_image'](decoder_outs[-1])
                if self.config.predict_normals:
                    self.critics['phong'](normals)
                    self.critics['depth_phong'](normals)
                    self.critics['normals'](normals)

        self.load_state_dict(ckpt['state_dict'], strict=False)

    def setup_generator_optimizer(self):
        opt = opts[self.config.generator_optimizer.lower()]
        gen_optimizer = opt(filter(lambda p: p.requires_grad, self.generator.parameters()),
                            lr=self.config.generator_lr)
        self._unwrapped_optimizers.append(gen_optimizer)
        last_epoch = self._start_step
        if None not in [self.config.lr_scheduler_step_size, self.config.lr_scheduler_gamma]:
            gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=self.config.lr_scheduler_step_size,
                                                            gamma=self.config.lr_scheduler_gamma, last_epoch=last_epoch)
            self._unwrapped_optimizer_sched.append(gen_scheduler)

    def setup_discriminators(self):
        self.discriminator_losses['d_discriminators_loss'] = 0.0
        for k in range(self.config.discriminator_ensemble_size):
            if self.config.use_feature_level:
                d_feat_list = []
                for d_in_shape in self.generator.feature_levels[-self._num_feature_level_disc:]:
                    d_config: DiscriminatorConfig = self.config.feature_level_discriminator.copy()
                    d_config.in_channels = d_in_shape
                    d = Discriminator(d_config)
                    d_feat_list.append(d)
                self.discriminators[f'features-{k}'] = torch.nn.ModuleList(modules=d_feat_list)
                self.discriminator_losses.update(
                    {f'd_loss_discriminator_feature-{k}_{i}': 0.0 for i in range(len(d_feat_list))})
                self.discriminator_losses.update(
                    {f'd_loss_reg_discriminator_feature-{k}_{i}': 0.0 for i in range(len(d_feat_list))})
                self.generator_losses.update(
                    {f'g_loss_discriminator_feature-{k}_{i}': 0.0 for i in range(len(d_feat_list))})
            if self.config.predict_normals:
                self.discriminators[f'phong-{k}'] = Discriminator(self.config.phong_discriminator)
                # self.discriminators[f'depth_phong-{k}'] = Discriminator(self.config.phong_discriminator)
                self.discriminators[f'normals-{k}'] = Discriminator(self.config.normals_discriminator)
                self.discriminator_losses.update({f'd_loss_discriminator_phong-{k}': 0.0,
                                                  # f'd_loss_discriminator_depth_phong-{k}': 0.0,
                                                  f'd_loss_discriminator_normals-{k}': 0.0})
                self.discriminator_losses.update({f'd_loss_reg_discriminator_phong-{k}': 0.0,
                                                  # f'd_loss_reg_discriminator_depth_phong-{k}': 0.0,
                                                  f'd_loss_reg_discriminator_normals-{k}': 0.0})
                self.generator_losses.update({f'g_loss_discriminator_phong-{k}': 0.0,
                                              # f'g_loss_discriminator_depth_phong-{k}': 0.0,
                                              f'g_loss_discriminator_normals-{k}': 0.0})
            self.discriminators[f'depth_image-{k}'] = Discriminator(self.config.depth_discriminator)
            self.discriminator_losses.update({f'd_loss_discriminator_depth_img-{k}': 0.0,
                                              f'd_loss_reg_discriminator_depth_img-{k}': 0.0})
            self.generator_losses.update({f'g_loss_discriminator_depth_img-{k}': 0.0})
        opt = opts[self.config.discriminator_optimizer.lower()]
        disc_opt = opt(filter(lambda p: p.requires_grad, self.discriminators.parameters()),
                       lr=self.config.discriminator_lr)
        self._unwrapped_optimizers.append(disc_opt)
        self.discriminators_opt_idx = self.critic_opt_idx + 1
        last_epoch = self._start_step
        if None not in [self.config.lr_scheduler_step_size, self.config.lr_scheduler_gamma]:
            disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_opt, step_size=self.config.lr_scheduler_step_size,
                                                             gamma=self.config.lr_scheduler_gamma,
                                                             last_epoch=last_epoch)
            self._unwrapped_optimizer_sched.append(disc_scheduler)

    def setup_losses(self):
        if self.calculated_normals_loss is None:
            self.calculated_normals_loss = CosineSimilarity(device=self.device, ignore_direction=True)

    def setup_critics(self):
        self.critic_losses['d_critics_loss'] = 0.0
        if self.config.use_feature_level:
            d_feat_list = []
            for d_in_shape in self.generator.feature_levels[-self._num_feature_level_disc:]:
                d_config: DiscriminatorConfig = self.config.feature_level_critic.copy()
                d_config.in_channels = d_in_shape
                d = Discriminator(d_config)
                d_feat_list.append(d)
            self.critics['features'] = torch.nn.ModuleList(modules=d_feat_list)
            self.critic_losses.update({f'd_loss_critic_feature_{i}': 0.0 for i in range(len(d_feat_list))})
            self.generator_losses.update({f'g_loss_critic_feature_{i}': 0.0 for i in range(len(d_feat_list))})
            self.critic_losses.update({f'd_loss_critic_gp_feature_{i}': 0.0 for i in range(len(d_feat_list))})
        if self.config.predict_normals:
            self.critics['phong'] = Discriminator(self.config.phong_critic)
            self.critics['depth_phong'] = Discriminator(self.config.phong_critic)
            self.critics['normals'] = Discriminator(self.config.normals_critic)
            self.critic_losses.update({'d_loss_critic_phong': 0.0,
                                       'd_loss_critic_depth_phong': 0.0,
                                       'd_loss_critic_normals': 0.0,
                                       'd_loss_critic_gp_phong': 0.0,
                                       'd_loss_critic_gp_depth_phong': 0.0,
                                       'd_loss_critic_gp_normals': 0.0})
            self.generator_losses.update({'g_loss_critic_phong': 0.0, 'g_loss_critic_depth_phong': 0.0,
                                          'g_loss_critic_normals': 0.0})
        self.critics['depth_image'] = Discriminator(self.config.depth_critic)
        self.critic_losses.update({'d_loss_critic_depth_img': 0.0, 'd_loss_critic_gp_depth_img': 0.0})
        self.generator_losses.update({'g_loss_critic_depth_img': 0.0})
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
        return super(GAN, self).__call__(*args, **kwargs)

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
        if self.config.predict_normals:
            if self.depth_model.config.merged_decoder:
                output = self.depth_model.decoder(encoder_outs)
                decoder_outs = [output[i][:, 0, ...].unsqueeze(1) for i in range(len(output))]
                normals = output[-1][:, 1:, ...]
            else:
                normals = self.depth_model.normals_decoder(encoder_outs)
                decoder_outs = self.depth_model.decoder(encoder_outs)
            normals = torch.where(decoder_outs[-1] > self.depth_model.config.min_depth,
                                  normals, torch.zeros([1], device=self.device))
        else:
            decoder_outs = self.depth_model.decoder(encoder_outs)
            normals = None

        return encoder_outs, encoder_mare_outs, decoder_outs, normals

    def training_step(self, batch, batch_idx):
        self.setup_losses()
        # start by freezing all batchnorm layers throughout the networks that shouldn't update statistics
        self.train()
        self.depth_model.apply(freeze_batchnorm)
        if self.config.freeze_batch_norm:
            self.generator.apply(freeze_batchnorm)

        self.total_train_step_count += 1
        if self._generator_training:
            # print('generator')
            self.generator_train_step(batch, batch_idx)
        else:
            self.discriminator_critic_train_step(batch, batch_idx)

    def generator_train_step(self, batch, batch_idx) -> None:
        # x = original image, z = generated image
        _, z = batch
        # set discriminators to eval so that any normalization statistics don't get updated
        self.discriminators.eval()
        self.critics.eval()
        if self.config.encoder.residual_learning:
            self.generator.set_residuals_train()

        # output of encoder when evaluating a generated image
        encoder_outs_generated, encoder_mare_outs_generated, decoder_outs_generated, normals_generated = self(z,
                                                                                                              generator=True)
        depth_out = decoder_outs_generated[-1]

        g_loss: Tensor = 0.0
        normals_similarity_loss = 0.0
        if self.config.predict_normals:
            original_phong_rendering = self.phong_renderer((depth_out, normals_generated))
            calculated_norms = depth_to_normals(depth_out, self.phong_renderer.camera_intrinsics[None],
                                                self.phong_renderer.resized_pixel_locations)
            # depth_phong = self.phong_renderer((depth_out, calculated_norms))
            normalized_generated = torch.nn.functional.normalize(normals_generated, dim=1)
            normals_generated = torch.where(normals_generated == 0, normals_generated, normalized_generated)
            normals_similarity_loss += self.calculated_normals_loss(calculated_norms, normals_generated)
            self.generator_losses['g_normals_similarity'] += normals_similarity_loss
            # g_loss += normals_similarity_loss

        if self.config.use_discriminator:
            discriminator_losses = []
            for k in range(self.config.discriminator_ensemble_size):
                if self.config.use_feature_level:
                    feat_outs = encoder_outs_generated[-len(self.discriminators[f'features-{k}']):]
                    for idx, feature_out in enumerate(feat_outs):
                        discriminator_losses.append(
                            self._apply_generator_discriminator_loss(feature_out,
                                                                     self.discriminators[f'features-{k}'][idx],
                                                                     f'discriminator_feature-{k}_{idx}'))
                discriminator_losses.append(
                    self._apply_generator_discriminator_loss(depth_out, self.discriminators[f'depth_image-{k}'],
                                                             f'discriminator_depth_img-{k}'))
                if self.config.predict_normals:
                    discriminator_losses.append(self._apply_generator_discriminator_loss(original_phong_rendering,
                                                                                         self.discriminators[
                                                                                             f'phong-{k}'],
                                                                                         f'discriminator_phong-{k}'))
                    # discriminator_losses.append(
                    #     self._apply_generator_discriminator_loss(depth_phong, self.discriminators[f'depth_phong-{k}'],
                    #                                              f'discriminator_depth_phong-{k}'))
                    discriminator_losses.append(
                        self._apply_generator_discriminator_loss(normals_generated, self.discriminators[f'normals-{k}'],
                                                                 f'discriminator_normals-{k}'))
            if self.config.predict_normals:
                discriminator_losses.append(normals_similarity_loss)
            discriminator_losses = torch.stack(discriminator_losses)
            if self.config.hyper_volume_slack > 1.0:
                factors = self.hypervolume_optimization_coefficients(discriminator_losses)
                discriminator_losses = factors * discriminator_losses
            g_loss += discriminator_losses.sum()

        if self.config.use_critic:
            if self.config.use_feature_level:
                feat_outs = encoder_outs_generated[-len(self.critics['features']):]
                for idx, feature_out in enumerate(feat_outs):
                    generated_predicted = self.critics['features'][idx](feature_out).type_as(feature_out)
                    g_loss += self._apply_generator_critic_loss(generated_predicted, f'critic_feature_{idx}') \
                              * self.config.feature_discriminator_factor
            valid_predicted_depth = self.critics['depth_image'](depth_out)
            g_loss += self._apply_generator_critic_loss(valid_predicted_depth, 'critic_depth_img') \
                      * self.config.img_discriminator_factor

            if self.config.predict_normals:
                phong_discrimination = self.critics['phong'](original_phong_rendering)
                g_loss += self._apply_generator_critic_loss(phong_discrimination, 'critic_phong') \
                          * self.config.phong_discriminator_factor
                # g_loss += self._apply_generator_critic_loss(self.critics['depth_phong'](depth_phong),
                #                                             'critic_depth_phong') \
                #           * self.config.phong_discriminator_factor
                g_loss += self._apply_generator_critic_loss(self.critics['normals'](normals_generated),
                                                            'critic_normals') \
                          * self.config.normals_discriminator_factor

        self.manual_backward(g_loss)
        self.generator_losses['g_loss'] += g_loss.detach()
        self.batches_accumulated += 1
        if self.batches_accumulated == self.config.accumulate_grad_batches:
            self.generator_global_step += 1
            self.batches_accumulated = 0
            self._generator_training = False
            generator_opt = self.optimizers(True)[0]
            generator_opt.step()
            generator_opt.zero_grad()
            self.zero_grad()
            self.log_dict(self.generator_losses)
            self.reset_log_dict(self.generator_losses)
            schedulers = self.lr_schedulers()
            if schedulers is not None:
                scheduler = schedulers[0]
                scheduler.step()

    def discriminator_critic_train_step(self, batch, batch_idx) -> None:
        self.generator.eval()
        self.critics.train()
        self.discriminators.train()

        predictions = self.get_discriminator_critic_inputs(batch, batch_idx)
        self.batches_accumulated += 1
        full_batch = self.batches_accumulated == self.config.accumulate_grad_batches
        if full_batch:
            self.batches_accumulated = 0
        first_of_mini_batches = self.critic_global_step % self.config.wasserstein_critic_updates == 0
        last_mini_batch = (self.critic_global_step + 1) % self.config.wasserstein_critic_updates == 0 \
            if self.config.use_critic else True

        if self.config.use_discriminator and first_of_mini_batches:
            # print('discriminator')
            discriminator_loss = self._discriminators(predictions)
            self.manual_backward(discriminator_loss)
            self.discriminator_losses['d_discriminators_loss'] += discriminator_loss.detach()
            # +1 because they are stepped after critic update
            if full_batch:
                # print('discriminator step')
                discriminator_opt = self.optimizers(True)[self.discriminators_opt_idx]
                discriminator_opt.step()
                discriminator_opt.zero_grad()
                self.discriminator_losses.update({k: self.discriminator_losses[k] / self.config.accumulate_grad_batches
                                                  for k in self.discriminator_losses.keys()})
                self.log_dict(self.discriminator_losses)
                self.reset_log_dict(self.discriminator_losses)
                schedulers = self.lr_schedulers()
                if schedulers is not None:
                    scheduler = schedulers[self.discriminators_opt_idx]
                    scheduler.step()

        if self.config.use_critic:
            # print('critic')
            critic_loss = self._critics(predictions)
            self.manual_backward(critic_loss)
            self.critic_losses['d_critics_loss'] += critic_loss.detach()
            if full_batch:
                # print('critic step')
                critic_opt = self.optimizers(True)[self.critic_opt_idx]
                critic_opt.step()
                critic_opt.zero_grad()
                self.critic_global_step += 1
                self.critic_losses.update({k: self.critic_losses[k] / self.config.accumulate_grad_batches
                                           for k in self.critic_losses.keys()})
                self.log_dict(self.critic_losses)
                self.reset_log_dict(self.critic_losses)

        if last_mini_batch and full_batch:
            self._generator_training = True

    def get_discriminator_critic_inputs(self, batch, batch_idx) -> Predictions:
        """

        :param batch:
        :param batch_idx:
        :return: dict
        """
        x, z = batch
        results: Predictions = {}
        with torch.no_grad():
            # predictions with generated images through generator
            encoder_outs_generated, encoder_mare_outs_generated, decoder_outs_generated, normals_generated = self(z,
                                                                                                                  generator=True)
            # predictions with originaletic images through frozen network
            encoder_outs_original, encoder_mare_outs_original, decoder_outs_original, normals_original = self(x,
                                                                                                              generator=False)

            depth_generated = decoder_outs_generated[-1]
            depth_original = decoder_outs_original[-1]
            results['encoder_outs_generated'] = [e.detach() for e in encoder_outs_generated]
            results['encoder_outs_original'] = [e.detach() for e in encoder_outs_original]
            results['depth_generated'] = depth_generated.detach()
            results['depth_original'] = depth_original.detach()
            if self.config.predict_normals:
                phong_original = self.phong_renderer((depth_original, normals_original))
                phong_generated = self.phong_renderer((depth_generated, normals_generated))
                calculated_phong_original = self.phong_renderer((depth_original,
                                                                 depth_to_normals(depth_original,
                                                                                  self.phong_renderer.camera_intrinsics[
                                                                                      None],
                                                                                  self.phong_renderer.resized_pixel_locations)))
                calculated_phong_generated = self.phong_renderer((depth_generated,
                                                                  depth_to_normals(depth_generated,
                                                                                   self.phong_renderer.camera_intrinsics[
                                                                                       None],
                                                                                   self.phong_renderer.resized_pixel_locations)))
                results['phong_original'] = phong_original.detach()
                results['phong_generated'] = phong_generated.detach()
                results['calculated_phong_original'] = calculated_phong_original.detach()
                results['calculated_phong_generated'] = calculated_phong_generated.detach()
                results['normals_generated'] = normals_generated.detach()
                results['normals_original'] = normals_original.detach()

        return results

    def _discriminators(self, predictions: Predictions) -> Tensor:
        depth_generated = predictions['depth_generated']
        depth_original = predictions['depth_original']
        encoder_outs_generated = predictions['encoder_outs_generated']
        encoder_outs_original = predictions['encoder_outs_original']
        loss: Tensor = 0.0
        for k in range(self.config.discriminator_ensemble_size):
            loss += self._apply_discriminator_loss(depth_generated,
                                                   depth_original,
                                                   self.discriminators[f'depth_image-{k}'],
                                                   f'depth_img-{k}')
            if self.config.use_feature_level:
                feat_outs = zip(encoder_outs_generated[-self._num_feature_level_disc:],
                                encoder_outs_original[-self._num_feature_level_disc:])
                for idx, d_feat in enumerate(self.discriminators[f'features-{k}']):
                    feature_out_r, feature_out_s = next(feat_outs)
                    loss += self._apply_discriminator_loss(feature_out_r,
                                                           feature_out_s,
                                                           d_feat,
                                                           f'feature-{k}_{idx}')
            if self.config.predict_normals:
                phong_generated = predictions[f'phong_generated']
                phong_original = predictions[f'phong_original']
                calculated_phong_generated = predictions[f'calculated_phong_generated']
                calculated_phong_original = predictions['calculated_phong_original']
                loss += self._apply_discriminator_loss(phong_generated,
                                                       phong_original,
                                                       self.discriminators[f'phong-{k}'],
                                                       f'phong-{k}')
                # loss += self._apply_discriminator_loss(calculated_phong_generated,
                #                                        calculated_phong_original,
                #                                        self.discriminators[f'depth_phong-{k}'],
                #                                        f'depth_phong-{k}')
                loss += self._apply_discriminator_loss(predictions['normals_generated'],
                                                       predictions['normals_original'],
                                                       self.discriminators[f'normals-{k}'],
                                                       f'normals-{k}')
        return loss

    def _critics(self, predictions: Predictions) -> Tensor:
        depth_generated = predictions['depth_generated']
        depth_original = predictions['depth_original']
        encoder_outs_generated = predictions['encoder_outs_generated']
        encoder_outs_original = predictions['encoder_outs_original']
        loss: Tensor = 0.0
        loss += self._apply_critic_loss(depth_generated, depth_original, self.critics['depth_image'],
                                        self.config.wasserstein_lambda, 'depth_img')
        if self.config.use_feature_level:
            feat_outs = zip(encoder_outs_generated[-self._num_feature_level_disc:],
                            encoder_outs_original[-self._num_feature_level_disc:])
            for idx, feature_critic in enumerate(self.critics['features']):
                feature_out_r, feature_out_s = next(feat_outs)
                loss += self._apply_critic_loss(feature_out_r, feature_out_s, feature_critic,
                                                self.config.wasserstein_lambda, f'feature_{idx}')
        if self.config.predict_normals:
            phong_generated = predictions['phong_generated']
            phong_original = predictions['phong_original']
            calculated_phong_generated = predictions['calculated_phong_generated']
            calculated_phong_original = predictions['calculated_phong_original']
            loss += self._apply_critic_loss(phong_generated, phong_original, self.critics['phong'],
                                            self.config.wasserstein_lambda, 'phong')
            loss += self._apply_critic_loss(calculated_phong_generated, calculated_phong_original,
                                            self.critics['depth_phong'],
                                            self.config.wasserstein_lambda, 'depth_phong')
            loss += self._apply_critic_loss(predictions['normals_generated'], predictions['normals_original'],
                                            self.critics['normals'],
                                            self.config.wasserstein_lambda, 'normals')
        return loss

    def _apply_generator_discriminator_loss(self,
                                            discriminator_in: Tensor,
                                            discriminator: torch.nn.Module,
                                            name: str,
                                            label: float = 1.0) -> Tensor:
        loss, penalty = self.generator_discriminator_loss(discriminator_in, label, discriminator)
        self.generator_losses[f'g_loss_{name}'] += loss.detach()
        return loss

    def _apply_generator_critic_loss(self,
                                     discriminator_out: Tensor,
                                     name: str, ) -> Tensor:
        loss, penalty = self.generator_critic_loss(discriminator_out)
        self.generator_losses[f'g_loss_{name}'] += loss.detach()
        return loss

    def _apply_discriminator_loss(self,
                                  generated: Tensor,
                                  original: Tensor,
                                  discriminator: torch.nn.Module,
                                  name: str) -> Tensor:
        apply_original_reg = False
        apply_generated_reg = False
        if 'r1' in self.config.discriminator_loss.lower():
            apply_original_reg = True
        elif 'r2' in self.config.discriminator_loss.lower():
            apply_generated_reg = True

        loss_generated, gen_penalty = self.discriminator_loss(generated, 0.0, discriminator,
                                                              apply_regularization=apply_generated_reg)
        loss_original, org_penalty = self.discriminator_loss(original, 1.0, discriminator,
                                                             apply_regularization=apply_original_reg)
        combined_loss = loss_original + loss_generated
        combined_penalty = gen_penalty + org_penalty
        self.discriminator_losses[f'd_loss_discriminator_{name}'] += combined_loss.detach()
        self.discriminator_losses[f'd_loss_reg_discriminator_{name}'] += combined_penalty.detach()
        return combined_loss + combined_penalty

    def _apply_critic_loss(self, generated: Tensor, original: Tensor, critic: torch.nn.Module,
                           wasserstein_lambda: float, name: str):
        critic_loss, penalty = self.critic_loss(generated, original, critic, wasserstein_lambda)
        self.critic_losses[f'd_loss_critic_{name}'] += critic_loss.detach()
        self.critic_losses[f'd_loss_critic_gp_{name}'] += penalty.detach()
        return critic_loss + penalty

    def validation_step(self, batch, batch_idx):
        """
        TODO: This function only does plotting... We need some sort of metric
        :param batch:
        :param batch_idx:
        :return:
        """
        self.eval()
        with torch.no_grad():
            if batch_idx != 0 or self.validation_epoch % self.config.val_plot_interval != 0:
                # just plot one batch worth of images. In case there are a lot...
                return
            x, z = batch
            # predictions with generated images through generator
            _, _, decoder_outs_adapted, normals_adapted = self(z, generator=True)
            depth_adapted = decoder_outs_adapted[-1]

            if self.unadapted_images_for_plotting is None:
                _, _, decoder_outs_unadapted, normals_unadapted = self(z, generator=False)
                depth_unadapted = decoder_outs_unadapted[-1].detach()
                if normals_unadapted is not None:
                    phong_unadapted = self.phong_renderer((depth_unadapted, normals_unadapted)).cpu()
                else:
                    phong_unadapted = None
                self.unadapted_images_for_plotting = (depth_unadapted,
                                                      normals_unadapted.detach() if normals_unadapted
                                                                                    is not None else normals_unadapted,
                                                      phong_unadapted.detach() if phong_unadapted
                                                                                  is not None else phong_unadapted)

            depth_unadapted, normals_unadapted, phong_unadapted = self.unadapted_images_for_plotting
            denormed_images = torch.clamp(self.imagenet_denorm(z).cpu(), 0, 1)
            plot_tensors = [denormed_images]
            labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted", "Diff"]
            centers = [None, None, None, 0]
            minmax = []
            plot_tensors.append(depth_adapted)
            plot_tensors.append(depth_unadapted.cpu())
            plot_tensors.append((depth_adapted - depth_unadapted).cpu())

            self.log_gate_coefficients(step=self.global_step)
            for idx, imgs in enumerate(zip(*plot_tensors)):
                fig = generate_heatmap_fig(imgs, labels=labels, centers=centers, minmax=minmax,
                                           align_scales=True)
                self.logger.experiment.add_figure(f"GAN Prediction Result-{idx}", fig, self.global_step)
                plt.close(fig)

            if normals_adapted is not None:
                phong_adapted = self.phong_renderer((depth_adapted, normals_adapted)).cpu()

                labels = ["Input Image", "Predicted Adapted", "Predicted Unadapted"]
                for idx, img_set in enumerate(zip(denormed_images, phong_adapted, phong_unadapted)):
                    fig = generate_img_fig(img_set, labels)
                    self.logger.experiment.add_figure(f'GAN-phong-{idx}', fig, self.global_step)
                    plt.close(fig)

    def hypervolume_optimization_coefficients(self, loss_vector):
        # Multi-objective training of GANs with multiple discriminators
        nadir_point = self.config.hyper_volume_slack * loss_vector.max()
        T = (1 / (nadir_point - loss_vector)).sum()
        return 1 / (T * (nadir_point - loss_vector))

    def configure_optimizers(self):
        if self._unwrapped_optimizer_sched:
            return self._unwrapped_optimizers, self._unwrapped_optimizer_sched
        else:
            return self._unwrapped_optimizers

    def _on_epoch_end(self):
        if self.lr_schedulers() is not None:
            self.log("generator_lr", self.lr_schedulers()[0].get_last_lr()[0])
            self.log("discriminator_lr", self.lr_schedulers()[self.discriminators_opt_idx].get_last_lr()[0])

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
