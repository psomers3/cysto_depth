from torch import nn
import torch
from torch import Tensor
from config.training_config import EncoderConfig
from models.adaptive_encoder import AdaptiveEncoder
from models.decoder import Decoder


class DepthNorm2Image(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, depth_scale: float = 1e-3, add_noise: bool = False):
        super(DepthNorm2Image, self).__init__()
        self.add_noise = add_noise
        in_channels = 6 if self.add_noise else 5
        self.encoder = AdaptiveEncoder(encoder_config, num_input_channels=in_channels)
        self.decoder = Decoder(self.encoder.feature_levels[::-1], num_output_channels=3)
        self.sigmoid = torch.nn.Sigmoid()
        self.depth_scale = depth_scale

    def forward(self, depth: Tensor, normals: Tensor, source_id: int) -> Tensor:
        """

        :param depth: depth of shape [N, 1, h, w]
        :param normals: normals of shape [N, 3, h, w]
        :param source_id: id for which original domain it should try to create images for
        :return: a generated image of shape [N, 3, h, w]
        """
        source_layer = torch.full_like(depth,
                                       fill_value=source_id,
                                       device=depth.device,
                                       dtype=depth.dtype)
        inputs = [depth*self.depth_scale, normals, source_layer]
        if self.add_noise:
            noise = torch.rand_like(depth, device=depth.device, dtype=depth.dtype)
            inputs.append(noise)
        stacked = torch.cat(inputs, dim=1)
        skip_outs, _ = self.encoder(stacked)
        return self.sigmoid(self.decoder(skip_outs))

    def __call__(self, *args, **kwargs) -> Tensor:
        return super(DepthNorm2Image, self).__call__(*args, **kwargs)
