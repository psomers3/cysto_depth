from torch import nn
import torch
from utils.torch_utils import convrelu
from typing import *


def init_subpixel(weight):
    """ https://juliusruseckas.github.io/ml/pixel_shuffle.html """
    co, ci, h, w = weight.shape
    co2 = co // 4
    # initialize sub kernel
    k = torch.empty([co2, ci, h, w])
    nn.init.kaiming_uniform_(k)
    # repeat 4 times
    k = k.repeat_interleave(4, dim=0)
    weight.data.copy_(k)


class UpsampleShuffle(nn.Sequential):
    """ https://juliusruseckas.github.io/ml/pixel_shuffle.html """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def reset_parameters(self):
        init_subpixel(self[0].weight)
        nn.init.zeros_(self[0].bias)


class Decoder(torch.nn.Module):
    def __init__(self,
                 feature_levels: List[int],
                 num_output_channels: int = 1,
                 output_each_level: bool = False) -> None:
        """

        :param feature_levels: bottleneck to output
        :param num_output_channels: last convolutional layer feature number
        :param output_each_level: on call, whether to return output at each upscale
        """
        super().__init__()
        self.output_each_level = output_each_level
        self.num_output_channels = num_output_channels

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        feature_levels_doubled = feature_levels[:-2] * 2
        feature_levels_doubled.sort(reverse=True)
        self.upsamples = torch.nn.ModuleList([UpsampleShuffle(nfeat, nfeat) for nfeat in feature_levels_doubled[:-1]])

        self.conv_ups = torch.nn.ModuleList([convrelu(feature_levels[i+1] + feature_levels_doubled[i], nfeat, 3, 1)
                                             for i, nfeat in enumerate(feature_levels_doubled[1:-1])])

        if self.output_each_level:
            self.conv_ups_out = torch.nn.ModuleList([convrelu(nfeat, 1, 3, 1)
                                                     for nfeat in feature_levels_doubled[-4:-1]])

        self.conv_original_size2 = convrelu(feature_levels[-3], feature_levels[-1], 3, 1)
        # self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(feature_levels[-1], num_output_channels, 1)

    def forward(self, _input):
        x_original, layer0, layer1, layer2, layer3, layer4 = _input
        x = self.upsamples[0](layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_ups[0](x)

        x = self.upsamples[1](x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_ups[1](x)
        if self.output_each_level:
            out1 = self.conv_ups_out[0](x)

        x = self.upsamples[2](x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_ups[2](x)
        if self.output_each_level:
            out2 = self.conv_ups_out[1](x)

        x = self.upsamples[3](x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_ups[3](x)
        if self.output_each_level:
            out3 = self.conv_ups_out[2](x)

        x = self.upsamples[4](x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out4 = self.conv_last(x)

        if self.output_each_level:
            return [out1, out2, out3, out4]
        else:
            return out4