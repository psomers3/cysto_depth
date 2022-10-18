from torch import nn
import torch
from utils.torch_utils import convrelu


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
    def __init__(self) -> None:
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = UpsampleShuffle(512, 512)
        self.upsample2 = UpsampleShuffle(512, 512)
        self.upsample1 = UpsampleShuffle(256, 256)
        self.upsample0 = UpsampleShuffle(256, 256)
        self.upsample_final = UpsampleShuffle(128, 128)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_up_2_out = convrelu(256, 1, 3, 1)
        self.conv_up1_out = convrelu(256, 1, 3, 1)
        self.conv_up0_out = convrelu(128, 1, 3, 1)

        self.conv_original_size2 = convrelu(128, 64, 3, 1)
        # self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x_original, layer0, layer1, layer2, layer3, layer4 = input
        x = self.upsample3(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample2(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        out1 = self.conv_up_2_out(x)

        x = self.upsample1(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        out2 = self.conv_up1_out(x)

        x = self.upsample0(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        out3 = self.conv_up0_out(x)

        x = self.upsample_final(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out4 = self.conv_last(x)
        return [out1, out2, out3, out4]
