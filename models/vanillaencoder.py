import torch
from torchvision import models
from utils.torch_utils import convrelu
from torch import nn
from typing import *

_base_model = {'resnet18': models.resnet18,
               'resnet34': models.resnet34,
               'resnet50': models.resnet50}
_image_net_weights = {'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
                      'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
                      'resnet50': models.ResNet50_Weights.IMAGENET1K_V1}


def _get_output_features(module: torch.nn.Sequential):
    while isinstance(module._modules['1'], torch.nn.Sequential):
        module = module._modules['1']

    last_lyr = module._modules[list(module._modules.keys())[-1]]
    last_lyr_key = [k for k in last_lyr._modules.keys() if 'conv' in k][-1]
    return getattr(last_lyr, last_lyr_key).out_channels


class VanillaEncoder(torch.nn.Module):
    def __init__(self, backbone: str = 'resnet18', imagenet_weights: bool = True):
        super().__init__()
        self.base_model = _base_model['resnet18'](weights=_image_net_weights['resnet18'] if imagenet_weights else None)
        base_layers = list(self.base_model.children())
        self.feature_levels = [64, 64, 128, 256, 512]

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        self.layer0 = torch.nn.Sequential(*base_layers[:3])  # shape=(N, 64, x.H/2, x.W/2)
        self.layer0_skip = CoordConv2dELU(64, 64, 3, padding=1)
        self.layer1 = torch.nn.Sequential(*base_layers[3:5])  # shape=(N, 64, x.H/4, x.W/4)
        self.layer1_skip = CoordConv2dELU(64, 64, 3, padding=1)
        self.layer2 = base_layers[5]  # shape=(N, 128, x.H/8, x.W/8)
        self.layer2_skip = CoordConv2dELU(128, 128, 3, padding=1)
        self.layer3 = base_layers[6]  # shape=(N, 256, x.H/16, x.W/16)
        self.layer3_skip = CoordConv2dELU(256, 256, 3, padding=1)
        self.layer4 = base_layers[7]  # shape=(N, 512, x.H/32, x.W/32)
        self.layer4_skip = CoordConv2dELU(512, 512, 3, padding=1)

    def forward(self, encoder_input) -> Tuple[List[torch.Tensor], List]:
        x_original = self.conv_original_size0(encoder_input)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(encoder_input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer0 = self.layer0_skip(layer0)
        layer1 = self.layer1_skip(layer1)
        layer2 = self.layer2_skip(layer2)
        layer3 = self.layer3_skip(layer3)
        layer4 = self.layer4_skip(layer4)
        # return outs and residual outs (no residual outs for the standard encoder)
        return [x_original, layer0, layer1, layer2, layer3, layer4], []

    def __call__(self, *args, **kwargs) -> Tuple[List[torch.Tensor], List]:
        return super(VanillaEncoder, self).__call__(*args, **kwargs)


class CoordConv2dELU(nn.modules.conv.Conv2d):
    """ A coordinate convolution with an ELU activation """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param with_r:
        """
        super(CoordConv2dELU, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, groups, bias)
        self.rank = 2
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)
        self.elu = nn.ELU()

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = create_coordinate_layer(input_tensor)
        out = self.conv(out)
        out = self.elu(out)

        return out

    def __call__(self, *args, **kwargs):
        return super(CoordConv2dELU, self).__call__(*args, **kwargs)


def create_coordinate_layer(input_tensor) -> torch.Tensor:
    """
    :param input_tensor: shape (N, C_in, H, W)
    :return:
    """

    batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
    xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
    yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

    xx_range = torch.arange(dim_y, dtype=torch.int32)
    yy_range = torch.arange(dim_x, dtype=torch.int32)
    xx_range = xx_range[None, None, :, None]
    yy_range = yy_range[None, None, :, None]

    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)

    # transpose y
    yy_channel = yy_channel.permute(0, 1, 3, 2)

    xx_channel = xx_channel.float() / (dim_y - 1)
    yy_channel = yy_channel.float() / (dim_x - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
    yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

    xx_channel = xx_channel.type_as(input_tensor)
    yy_channel = yy_channel.type_as(input_tensor)
    out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
    return out
