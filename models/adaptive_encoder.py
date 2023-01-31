from utils.loss import AvgTensorNorm
from models.vanillaencoder import VanillaEncoder
from utils.torch_utils import convrelu
from torch import batch_norm, nn
import torch


class AdaptiveEncoder(VanillaEncoder):
    def __init__(self,
                 adaptive_gating: bool = False,
                 backbone: str = 'resnet18',
                 use_image_net_weights: bool = False):
        """

        :param adaptive_gating: whether to add the resnet blocks for adaptive transfer learning. If false,
                                behaves as a normal vanilla encoder.
        :param backbone: base encoder structure to use.
        :param use_image_net_weights: whether to initialize with imagenet weights
        """
        super().__init__(backbone=backbone, imagenet_weights=use_image_net_weights)
        init_zero = False
        activation = "leaky"
        norm = 'batch'
        self.adaptive_gating = adaptive_gating

        if not self.adaptive_gating:
            return

        for param in self.parameters():
            param.requires_grad = False

        self.gate_coefficients = nn.Parameter(torch.zeros(5), requires_grad=self.adaptive_gating, )

        self.res_layer0 = nn.Sequential(convrelu(3, 64, 5, 2, 1, relu=activation, norm=norm, init_zero=init_zero),
                                        convrelu(64, 3, 5, 2, 1, relu=activation, init_zero=init_zero, norm=norm))
        self.res_layer1 = nn.Sequential(convrelu(64, 64, 5, 2, 1, relu=activation, norm=norm, init_zero=init_zero),
                                        convrelu(64, 64, 5, 2, 1, relu=activation, init_zero=init_zero, norm=norm))
        self.res_layer2 = nn.Sequential(convrelu(64, 128, 5, 2, 1, relu=activation, norm=norm, init_zero=init_zero),
                                        convrelu(128, 64, 5, 2, 1, relu=activation, init_zero=init_zero, norm=norm))
        self.res_layer3 = nn.Sequential(
            convrelu(128, 256, 3, 1, 1, relu=activation, norm=norm, init_zero=init_zero),
            convrelu(256, 128, 3, 1, 1, relu=activation, init_zero=init_zero, norm=norm)
        )
        self.res_layer4 = nn.Sequential(
            convrelu(256, 512, 3, 1, 1, relu=activation, norm=norm, init_zero=init_zero),
            convrelu(512, 256, 3, 1, 1, relu=activation, init_zero=init_zero, norm=norm)
        )
        self.criterion = AvgTensorNorm()

    def _gate(self, input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, level: int) -> torch.Tensor:
        if self.adaptive_gating:
            return input_tensor1 + torch.tanh(self.gate_coefficients[level]) * input_tensor2
        else:
            return input_tensor1 + input_tensor2

    def forward(self, encoder_input):
        if not self.adaptive_gating:
            return VanillaEncoder.forward(self, encoder_input)
        x_original = self.conv_original_size0(encoder_input)
        x_original = self.conv_original_size1(x_original)

        res_layer0 = self.res_layer0(encoder_input)
        layer0 = self.layer0(self._gate(encoder_input, res_layer0, 0))

        res_layer1 = self.res_layer1(layer0)
        layer1 = self.layer1(self._gate(layer0, res_layer1, 1))

        res_layer2 = self.res_layer2(layer1)
        layer2 = self.layer2(self._gate(layer1, res_layer2, 2))

        res_layer3 = self.res_layer3(layer2)
        layer3 = self.layer3(self._gate(layer2, res_layer3, 3))

        res_layer4 = self.res_layer4(layer3)
        layer4 = self.layer4(self._gate(layer3, res_layer4, 4))

        layer0_mare = self.criterion(res_layer0)
        layer1_mare = self.criterion(res_layer1)
        layer2_mare = self.criterion(res_layer2)
        layer3_mare = self.criterion(res_layer3)
        layer4_mare = self.criterion(res_layer4)

        layer0 = self.layer0_skip(layer0)
        layer1 = self.layer1_skip(layer1)
        layer2 = self.layer2_skip(layer2)
        layer3 = self.layer3_skip(layer3)
        layer4 = self.layer4_skip(layer4)
        return ([x_original, layer0, layer1, layer2, layer3, layer4],
                [layer0_mare, layer1_mare, layer2_mare, layer3_mare, layer4_mare])


class ConditionalMeanRelativeLoss(nn.Module):
    def __init__(self):
        super(ConditionalMeanRelativeLoss, self).__init__()

    def forward(self, output, target):
        # calculate absolute errors
        absolute_output = torch.abs(output)
        # where target is too small, use just the absolute errors to avoid divide by 0
        # but clamp abs (target) away from zero to avoid "ghost" divide by 0
        abs_target = torch.abs(target).clamp(0.0005)
        loss = torch.where(abs_target < 0.001, absolute_output, torch.divide(absolute_output, abs_target))
        # return mean loss
        return torch.mean(loss)
