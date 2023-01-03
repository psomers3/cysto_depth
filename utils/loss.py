import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import data.data_transforms as d_transforms
from utils.metrics import SILog
from utils.rendering import get_pixel_locations, get_image_size_from_intrisics, render_rgbd, PointLights, Materials
from config.training_config import PhongConfig
from typing import *


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.loss = torch.nn.CosineSimilarity(dim=1)

    def forward(self, predicted, target):
        predicted = F.normalize(predicted, dim=1)

        return (1 - self.loss(predicted, target)).mean()


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, predicted, target):
        # fill background with 
        if not predicted.shape == target.shape:
            _, _, h, w = target.shape
            predicted = F.interpolate(predicted, size=(h, w), mode='bilinear', align_corners=True)
        # mask = target > 0
        # predicted = predicted * mask
        diff = torch.abs(target - predicted)
        delta = self.threshold * torch.max(diff).item()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 - delta ** 2, 0., -delta ** 2.) + delta ** 2
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss


class GradientLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
        self.silog = SILog()

    def forward(self, predicted, target):
        if not predicted.shape == target.shape:
            _, _, H, W = target.shape
            predicted = F.interpolate(predicted, size=(H, W), mode='bilinear', align_corners=True)
        p_x, p_y = self.get_gradient(predicted)
        t_x, t_y = self.get_gradient(target)
        dy = p_y - t_y
        dx = p_x - t_x
        si_loss = self.silog(predicted, target)
        grad_loss = torch.mean(torch.pow(dx, 2) + torch.pow(dy, 2))
        return grad_loss

    def get_gradient(self, x):
        G_x = self.conv1(Variable(x))
        G_y = self.conv2(Variable(x))
        return G_x, G_y


class PhongLoss(nn.Module):
    def __init__(self, config: PhongConfig, image_size: int = 256, device=None) -> None:
        super(PhongLoss, self).__init__()
        self.config = config
        self.camera_intrinsics = torch.Tensor(config.camera_intrinsics, device='cpu')
        self.camera_intrinsics.requires_grad_(False)
        self.squarify = d_transforms.Squarify(image_size)
        # get the original camera pixel locations at the desired image resolution
        original_image_size = get_image_size_from_intrisics(self.camera_intrinsics)
        self.camera_intrinsics.to(device)
        pixels = get_pixel_locations(*original_image_size)
        self.resized_pixel_locations = self.squarify(torch.permute(pixels, (2, 0, 1)))
        self.resized_pixel_locations = torch.permute(self.resized_pixel_locations, (1, 2, 0)).to(device)
        self.resized_pixel_locations.requires_grad_(False)
        self.grey = torch.ones((image_size, image_size, 3), device=device) * .5
        self.grey.requires_grad_(False)
        self.material = Materials(shininess=config.material_shininess, device=device)
        self.material.requires_grad_(False)
        self.light = PointLights(location=(((0, 0, 0),),),
                                 diffuse_color=(config.diffusion_color,),
                                 specular_color=(config.specular_color,),
                                 ambient_color=(config.ambient_color,),
                                 attenuation_factor=config.attenuation,
                                 device=device)
        self.light.requires_grad_(False)
        self.image_loss = torch.nn.MSELoss()
        self.device = device

    def forward(self, predicted_depth_normals: Tuple[torch.Tensor, ...], true_phong: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param predicted_depth_normals:
        :param true_phong:
        :return: the loss value and the rendered images
        """
        depth, normals = predicted_depth_normals
        mask = depth < 0.1
        normals = torch.where(mask, normals, 0)
        rendered = render_rgbd(torch.permute(depth, (0, 2, 3, 1)),
                               self.grey,
                               normals.permute((0, 2, 3, 1)),
                               self.camera_intrinsics,
                               self.light,
                               self.material,
                               self.resized_pixel_locations,
                               device=self.device)
        rendered = rendered.permute(0, 3, 1, 2)
        return self.image_loss(rendered, true_phong), rendered


class AvgTensorNorm(nn.Module):
    def forward(self, predicted):
        avg_norm = torch.norm(predicted, p='fro')
        return avg_norm

