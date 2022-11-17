import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.metrics import SILog


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


class AvgTensorNorm(nn.Module):
    def forward(self, predicted):
        avg_norm = torch.norm(predicted, p='fro')
        return avg_norm

# # https://github.com/ansj11/SANet
# class GradLoss2(nn.Module):
#     def __init__(self):
#         super(GradLoss2, self).__init__()

#     def forward(self, fake, real, mask=None):
#         if not fake.shape == real.shape:
#             _, _, H, W = real.shape
#             fake = F.upsample(fake, size=(H, W), mode='bilinear')
#         real = real + (mask == 0).float() * fake
#         scales = [1, 2, 4, 8, 16]
#         grad_loss = 0
#         for scale in scales:
#             pre_dx, pre_dy, pre_m_dx, pre_m_dy = gradient2(fake, mask, scale)
#             gt_dx, gt_dy, gt_m_dx, gt_m_dy = gradient2(real, mask, scale)
#             diff_x = pre_dx - gt_dx
#             diff_y = pre_dy - gt_dy
#             grad_loss += torch.sum(torch.abs(diff_x*pre_m_dx))/(torch.sum(pre_m_dx) + 1e-6) + torch.sum(torch.abs(diff_y*pre_m_dy))/(torch.sum(pre_m_dy) + 1e-6)

#         return grad_loss

# def gradient(depth, mask):
#     D_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
#     D_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
#     mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
#     mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
#     return D_dx, D_dy, mask_dx, mask_dy
