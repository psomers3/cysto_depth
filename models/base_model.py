import pytorch_lightning as pl
from utils.torch_utils import scale_median
from utils.metrics import RMSE, Accuracy, RMSELog, RelError, SILog, compute_errors
import torch
import numpy as np
from scipy.interpolate import griddata
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rmse_log = RMSELog()
        self.rmse = RMSE()
        self.rel_error = RelError()
        self.acc = Accuracy()
        self.silog = SILog()
        self.gan = False
        self.resize = nn.Sequential(
            transforms.Resize([256, 512]),  # We use single int value inside a list due to torchscript type restrictions
        )

    def calculate_metrics(self, prefix, predicted, target):
        depth_mask = target > 0
        target_depth = target[depth_mask]
        pred_depth = predicted[depth_mask]
        scaled_img = None
        # depth_rmse = self.rmse(pred_depth,target_depth)
        # rmse_log_err = self.rmse_log(pred_depth,target_depth)
        # abs_rel_error= self.rel_error(pred_depth,target_depth,squared = False)
        # sq_rel_err = self.rel_error(pred_depth,target_depth, squared = True)
        # accs = self.acc(pred_depth, target_depth)
        silog = self.silog(pred_depth, target_depth)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(target_depth.cpu().numpy(),
                                                                     pred_depth.cpu().numpy())
        metric_dict = {}
        metric_dict.update({
            f"{prefix}_acc0": a3,
            f"{prefix}_acc1": a2,
            f"{prefix}_acc2": a1,
            f"{prefix}_rmse": rmse,
            f"{prefix}_rel_err": abs_rel,
            f"{prefix}_sq_err": sq_rel,
            f"{prefix}_rmse_log": rmse_log,
            f"{prefix}_silog": silog
        })
        return metric_dict, scaled_img
