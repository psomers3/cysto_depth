from torch import nn
import torch
import torch.nn.functional as F
from utils.torch_utils import scale_median
import numpy as np


class RMSE(nn.Module):
    def forward(self, predicted,target):
        return torch.sqrt(F.mse_loss(predicted,target))
        
class RMSELog(nn.Module):
    def forward(self, predicted,target):
        target_depth_log = torch.log(torch.clamp(target,min = 1e-3)) # lowest possible prediction is zero, log10(1)=0
        predicted_depth_log = torch.log(torch.clamp(predicted,min = 1e-3)) # lowest possible prediction is zero, log10(1)=0
        return torch.sqrt(F.mse_loss(predicted_depth_log,target_depth_log
                                     ))
    
class Accuracy(nn.Module):
    def __init__(self, thresholds = torch.Tensor([1.25,1.25**2,1.25**3])):
        self.thresholds = thresholds
        super().__init__()
        
        
    def forward(self,predicted,target):
        n_depth = predicted.size(0)
        acc_thresholds = self.thresholds.type_as(predicted).expand(n_depth,3)
        # calculate max of relatives, replace nan with inf (which is always above threshold)
        max_rel = torch.max(predicted/target,torch.nan_to_num(target/predicted,posinf=True))                 
        accuracies = torch.sum((max_rel < acc_thresholds).float(), dim=1) / n_depth
        return accuracies
    
class RelError(nn.Module):
    def forward(self,predicted,target,squared = False):
        exp  = 1
        if squared:
            exp = 2
        return torch.mean(torch.pow(torch.abs(target-predicted),exp)/torch.abs(target))
    
class SILog(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    
    def forward(self,predicted,target):
        target_depth_log = torch.log(torch.clamp(target,min = 1)) # lowest possible prediction is zero, log10(1)=0
        predicted_depth_log = torch.log(torch.clamp(predicted,min = 1)) # lowest possible prediction is zero, log10(1)=0
        d = predicted_depth_log - target_depth_log
        s1 = torch.mean(d**2)
        s2 = torch.mean(d)**2
        return torch.sqrt(s1-s2)*100
    
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt = np.clip(gt, 1e-3, None)
    pred = np.clip(pred,1e-3,None)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
        

        
        
        
        