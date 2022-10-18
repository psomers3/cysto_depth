import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
import pandas as pd
import numpy as np
import torch    
from PIL import Image
import torch.nn as nn
import random
import torchvision.transforms.functional as TF


class KittiDataset(Dataset):
    def __init__(self, annotations_path, mode, gan_data_dir, transform=None, normalize=None, img_size=[256, 256], stage="train", depthmaps=None):
        self.mode = mode
        self.transform = transform
        self.normalize = normalize
        annotations = pd.read_csv(annotations_path)
        self.img_size = img_size
        self.stage = stage
        self.depthmaps = depthmaps
        self.transform_to_float_tensor = transforms.ConvertImageDtype(torch.float)
        annotations = annotations[annotations["Set"] == stage]
        self.synth_annotations = annotations[annotations["Type"] == "synth"]
        self.real_annotations = annotations[annotations["Type"] == "real"]
        self.real_image_data_dir = os.path.join(gan_data_dir, stage)
        self.pil_to_tensor = transforms.PILToTensor()
        self.resize = nn.Sequential(
            transforms.Resize([256, 512]),  # We use single int value inside a list due to torchscript type restrictions
        )
        self.full_res_shape = (1242, 375)

    def __len__(self):
        if self.mode == "gan":
            return len(self.real_annotations.index) 
        else:
            return len(self.synth_annotations.index)

    def __getitem__(self, idx):
        if self.mode == "gan":
            real_idx = idx
            synth_idx = np.random.randint(0, len(self.synth_annotations.index))
        else:
            synth_idx = idx
            real_idx = np.random.randint(0,len(self.real_annotations.index))
        synth_img_path, synth_depth_path = self.synth_annotations[self.synth_annotations["Idx"] == synth_idx][["Path", "Path_Depth"]].values[0]
        real_img_path, real_depth_path = self.real_annotations[self.real_annotations["Idx"] == real_idx][["Path", "Path_Depth"]].values[0]
        synth_image = read_image(f'.{synth_img_path}', ImageReadMode.RGB)
        synth_image = self.transform_to_float_tensor(synth_image)
        # if self.stage == "test":
        #     real_label = self.depthmaps[real_idx]
        #     real_label = torch.unsqueeze(torch.from_numpy(real_label),dim = 0)
        #else:
        real_label = Image.open(f'.{real_depth_path}')
        real_label = real_label.resize(self.full_res_shape, Image.NEAREST)
        real_label = torch.div(self.pil_to_tensor(real_label),256) # to meters
            
        real_image = read_image(f'.{real_img_path}', ImageReadMode.RGB)
        
        real_image = self.transform_to_float_tensor(real_image)
        synth_label = Image.open(f'.{synth_depth_path}')
        
        synth_label = torch.div(self.pil_to_tensor(synth_label),100) # to meters
        # synth_label[synth_label == 655.35] = 0 # treat far plane as zero
        synth_image = self.resize(synth_image)
        synth_label = self.resize(synth_label)
        real_image = self.resize(real_image)
        
        if self.transform and random.random() > 0.5:
            synth_image = self.transform(synth_image)
            real_image = self.transform(real_image)
        if self.stage == "train" and random.random() > 0.5:
            synth_image = TF.hflip(synth_image)
            synth_label = TF.hflip(synth_label)
            real_image = TF.hflip(real_image)
            real_label = TF.hflip(real_label)
        if self.normalize:
            synth_image = self.normalize(synth_image)
            real_image = self.normalize(real_image)
        return synth_image, real_image, synth_label, real_label