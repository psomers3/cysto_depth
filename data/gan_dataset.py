from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import pandas as pd
import os
import numpy as np
import torch
from utils.image_utils import create_circular_mask


class GANDataset(Dataset):
    def __init__(self, img_dir,annotations_path, transform=None, mode="train"):
        self.transform = transform
        self.mask2d = create_circular_mask(256,256)
        annotations = pd.read_csv(annotations_path)
        if mode=="train":
           self.img_dir = os.path.join(img_dir,"train")
           self.annotations = annotations[annotations["Set"]== "Train"]
        elif mode=="test":
            self.img_dir = os.path.join(img_dir,"test")
            self.annotations = annotations[annotations["Set"]== "Test"]
        else:
            self.img_dir = os.path.join(img_dir,"val")
            self.annotations = annotations[annotations["Set"]== "Train"]
        


    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # get input image
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = read_image(img_path, ImageReadMode.RGB)
        # get target by sampling random synthetic image and encoding it
        idx_synthetic = np.random.randint(0,len(self.annotations.index))
        synthetic_img_path = self.annotations[self.annotations["Idx"]==idx_synthetic]["Path"].values[0] + ".png"
        synthetic_image = read_image(synthetic_img_path, ImageReadMode.RGB)
        border_color =  torch.randint(0, 20, (3,1), dtype=torch.uint8)
        synthetic_image[:,self.mask2d] = border_color
        if self.transform:
            image = self.transform(image)
            synthetic_image = self.transform(synthetic_image)
        return synthetic_image, image, idx