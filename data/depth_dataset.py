import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
import pandas as pd
from utils.exr_utils import exr2numpy
import numpy as np
import torch
from utils.image_utils import create_circular_mask
import random
import torchvision.transforms.functional as TF


class DepthDataset(Dataset):
    def __init__(self,
                 annotations_path,
                 mode: str,
                 gan_data_dir: str,
                 transform=None,
                 normalize=None,
                 img_size=(256, 256),
                 stage="train"):
        self.mode = mode
        self.transform = transform
        self.normalize = normalize
        annotations = pd.read_csv(annotations_path)
        self.img_size = img_size
        self.transform_to_float_tensor = transforms.ConvertImageDtype(torch.float)
        self.mask2d = create_circular_mask(img_size[0], img_size[1])
        self.annotations = annotations[annotations["Set"] == stage]
        self.real_image_data_dir = os.path.join(gan_data_dir, stage)

    def __len__(self):
        if self.mode == "gan":
            return len(os.listdir(self.real_image_data_dir))
        else:
            return len(self.annotations.index)

    def __getitem__(self, idx):
        if self.mode == "gan":
            real_idx = idx
            synth_idx = np.random.randint(0, len(self.annotations.index))
        else:
            synth_idx = idx
            real_idx = np.random.randint(0, len(os.listdir(self.real_image_data_dir)))

        synth_img_path = self.annotations[self.annotations["Idx"] == synth_idx]["Path"].values[0]
        synth_image = read_image(synth_img_path + ".png", ImageReadMode.RGB)
        synth_image = self.transform_to_float_tensor(synth_image)
        real_img_path = os.path.join(self.real_image_data_dir, os.listdir(self.real_image_data_dir)[real_idx])
        real_image = read_image(real_img_path, ImageReadMode.RGB)
        real_image = self.transform_to_float_tensor(real_image)
        label = torch.Tensor(exr2numpy(synth_img_path + ".exr", normalize=False))
        label = torch.unsqueeze(self.transform_to_float_tensor(label), dim=0)

        if self.transform and random.random() > 0.5:
            synth_image = self.transform(synth_image)
            real_image = self.transform(real_image)
            real_image = self.single_transform(real_image)

        synth_image, label = self.shared_transform(synth_image, label)

        if self.normalize:
            synth_image = self.normalize(synth_image)
            real_image = self.normalize(real_image)

        real_labels = []
        return synth_image, real_image, label, real_labels

    def shared_transform(self, image, label):
        border_color = torch.rand((3, 1), dtype=torch.float) / 10
        image[:, self.mask2d] = border_color
        label[:, self.mask2d] = 0

        # Random affine
        angle, translate, scale, shear = transforms.RandomAffine([0, 360]).get_params([0, 360],
                                                                                      [0.1, 0.1],
                                                                                      None,
                                                                                      None,
                                                                                      self.img_size)
        image = TF.affine(image, angle, translate, scale, shear, fill=torch.flatten(border_color).tolist())
        label = TF.affine(label, angle, translate, scale, shear, fill=0)
        return image, label

    def single_transform(self, image):
        # Random affine
        angle, translate, scale, shear = transforms.RandomAffine([0, 360]).get_params([0, 360],
                                                                                      [0.1, 0.1],
                                                                                      None,
                                                                                      None,
                                                                                      image.size())
        image = TF.affine(image, angle, translate, scale, shear, fill=0)
        return image
