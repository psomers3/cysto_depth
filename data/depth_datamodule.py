import ast
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.exr_utils import extract_frames
from utils.torch_utils import generateImageAnnotations

from data.depth_dataset import DepthDataset


class DepthDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, annotations_path, mode,  dataset_dir,  generate_data = False):
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.annotations_path = annotations_path
        self.synthetic_data_annotations_path = os.path.join(self.annotations_path, "bladder_annotations.csv")
        self.generate_data = generate_data
        self.gan_data_dir = os.path.join(dataset_dir, "gan_data")
        self.synthetic_data_dir = os.path.join(dataset_dir, "depth_data_DefaultMaterial_DefaultParticleMaterial_par_dis")
        self.synthetic_data_test_dir = os.path.join(dataset_dir, "depth_data_BlankMaterial_BlankMaterial_par_dis")
        self.mode = mode

        self.data_train: DepthDataset = None
        self.data_val: DepthDataset = None
        self.data_test: DepthDataset = None

    def prepare_data(self):
        if not os.path.exists(self.synthetic_data_annotations_path):
            generateImageAnnotations(self.synthetic_data_annotations_path,
                                     self.synthetic_data_dir,
                                     self.synthetic_data_test_dir)
        if self.generate_data:
            vid_annotations_path = os.path.join(self.annotations_path, "video_annotations.csv")
            vid_annotations = pd.read_csv(vid_annotations_path)
            failed_cropping_folder_name = "failed_cropping"
            folders = np.append(vid_annotations["Type"].unique(), failed_cropping_folder_name)
            for folder in folders: 
                path = os.path.join(self.gan_data_dir, folder)
                Path(path).mkdir(parents=True, exist_ok=True)
            failed_cropping_path = os.path.join(self.gan_data_dir, failed_cropping_folder_name)
            gan_source_path = os.path.join(self.gan_data_dir, "source")
            for _, video in vid_annotations.iterrows():
                print("Extracting Frames from Video {}".format(video["Title"]))
                target_dir = os.path.join(self.gan_data_dir, video["Type"])
                extract_frames(gan_source_path,
                               target_dir,
                               video["Title"],
                               ast.literal_eval(video["Scenes"]),
                               256,
                               failed_cropping_path)
            print("Finished Processing Video Frames")

    def setup(self, stage: str = None):
        normalize = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        ])
        
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])          
            
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = DepthDataset(self.synthetic_data_annotations_path,
                                           self.mode,
                                           self.gan_data_dir,
                                           transform=transform,
                                           normalize=normalize,
                                           stage="train")
            self.data_val = DepthDataset(self.synthetic_data_annotations_path,
                                         self.mode,
                                         self.gan_data_dir,
                                         normalize=normalize,
                                         stage="val")
            # self.dims = tuple(self.mnist_train[0][0].shape)
        if stage == "validate":
            self.data_val = DepthDataset(self.synthetic_data_annotations_path,
                                         self.mode,
                                         self.gan_data_dir,
                                         normalize=normalize,
                                         stage="val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = DepthDataset(self.synthetic_data_annotations_path,
                                          self.mode,
                                          self.gan_data_dir,
                                          normalize=normalize,
                                          stage="test")
            # self.dims = tuple(self.mnist_test[0][0].shape)    

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=6, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=6, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=6, shuffle=False, pin_memory=True)


