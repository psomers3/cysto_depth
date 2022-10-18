import os
from pathlib import Path

import pytorch_lightning as pl
from torchvision import transforms

import torch
from data.gan_dataset import GANDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
from utils.exr_utils import extract_frames
import ast


class GANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        gan_target_data_dir,
        annotations_path,
        gan_source_data_dir: str,
        batch_size: int,
        num_workers: int,
        generate_data = False
    ):
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.gan_source_data_dir = gan_source_data_dir
        self.gan_target_data_dir = gan_target_data_dir
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generate_data = generate_data

        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])    

    def prepare_data(self): 
        if self.generate_data:
            vid_annotations_path = os.path.join(self.gan_source_data_dir, "video_annotations.csv")
            vid_annotations = pd.read_csv(vid_annotations_path)
            testset_path = os.path.join(self.gan_target_data_dir, "test")
            trainset_path = os.path.join(self.gan_target_data_dir, "train")
            valset_path = os.path.join(self.gan_target_data_dir, "val")
            failed_cropping_path = os.path.join(self.gan_target_data_dir, "failed_cropping")
            Path(testset_path).mkdir(parents=True,exist_ok=True)
            Path(trainset_path).mkdir(parents=True,exist_ok=True)
            Path(valset_path).mkdir(parents=True,exist_ok=True)
            Path(failed_cropping_path).mkdir(parents=True,exist_ok=True)
            for _, video in vid_annotations.iterrows():
                print("Extracting Frames from Video {}".format(video["Title"]))
                target_dir = testset_path if video["Type"] == "Test" else trainset_path if video["Type"] == "Train" else valset_path
                extract_frames(self.gan_source_data_dir, target_dir, video["Title"], ast.literal_eval(video["Scenes"]), 256, failed_cropping_path)
            print("Finished Processing Video Frames")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = GANDataset(self.gan_target_data_dir, self.annotations_path,mode="train", transform=self.transform)
            self.val = GANDataset(self.gan_target_data_dir, self.annotations_path,mode = "val", transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = GANDataset(self.gan_target_data_dir, self.annotations_path, mode="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)