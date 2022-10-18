import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.torch_utils import generateKittiAnnotations
from data.kitti_dataset import KittiDataset
import multiprocessing


class KittiDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, annotations_path, mode,  dataset_dir):
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.annotations_path = annotations_path
        self.kitti_annotations_path = os.path.join(self.annotations_path, "kitti_annotations.csv")
        self.annotations_path = annotations_path
        self.gan_data_dir = os.path.join(dataset_dir, "kitti")
        self.synthetic_data_dir = os.path.join(dataset_dir, "kitti_synth")
        self.mode = mode
        self.eigen_split: np.ndarray = None

    def prepare_data(self):
        """ Generate annotations if they don't exist yet.
        This function is only called by one process used for training """
        if not os.path.exists(self.kitti_annotations_path):
            generateKittiAnnotations(self.kitti_annotations_path,
                                     self.annotations_path,
                                     self.synthetic_data_dir,
                                     self.gan_data_dir)

    def setup(self, stage: str = None):
        """ This function will be called for each process used for training """
        with np.load(os.path.join(self.dataset_dir, "kitti_depthmaps", "gt_depths.npz"), allow_pickle=True) as data:
            self.eigen_split = np.asarray(data['data'])
        normalize = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        ])
        
        if self.mode == "gan":
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])  
        else:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])     
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = KittiDataset(self.kitti_annotations_path,
                                           self.mode,
                                           self.gan_data_dir,
                                           transform=transform,
                                           normalize=normalize,
                                           stage="train")
            self.data_val = KittiDataset(self.kitti_annotations_path,
                                         self.mode,
                                         self.gan_data_dir,
                                         normalize=normalize,
                                         stage="val")
            # self.dims = tuple(self.mnist_train[0][0].shape)
        if stage == "validate":
            self.data_val = KittiDataset(self.kitti_annotations_path,
                                         self.mode,
                                         self.gan_data_dir,
                                         normalize=normalize,
                                         stage="val")

        if stage == "test" or stage is None:
             self.data_test = KittiDataset(self.kitti_annotations_path,
                                           self.mode,
                                           self.gan_data_dir,
                                           normalize=normalize,
                                           stage="test",
                                           depthmaps=self.eigen_split)
            # self.dims = tuple(self.mnist_test[0][0].shape)    

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=min(6, multiprocessing.cpu_count()//2),
                          shuffle=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          num_workers=min(6, multiprocessing.cpu_count()//2),
                          shuffle=False,
                          pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          num_workers=min(6, multiprocessing.cpu_count()//2),
                          shuffle=False,
                          pin_memory=False)


