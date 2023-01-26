import os
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from utils.exr_utils import extract_frames
import ast
from torchvision import transforms as torch_transforms
from data.image_dataset import ImageDataset
import data.data_transforms as d_transforms
from data.general_data_module import FileLoadingDataModule, _mac_regex
import re
import csv
import json
from typing import *
from omegaconf import ListConfig

_video_types = ['.mpg', '.mp4']
_original_exclusion = re.compile(r'^(?!.*original)')
_failed_exclusion = r'^(?!.*failed)'
_annotations_csv = re.compile(r'video_annotations')


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.length = int(min([len(d) for d in self.datasets]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> List[torch.Tensor]:
        output = []
        for d in self.datasets:
            output.extend(d[idx])
        return output


class GANDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 color_image_directories: Union[str, List[str]],
                 video_directories: Union[str, List[str]],
                 generate_output_directory: str,
                 generate_data: bool = False,
                 synth_split: dict = None,
                 image_size: int = 256,
                 workers_per_loader: int = 6):
        """ A Data Module for loading rendered endoscopic images and real images. The rendered images will be made
         square and a circular mask applied to simulate actual endoscopic images. The real images are generated
         from endoscopic videos.
        TODO: set up to use a configuration for control of all transforms and other hyperparams.

        :param batch_size: batch sized to use for training.
        :param color_image_directories: path/s to the color images. All will be searched recursively.
        :param video_directories: path/s to the videos. All will be searched recursively. An optional
                                "video_annotations.csv" file may exist identifying only scenes of interest
                                within the videos. It should be of the format:
                                Title          Type                Scenes
                                ________________________________________________
                                GRK08.mpg      train       [(4150, 5000), 7000]

                                The Scenes are frame numbers. pairs of frames include all in between. Type can be
                                train, val, or test
        :param generate_output_directory: path to the directory where the extracted video frames are put.
        :param generate_data: whether to process the videos before starting training. Should only need to do this once.
        :param synth_split: see parent class FileLoadingDataModule. This affects only the synthetic data split.
        :param image_size: final `square` image size to return for training.
        :param workers_per_loader: cpu threads to use for each data loader.
        """
        super(GANDataModule, self).__init__()
        directories = {'synth': color_image_directories, 'real': os.path.join(generate_output_directory)}
        self.output_directories = {'train': os.path.join(generate_output_directory, 'train'),
                                   'val': os.path.join(generate_output_directory, 'validate'),
                                   'test': os.path.join(generate_output_directory, 'test'),
                                   'failed': os.path.join(generate_output_directory, 'failed_cropping')}
        [os.makedirs(directory, exist_ok=True) for directory in self.output_directories.values()]
        self.batch_size = batch_size
        self.directories = directories
        self.workers_per_loader = workers_per_loader
        self.synth_split = synth_split
        self.save_hyperparameters("batch_size")
        self.image_size = image_size
        self.video_directories = video_directories
        self.generate_data = generate_data
        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def save_split(self, file_name: str):
        """
        Write the generated split to a file. TODO: deal with relative/absolute paths for portability

        :param file_name: the name of the file to write the split to.
        """
        if file_name[-4:] != 'json':
            file_name = f'{file_name}.json'
        split_files = self.synth_split if isinstance(self.synth_split, dict) else dict(self.synth_split)
        with open(file_name, 'w') as f:
            json.dump(split_files, f)

    def prepare_data(self) -> None:
        if self.generate_data:
            annotations = []
            if isinstance(self.video_directories, (list, ListConfig)):
                [annotations.extend([str(f) for f in Path(p).rglob('*') if _annotations_csv.search(str(f))][0]) \
                 for p in self.video_directories]
            else:
                annotations = [str(f) for f in Path(self.video_directories).rglob('*') if _annotations_csv.search(str(f))][0]
            scenes = {}
            with open(annotations, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['Title'].lower() not in scenes:
                        scenes[row['Title'].lower()] = {'train': None, 'val': None, 'test': None}
                    scenes[row['Title'].lower()][row['Type'].lower()] = ast.literal_eval(row["Scenes"])
            videos = [str(f) for f in Path(self.video_directories).rglob('*') if
                      _mac_regex.search(str(f)) and
                      _original_exclusion.search(str(f)) and
                      (os.path.splitext(f)[-1].lower() in _video_types)]
            manually_labeled_videos = []
            for video in videos:
                # TODO: This will fail if subfolders contain videos with the same name
                video_key = os.path.basename(video).lower()
                if video_key in scenes:
                    manually_labeled_videos.append(video)
                    for mode, scene_list in scenes[video_key].items():
                        extract_frames(video,
                                       self.output_directories[mode],
                                       scene_list,
                                       self.image_size,
                                       self.output_directories['failed'])
            [videos.remove(v) for v in manually_labeled_videos]
            # TODO: Handle videos not in the annotations csv

    def setup(self, stage: str = None):
        # NOTE!! if any SynchronizeTransforms are used, then each dataset needs its own set of transforms. See the
        # EndoDepthDataModule
        squarify = d_transforms.Squarify(image_size=self.image_size)
        mask = d_transforms.EndoMask(radius_factor=[0.9, 1.0])
        imagenet_norm = d_transforms.ImageNetNormalization()
        affine_transform = d_transforms.RandomAffine(degrees=(0, 360), translate=(.05, .05), use_corner_as_fill=True)
        synth_transforms = torch_transforms.Compose([mask, squarify, affine_transform, imagenet_norm])
        real_transforms = torch_transforms.Compose([squarify, affine_transform, imagenet_norm])

        real_split = FileLoadingDataModule.create_file_split({'real': self.directories['real']},
                                                             exclusion_regex=_failed_exclusion)
        synth_split = FileLoadingDataModule.create_file_split({'synth': self.directories['synth']},
                                                              split=self.synth_split,
                                                              exclusion_regex=_failed_exclusion)
        keys = ['train', 'validate', 'test']
        real, synth = [], []
        for key in keys:
            real.append(ImageDataset(files=real_split[key]['real'],
                                     transforms=real_transforms if key == 'train' else None,
                                     randomize=True))
            synth.append(ImageDataset(files=synth_split[key]['synth'],
                                      transforms=synth_transforms if key == 'train' else None,
                                      randomize=True))

        self.data_train = ConcatDataset([synth[0], real[0]])
        self.data_val = ConcatDataset([synth[1], real[1]])
        self.data_test = ConcatDataset([synth[2], real[2]])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=self.workers_per_loader,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          num_workers=self.workers_per_loader,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          num_workers=self.workers_per_loader,
                          shuffle=False,
                          pin_memory=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.image_utils import matplotlib_show

    color_dir = r'/Users/peter/isys/output/color'
    real_output = r'../gan_data'
    video_dir = r'/Users/peter/isys/videos'
    dm = GANDataModule(batch_size=3,
                       color_image_directory=color_dir,
                       generate_output_directory=real_output,
                       generate_data=False,
                       video_directory=video_dir,
                       synth_split={'train': .6, 'validate': 0.4, 'test': ".*00015.*"})
    dm.prepare_data()
    dm.setup('fit')
    loader = dm.train_dataloader()
    sample = next(iter(loader))
    matplotlib_show(*sample)
    plt.show(block=True)
