import os.path
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import re
import json
from data.image_dataset import ImageDataset

_mac_regex = re.compile(r'^(?!.*\._)')


class FileLoadingDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 directories: dict,
                 split: dict = None,
                 workers_per_loader: int = 6,
                 exclude_regex: str = None):
        """
         A Data Module for loading paired files located in different directories. See the split parameter for
         thoughts on how best to set up your data structure for use with this module.

        :param batch_size: batch sized to use for training
        :param directories: a dictionary with each key-pair as the data role and directory, respectively.\
                            i.e. {'img': /my/path, 'label': /my/other/path}

        :param split: can be one of the following:
                     - dictionary of regex strings to filter the filenames with:
                       {'train': ".*train.*", 'validate': ".*val.*", 'test': ".*test.*"} <- i.e. if divided into subfolders
                     - dictionary of floats specifying the random data split:
                       {'train': 0.5, 'validate': 0.3, 'test': 0.2}
                     - A combination of the two previous options:
                       {'train': 0.75, 'validate': 0.25, 'test': ".*model_01.*"}
                       The files for any regex matches will be assigned first and the remainder will respect the float splits
                     - string pointing to a file from a previously saved split. Needs to be a json file with a dictionary
                       of the form json_dict[stage][role] = List[files], where stage is one of ['train', 'validate', 'test']
                       and role matches the keys provided in `directories`.
                     split defaults to the first example assuming subfolders with "train", "val", and "test"

        :param workers_per_loader: cpu threads to use for each data loader.
        :param exclude_regex: regex for excluding files.
        """
        super().__init__()
        self.workers_per_loader = workers_per_loader
        self.batch_size = batch_size
        if split is None:
            split = {'train': ".*train.*", 'validate': ".*val.*", 'test': ".*test.*"}

        if isinstance(split, str):
            with open(split, 'r') as f:
                self.split_files = json.load(f)
        else:
            image_files = {
                key: [str(f) for f in Path(val).rglob('*') if _mac_regex.search(str(f)) and os.path.isfile(f)]
                for key, val in directories.items()
            }
            if exclude_regex is not None:
                exclude_regex = re.compile(exclude_regex)
                [image_files.update({key: [f for f in image_files[key] if exclude_regex.search(f)]}) for key in
                 image_files]
            if len(list(image_files.keys())) > 1:
                list_lengths = set([len(image_files[key]) for key in image_files])
                assert len(list_lengths) == 1, f'Different number of files found between data roles.'

            [image_files[key].sort() for key in image_files]
            self.split_files = {'test': {}, 'validate': {}, 'train': {}}
            stages = list(self.split_files.keys())
            for stage in stages:
                if isinstance(split[stage], str):
                    test_re = re.compile(split[stage])
                    for key, file_list in image_files.items():
                        self.split_files[stage][key] = [s for s in file_list if test_re.search(s)]
                        for f in self.split_files[stage][key]:
                            file_list.remove(f)
            first_role = list(image_files.keys())[0]
            indices = np.random.permutation(np.linspace(0, len(image_files[first_role]) - 1,
                                                        len(image_files[first_role]))).astype(int)
            remaining_count = len(indices)
            for stage in stages:
                if isinstance(split[stage], float):
                    stage_indices = indices[:int(split[stage] * remaining_count) + 1]
                    for key, file_list in image_files.items():
                        self.split_files[stage][key] = np.asarray(file_list)[stage_indices]
                    indices = indices[len(stage_indices):]

        self.data_train: ImageDataset = None
        self.data_val: ImageDataset = None
        self.data_test: ImageDataset = None

    def save_split(self, file_name: str):
        """
        Write the generated split to a file. TODO: deal with relative/absolute paths for portability

        :param file_name: the name of the file to write the split to.
        """
        if file_name[-4:] != 'json':
            file_name = f'{file_name}.json'
        with open(file_name, 'w') as f:
            json.dump(self.split_files, f)

    def setup(self, stage: str = None):
        """
        setup must be overridden and set the datasets self.data_train, self.data_val, and self.data_test.
        The files to provide to the datasets can be accessed from self.split_files.
        """
        raise NotImplementedError()

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
