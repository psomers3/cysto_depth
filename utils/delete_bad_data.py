#! /usr/bin/env python
from data.depth_datamodule import EndoDepthDataModule
from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    args = parser.parse_args()

    color_dir = Path(args.directory, 'color')
    depth_dir = Path(args.directory, 'depth')
    normals_dir = Path(args.directory, 'normals')

    dm = EndoDepthDataModule(batch_size=3,
                             data_roles=['color', 'depth'],
                             data_directories=[str(color_dir), str(depth_dir)],
                             split={'train': .6, 'validate': 0.3, 'test': .1},
                             inverse_depth=True,
                             memorize_check=False)
    dm.setup('fit')
    loader = iter(dm.train_dataloader())
    for sample in loader:
        pass

