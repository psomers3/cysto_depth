#! /usr/bin/env python

import sys
import os

sys.path.append(os.path.join('..', os.path.dirname(__file__)))
from pathlib import Path
from argparse import ArgumentParser
from utils.exr_utils import exr_2_numpy
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    args = parser.parse_args()

    directory = Path(args.directory)
    files = [d for d in directory.rglob('*') if d.is_file() and d.suffix.lower() == '.exr']
    for file in tqdm(files):
        depth_map = exr_2_numpy(str(file))
        if depth_map.max() >= 1000:
            print(f'deleting: {file}')
            file.unlink()

