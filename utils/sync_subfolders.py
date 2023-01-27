#! /usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser
import numpy as np


def numpy_combinations(x):
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)

    return x[idx]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    args = parser.parse_args()

    directory = Path(args.directory)
    sub_directories = [d for d in directory.glob('*') if d.is_dir()]
    files = {}
    for sub_dir in sub_directories:
        files[sub_dir] = set([str(f.relative_to(sub_dir))[:-4] for f in sub_dir.rglob('*') if f.is_file() and f.name[0] != '.'])

    combos = numpy_combinations(np.linspace(0, stop=len(files.keys())-1, num=len(files.keys()))).astype(int)
    differences = [list(files[sub_directories[c[0]]] - files[sub_directories[c[1]]]) for c in combos]
    differences.extend([list(files[sub_directories[c[1]]] - files[sub_directories[c[0]]]) for c in combos])
    diffs = []
    [diffs.extend(d) for d in differences]
    differences = set(diffs)
    for sub_dir in sub_directories:
        for diff in differences:
            if diff in files[sub_dir]:
                next(Path(str(Path(directory, sub_dir))).rglob(f'{diff}*')).unlink()
