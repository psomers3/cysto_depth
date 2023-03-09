from pathlib import Path
import os
import re

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from collections import defaultdict
from tensorflow.python.framework import tensor_util
from typing import *


def get_scalars_from_events(log_directory: str, scalar_tags: Union[str, List[str]]) \
        -> Union[Tuple[str, Dict[str, np.ndarray]], List[Tuple[str, Dict[str, np.ndarray]]]]:
    """
    extracts the values for scalar values recorded in tensorboard.

    :param log_directory: path to directory where log files are
    :param scalar_tags: list of tags to look for
    :return: list of tuples of form (tag, dictionary), where the dictionary contains the values, steps, and wall times
             under different keys of the form: tag, tag_step, tag_t
    """
    regex_compiled = re.compile('.*tfevents.*')
    event_files = [p for p in Path(log_directory).rglob("*") if regex_compiled.search(str(p))]
    event_files.sort()
    scalar_tags = [scalar_tags] if not isinstance(scalar_tags, list) else scalar_tags
    data_to_plot = [defaultdict(list) for _ in scalar_tags]
    for file in event_files:
        subdir = os.path.split(os.path.dirname(file))[-1]
        for e in EventFileLoader(str(file)).Load():
            for v in e.summary.value:
                for i, _tag in enumerate(scalar_tags):
                    if (_tag in v.tag) and (_tag in subdir):
                        data_to_plot[i][subdir].append(tensor_util.MakeNdarray(v.tensor))
                        data_to_plot[i][f'{subdir}_step'].append(e.step)
                        data_to_plot[i][f'{subdir}_t'].append(e.wall_time)
                    elif _tag in v.tag:
                        data_to_plot[i][v.tag].append(tensor_util.MakeNdarray(v.tensor))
                        data_to_plot[i][f'{v.tag}_step'].append(e.step)
                        data_to_plot[i][f'{v.tag}_t'].append(e.wall_time)

    data_to_return = [{} for _ in scalar_tags]
    for i in range(len(scalar_tags)):
        for key in data_to_plot[i]:
            data_to_return[i][key] = np.asarray(data_to_plot[i][key])

    return [(scalar_tags[i], data_to_return[i]) for i in range(len(scalar_tags))]
