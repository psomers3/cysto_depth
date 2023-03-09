#! /usr/bin/env python
from argparse import ArgumentParser
import os
from typing import *
from pathlib import Path
from tensorboard_parsing import get_scalars_from_events
import numpy as np
import matplotlib.pyplot as plt
import pandas
plt.style.use(Path(Path(__file__).parent, f'paper_style.mplstyle'))
plt.rcParams.update()

# Paper sizes
single_column_width = 3.25  # inches
double_column_width = 7.5  # inches
height_per_plot = 3
inches2cm = 2.54
cm2inches = 1/inches2cm

# corporate design
anthracit = [0.2431, 0.2667, 0.2980]
light_blue = [0.0000, 0.7451, 1.0000]
dark_blue = [0.0000, 0.3176, 0.6196]
light_grey = [0.6235, 0.6000, 0.5961]
yellow = [1.0000, 0.8353, 0.0000]
colors = [light_blue, dark_blue, light_grey, yellow, anthracit]


class PlotData(TypedDict):
    fig: plt.Figure
    ax: plt.Axes


def smooth(data:np.ndarray, tensorboard_factor: float = 0.6) -> np.ndarray:
    """ apply exponential weighted MA """
    # alpha = 1 - tensorboard_factor
    # alpha_rev = 1-alpha
    # n = data.shape[0]
    # pows = alpha_rev**(np.arange(n+1))
    # scale_arr = 1/pows[:-1]
    # offset = data[0]*pows[1:]
    # pw0 = alpha*alpha_rev**(n-1)
    # mult = data*pw0*scale_arr
    # cumsums = mult.cumsum()
    # out = offset + cumsums*scale_arr[::-1]
    df = pandas.DataFrame(data=data)
    return df.ewm(alpha=(1 - tensorboard_factor)).mean().to_numpy()


def get_plots(log_directories, tags, width, log) -> Dict[str, PlotData]:
    return_plots = {}
    data = [get_scalars_from_events(directory, tags) for directory in log_directories]
    for tag in tags:
        return_plots[tag] = {}
        fig = plt.figure(figsize=(width * cm2inches, height_per_plot))
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        return_plots[tag]['fig'] = fig
        return_plots[tag]['ax'] = ax
        # Legend outside
        # percent_offset = 0.02
        # desired_width = fig.get_size_inches()[0]
        fig.tight_layout(h_pad=0, w_pad=0, pad=0)  # tight margins
        # dpi = fig.dpi

        for i, run in enumerate(data):
            for scalar_tup in run:
                if isinstance(scalar_tup, str):
                    raise ValueError(f'WARNING: No data found for {scalar_tup}')
                if scalar_tup[0] == tag:
                    try:
                        if tag in scalar_tup[1]:
                            y = scalar_tup[1][tag]
                            x = scalar_tup[1][f'{tag}_step']
                            ax.plot(x, y, label=f'{tag}_{i}', color=colors[i])
                        else:
                            sub_tags = [key for key in scalar_tup[1] if ('_step' not in key and '_t' not in key)]
                            for j, sub_tag in enumerate(sub_tags):
                                y = scalar_tup[1][sub_tag]
                                x = scalar_tup[1][f'{sub_tag}_step']
                                ax.plot(x, y, label=f'{sub_tag}_{i}', color=colors[j], linestyle=linestyle_str[i])
                    except ValueError as e:
                        print(e)
                        print('ERROR occured for directory:', i, tag)

        ax.set_position((0, 0, ax.get_position().width, ax.get_position().height))
        if log:
            plt.yscale('log')
        legend = ax.legend(loc='best')
        return_plots[tag]['legend'] = legend
        plt.draw()
    return return_plots


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_dirs', nargs='+', default=[])
    parser.add_argument('--tags', nargs='+', default=[])
    parser.add_argument('--width', type=float, default=11.0, help='plot width in cm')
    parser.add_argument('--output', default='.', type=str, help='path to folder for output')
    parser.add_argument('--log', action='store_true', help='use log y-axis')
    parser.add_argument('--smoothing', default=0.0, type=float, help='if >0 -> apply tensorboard smoothing')

    args = parser.parse_args()
    tags: List[str] = args.tags
    directories: List[str] = args.log_dirs
    [exit(1) for directory in directories if not os.path.isdir(directory)]
    linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']

    for tag, items in get_plots(directories, tags, args.width, args.log).items():
        if 0 < args.smoothing < 1:
            new_lines = []
            for line in items['ax'].lines:
                line: plt.Line2D
                x = line.get_xdata(orig=False)
                y = line.get_ydata(orig=False)
                line.set_alpha(0.2)
                smoothed = smooth(y, args.smoothing)
                new_lines.append(plt.Line2D(x, smoothed, linewidth=line.get_linewidth(),
                                            linestyle=line.get_linestyle(), color=line.get_color()))
            [items['ax'].add_line(nline) for nline in new_lines]
            plt.draw()
        items['fig'].savefig(os.path.join(args.output, f'{tag}.pdf'), bbox_inches='tight', pad_inches=0.0, dpi='figure')