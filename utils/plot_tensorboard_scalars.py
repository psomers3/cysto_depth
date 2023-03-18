#! /usr/bin/env python
from argparse import ArgumentParser
import os
from typing import *
from pathlib import Path
from tensorboard_parsing import get_scalars_from_events
import numpy as np
import matplotlib.pyplot as plt
import pandas

inches2cm = 2.54
cm2inches = 1/inches2cm

# corporate design
anthracit = [0.2431, 0.2667, 0.2980]
light_blue = [0.0000, 0.7451, 1.0000]
dark_blue = [0.0000, 0.3176, 0.6196]
light_grey = [0.6235, 0.6000, 0.5961]
yellow = [1.0000, 0.8353, 0.0000]
corporate_colors = [light_blue,light_grey, dark_blue, yellow, anthracit]


class PlotData(TypedDict):
    fig: plt.Figure
    ax: plt.Axes


def migrate_lines(original_axes: plt.Axes, new_axes: plt.Axes):
    """ Move line artists from one plt.Axes to another """
    num_lines = len(original_axes.lines)
    for i in range(num_lines):
        line = original_axes.lines[0]
        line.remove()
        new_axes.add_line(line)
        line.axes = new_axes
        line.set_transform(new_axes.transData)


def stack_plots(figures: List[plt.Figure],
                axes: List[plt.Axes],
                sharex: bool = True,
                hspace: float = 0.02) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Vertically stack already created single figures and their respective Axes into one subplot figure.

    :param figures:
    :param axes:
    :param sharex:
    :param hspace:
    :return:
    """
    heights = [f.get_figheight() for f in figures]
    total_height = sum(heights)
    height_ratios = [(val/total_height)*min(heights) for val in heights]
    combined_plot_fig, combined_plot_axes = plt.subplots(nrows=len(figures),
                                                         ncols=1, sharex=sharex,
                                                         gridspec_kw = {'height_ratios': height_ratios,
                                                                        'hspace': hspace})
    combined_plot_fig.set_figheight(total_height)
    for i, ax in enumerate(axes):
        combined_plot_axes[i].set_xlim(ax.get_xlim())
        combined_plot_axes[i].set_ylim(ax.get_ylim())
        combined_plot_axes[i].set_yscale(ax.get_yscale())
        combined_plot_axes[i].set_ylabel(ax.get_ylabel())
        combined_plot_axes[i].set_xlabel(ax.get_xlabel())
        combined_plot_axes[i].yaxis.set_minor_formatter(ax.yaxis.get_minor_formatter())
        combined_plot_axes[i].xaxis.set_minor_formatter(ax.xaxis.get_minor_formatter())
        combined_plot_axes[i].yaxis.set_major_formatter(ax.yaxis.get_major_formatter())
        combined_plot_axes[i].xaxis.set_major_formatter(ax.xaxis.get_major_formatter())
        combined_plot_axes[i].set_xlabel(ax.get_xlabel())
        migrate_lines(ax, combined_plot_axes[i])
    return combined_plot_fig, combined_plot_axes


def smooth(data:np.ndarray, tensorboard_factor: float = 0.6) -> np.ndarray:
    """ apply exponential weighted MA """
    df = pandas.DataFrame(data=data)
    return df.ewm(alpha=(1 - tensorboard_factor)).mean().to_numpy()


def add_smoothing(ax: plt.Axes, smoothing_value: float = 0.6):
    """
    Apply EMA smoothing to any lines in the image. original line will be made translucent and new line takes
    original color

    :param ax:
    :param smoothing_value:
    :return:
    """
    if 0 < smoothing_value < 1:
        new_lines = []
        for line in ax.lines:
            line: plt.Line2D
            x = line.get_xdata(orig=False)
            y = line.get_ydata(orig=False)
            line.set_alpha(0.2)
            smoothed = smooth(y, smoothing_value)
            nline = plt.Line2D(x, smoothed, linewidth=line.get_linewidth(),
                                        linestyle=line.get_linestyle(), color=line.get_color(), label=line.get_label())
            line.set_label(f'_{nline.get_label()}')
            new_lines.append(nline)

        [ax.add_line(nline) for nline in new_lines]
        plt.draw()


def get_plots(log_directories,
              tags: List[str],
              width: float = 11.0,
              height: float = 5,
              log: bool = False,
              line_styles: List[str] = None,
              log_directories_connected: bool = False) -> Dict[str, PlotData]:
    """
    Get a separate plot for each tag in tags from the tensorboard plots in log_directories

    :param log_directories:
    :param tags:
    :param width:
    :param height:
    :param log:
    :param line_styles:
    :param log_directories_connected:
    :return:
    """
    return_plots = {}
    data = [get_scalars_from_events(directory, tags) for directory in log_directories]
    for tag in tags:
        return_plots[tag] = {}
        fig = plt.figure(figsize=(width * cm2inches, height*cm2inches))
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
                            if log_directories_connected and tag in [l.get_label() for l in ax.lines]:
                                l = [l for l in ax.lines if l.get_label() == tag][0]

                                current_data = np.stack([l.get_xdata(orig=True), l.get_ydata(orig=True)])
                                new_data = np.stack([x,y])
                                new_data = np.concatenate([current_data, new_data], axis=1)
                                new_data = new_data[:, new_data[0].argsort()]
                                l.set_xdata(new_data[0, :])
                                l.set_ydata(new_data[1, :])
                            else:
                                lbl = tag if log_directories_connected else f'{tag}_{i}'
                                ax.plot(x, y, label=lbl, color=corporate_colors[i])
                        else:
                            sub_tags = [key for key in scalar_tup[1] if ('_step' not in key and '_t' not in key)]
                            for j, sub_tag in enumerate(sub_tags):
                                y = scalar_tup[1][sub_tag]
                                x = scalar_tup[1][f'{sub_tag}_step']
                                if log_directories_connected and tag in [l.get_label() for l in ax.lines]:
                                    l = [l for l in ax.lines if l.get_label() == tag][0]
                                    current_data = np.stack([l.get_xdata(orig=True), l.get_ydata(orig=True)])
                                    new_data = np.stack([x,y])
                                    new_data = np.concatenate([current_data, new_data], axis=1)
                                    new_data = new_data[:, new_data[0].argsort()]
                                    l.set_xdata(new_data[0, :])
                                    l.set_ydata(new_data[1, :])
                                else:
                                    lbl = tag if log_directories_connected else f'{tag}_{i}'
                                    ax.plot(x, y, label=lbl, color=corporate_colors[j], linestyle=line_styles[i] if line_styles is not None else 'solid')
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
    parser.add_argument('--isolated', action='store_false', help="treat each directory as a completely unrelated run.")

    args = parser.parse_args()
    tags: List[str] = args.tags
    directories: List[str] = args.log_dirs
    [exit(1) for directory in directories if not os.path.isdir(directory)]
    linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']
    for tag, items in get_plots(directories, tags, args.width, args.log, line_styles=linestyle_str,
                                log_directories_connected=not args.isolated).items():
        add_smoothing(items['ax'], args.smoothing)
        items['fig'].savefig(os.path.join(args.output, f'{tag}.pdf'), bbox_inches='tight', pad_inches=0.0, dpi='figure')