from torch import nn
import csv
import os
import glob
from utils.exr_utils import get_exr_max_depth
import torch


def scale_median(predicted, gt):
    # eigen crop

    flattened_gt = torch.flatten(gt, start_dim=1)
    flattened_predicted = torch.flatten(predicted, start_dim=1)
    scale_predicted = []
    for target_flat, predicted_flat in zip(flattened_gt, flattened_predicted):
        # minimum value (0) and maximum value (sky) are not interesting -> ignore them in scaling
        # 
        mask = (target_flat > 1e-3) & (target_flat < 80)
        # only use values where target is not zero for sparse depth
        median_label = torch.median(torch.masked_select(target_flat, mask))
        median_predicted = torch.median(torch.masked_select(predicted_flat, mask))
        scale_predicted.append(median_label / median_predicted)
    scale_predicted = torch.stack(scale_predicted)
    # print(torch.std(scale_predicted))
    # print(torch.mean(scale_predicted))
    return scale_predicted.view(-1, 1, 1, 1)


_normalizations = {"instance": nn.InstanceNorm2d, "batch": nn.BatchNorm2d, "layer": nn.LayerNorm}


def convrelu(in_channels, out_channels, kernel, padding, stride=1, transpose=False, alpha=0.01, norm: str = "",
             activation='relu', init_zero=False):
    layers = []
    if transpose:
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding))
    if init_zero:
        conv_layer = layers[0]
        for p in conv_layer.parameters():
            p.data.fill_(0)
    if norm:
        layers.append(_normalizations[norm](out_channels))

    if activation == "leaky":
        layers.append(nn.LeakyReLU(inplace=True, negative_slope=alpha))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def generateImageAnnotations(annotations_path, data_dir, test_dir):
    testset_idx = 0
    trainset_idx = 0
    valset_idx = 0
    testset_folders = ["sim-labels-137_bladder_smooth", "sim-labels-136_bladder_smooth"]
    valset_folders = ["sim-labels-134_bladder_smooth", "sim-labels-133_bladder_smooth"]

    with open(annotations_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Set", "Idx", "Path", "Depth"])
        max_depth = 0
        dirs = [data_dir, test_dir]
        for dir_idx, dir in enumerate(dirs):
            for folder in os.listdir(dir):
                abs_path = os.path.join(dir, folder)
                # only consider dirs
                if os.path.isdir(abs_path):
                    png_files = glob.glob(f'{abs_path}/*.png')
                    for png_file in png_files:
                        filepath = png_file.removesuffix(".png")
                        depth = get_exr_max_depth(filepath)
                        if depth < 1000:
                            max_depth = max([max_depth, depth])
                            if folder in testset_folders:
                                writer.writerow(["test", testset_idx, filepath, depth])
                                testset_idx += 1
                            elif folder in valset_folders:
                                writer.writerow(["val", valset_idx, filepath, depth])
                                valset_idx += 1
                            else:
                                writer.writerow(["train", trainset_idx, filepath, depth])
                                trainset_idx += 1
            writer.writerow(["Max_Depth", None, None, max_depth])
