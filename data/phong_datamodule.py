from typing import *
import numpy as np
import torch
from torchvision import transforms as torch_transforms
from data.image_dataset import ImageDataset
import data.data_transforms as d_transforms
from data.general_data_module import FileLoadingDataModule
from utils.rendering import get_pixel_locations, get_image_size_from_intrisics, render_rgbd, get_normals_from_depthmap
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials


class PhongDataSet(ImageDataset):
    def __init__(self, *args, camera_intrinsics, image_size, **kwargs):
        super(PhongDataSet, self).__init__(*args, **kwargs)
        self.mask = d_transforms.SynchronizedTransform(transform=d_transforms.EndoMask(radius_factor=[0.9, 1.0]),
                                                       num_synchros=2, additional_args=[[None], [0]])
        self.camera_intrinsics = torch.Tensor(camera_intrinsics)
        self.squarify = d_transforms.Squarify(image_size)
        # get the original camera pixel locations at the desired image resolution
        original_image_size = get_image_size_from_intrisics(self.camera_intrinsics)
        pixels = get_pixel_locations(*original_image_size)
        self.resized_pixel_locations = self.squarify(torch.permute(pixels, (2, 0, 1)))
        self.resized_pixel_locations = torch.permute(self.resized_pixel_locations, (1, 2, 0))
        self.grey = torch.ones((image_size, image_size, 3)) * .5

        self.material = Materials(shininess=1)
        self.light = PointLights(location=((0, 0, 0),),
                                 diffuse_color=((1, 1, 1),),
                                 specular_color=((.0, .0, .0),),
                                 ambient_color=((0.0, 0.0, 0.0),),
                                 attenuation_factor=(4,))

    def __getitem__(self, idx):
        color, depth, normals = super(PhongDataSet, self).__getitem__(idx)
        # normals = get_normals_from_depthmap(torch.permute(depth*1e3, (1, 2, 0)))
        rendered = render_rgbd(torch.permute(depth, (1, 2, 0)),
                               self.grey,
                               torch.permute(normals, (1, 2, 0)),
                               # normals,
                               self.camera_intrinsics,
                               self.light,
                               self.material,
                               self.resized_pixel_locations)
        masked_color = self.mask(color)
        masked_rendering = self.mask(torch.permute(rendered, (2, 0, 1)))
        return masked_color, masked_rendering


class PhongDataModule(FileLoadingDataModule):
    def __init__(self,
                 batch_size,
                 color_image_directory: str,
                 depth_image_directory: str,
                 normals_image_directory: str,
                 camera_intrinsics: np.ndarray,
                 split: dict = None,
                 image_size: int = 256,
                 workers_per_loader: int = 6):
        """ A Data Module for loading rendered endoscopic images with corresponding depth maps. The color images should
        be stored in a different directory as the depth images. See the split parameter for thoughts on how best to
        set up your data structure for use with this module. The images will be made square and a circular mask applied
        to simulate actual endoscopic images.
        TODO: set up to use a configuration for control of all transforms and other hyperparams.

        :param batch_size: batch sized to use for training.
        :param color_image_directory: path to the color images. Will be searched recursively.
        :param depth_image_directory: path to the depth images. Will be searched recursively.
        :param normals_image_directory: path to the normals (in camera coords). Will be searched recursively. Ignored
                                        if None.
        :param camera_intrinsics: 3x3 camera intrinsics matrix
        :param split: see parent class FileLoadingDataModule.
        :param image_size: final `square` image size to return for training.
        :param workers_per_loader: cpu threads to use for each data loader.
        """
        directories = {'color': color_image_directory,
                       'depth': depth_image_directory,
                       'normals': normals_image_directory}

        super().__init__(batch_size, directories, split, workers_per_loader)
        self.save_hyperparameters()
        self.camera_intrinsics = camera_intrinsics
        self.image_size = image_size

    def get_transforms(self, stage: str) -> List[torch_transforms.Compose]:
        """ get the list of transforms for each data channel (i.e. image, label)
            TODO: don't apply random augmentations to validation and test transforms

            :param stage: one of 'train', 'val', 'test'
        """
        num_synchros = 3
        # normalize = torch_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
        squarify = d_transforms.Squarify(image_size=self.image_size)

        color_jitter = torch_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        to_mm = d_transforms.ElementWiseScale(1e3)
        channel_slice = d_transforms.TensorSlice((0, ...))  # depth exr saves depth in each RGB channel
        color_transforms = torch_transforms.Compose([color_jitter, squarify])
        depth_transforms = torch_transforms.Compose([channel_slice, to_mm, squarify])
        normals_transforms = torch_transforms.Compose([squarify])
        transforms = [color_transforms, depth_transforms, normals_transforms]
        return transforms

    def setup(self, stage: str = None):
        shared_params = {'camera_intrinsics': self.camera_intrinsics, 'image_size': self.image_size}
        self.data_train = PhongDataSet(**shared_params,
                                       files=list(zip(*self.split_files['train'].values())),
                                       transforms=self.get_transforms('train'))
        self.data_val = PhongDataSet(**shared_params,
                                     files=list(zip(*self.split_files['validate'].values())),
                                     transforms=self.get_transforms('validate'))
        self.data_test = PhongDataSet(**shared_params,
                                      files=list(zip(*self.split_files['test'].values())),
                                      transforms=self.get_transforms('test'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.image_utils import matplotlib_show

    intrinsics = np.array([[1038.1696537477499, 0, 0],
                           [0, 1039.8075384016558, 0],
                           [878.9617517840989, 572.9404979327502, 1]]).T

    color_dir = r'../test/output/color'
    depth_dir = r'../test/output/depth'
    normals_dir = r'../test/output/normal'
    dm = PhongDataModule(batch_size=3,
                         color_image_directory=color_dir,
                         depth_image_directory=depth_dir,
                         normals_image_directory=normals_dir,
                         camera_intrinsics=intrinsics,
                         split={'train': .5, 'validate': 0.25, 'test': .25})
    dm.setup('fit')
    loader = dm.train_dataloader()
    matplotlib_show(*next(iter(loader)))
    plt.show(block=True)
