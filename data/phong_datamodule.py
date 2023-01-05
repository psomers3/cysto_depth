from typing import *
import torch
from torchvision import transforms as torch_transforms
from data.image_dataset import ImageDataset
import data.data_transforms as d_transforms
from data.general_data_module import FileLoadingDataModule
from utils.rendering import get_pixel_locations, get_image_size_from_intrisics, render_rgbd, PointLights, Materials
from config.training_config import PhongConfig


class PhongDataSet(ImageDataset):
    """
    Load images, depth maps, and normals and create a phong shaded image. Order of returned items:
    color image, phong shaded image, depth (if requested), normals (if requested)
    """
    def __init__(self, *args, image_size: int, config: PhongConfig, **kwargs):
        """

        :param args:
        :param image_size:
        :param config:
        :param kwargs:
        """
        # pop the transforms to apply afterwards because the default ImageDataset will only know of the loaded images
        # and not the created ones
        super(PhongDataSet, self).__init__(*args, **kwargs)
        self.camera_intrinsics = torch.Tensor(config.camera_intrinsics)
        self.squarify = d_transforms.Squarify(image_size)
        # get the original camera pixel locations at the desired image resolution
        original_image_size = get_image_size_from_intrisics(self.camera_intrinsics)
        pixels = get_pixel_locations(*original_image_size)
        self.resized_pixel_locations = self.squarify(torch.permute(pixels, (2, 0, 1)))
        self.resized_pixel_locations = torch.permute(self.resized_pixel_locations, (1, 2, 0))
        self.grey = torch.ones((image_size, image_size, 3)) * .5

        self.material = Materials(shininess=config.material_shininess)
        self.light = PointLights(location=((0, 0, 0),),
                                 diffuse_color=(config.diffusion_color,),
                                 specular_color=(config.specular_color,),
                                 ambient_color=(config.ambient_color,),
                                 attenuation_factor=config.attenuation,
                                 device=self.device)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        """
        TODO: handle transformation orders better... mostly depth stuff for slicing and m -> mm
        :param idx:
        :return:
        """
        imgs = super(PhongDataSet, self).__getitem__(idx)  # these are channel first
        color, depth, normals = imgs
        rendered = render_rgbd(depth.permute((1, 2, 0)),
                               self.grey,
                               normals.permute((1, 2, 0)),
                               self.camera_intrinsics,
                               self.light,
                               self.material,
                               self.resized_pixel_locations)
        return color, rendered.permute((2, 0, 1)), depth, normals


class PhongDataModule(FileLoadingDataModule):
    def __init__(self,
                 batch_size,
                 color_image_directory: str,
                 depth_image_directory: str,
                 normals_image_directory: str,
                 split: dict = None,
                 image_size: int = 256,
                 workers_per_loader: int = 6,
                 phong_config: PhongConfig = PhongConfig()):
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
        :param phong_config: configuration for the phong dataset
        :param split: see parent class FileLoadingDataModule.
        :param image_size: final `square` image size to return for training.
        :param workers_per_loader: cpu threads to use for each data loader.
        """
        directories = {'color': color_image_directory,
                       'depth': depth_image_directory,
                       'normals': normals_image_directory}

        super().__init__(batch_size, directories, split, workers_per_loader)
        self.phong_config = phong_config
        self.save_hyperparameters()
        self.image_size = image_size
        self.num_synchros = 3

    def get_transforms(self, split_stage: str) -> List[torch_transforms.Compose]:
        """ get the list of transforms for each data channel (i.e. image, label)

            :param split_stage: one of 'train', 'validate', 'test'
        """
        to_mm = d_transforms.ElementWiseScale(1e3)
        mask = d_transforms.SynchronizedTransform(transform=d_transforms.EndoMask(radius_factor=[0.9, 1.0]),
                                                  num_synchros=self.num_synchros,
                                                  additional_args=[[None], [0], [0], [0]])
        squarify = d_transforms.Squarify(image_size=self.image_size)
        color_transforms = [mask, squarify]
        channel_slice = d_transforms.TensorSlice((0, ...))  # depth exr saves depth in each RGB channel
        depth_transforms = [channel_slice, to_mm, mask, squarify]
        normals_transforms = [mask, squarify]
        if split_stage == "train":
            # affine = d_transforms.SynchronizedTransform(d_transforms.PhongAffine(degrees=(0, 359),
            #                                                                      translate=(0, 0),
            #                                                                      image_size=self.image_size),
            #                                             num_synchros=self.num_synchros,
            #                                             additional_args=[[True],
            #                                                              [False, True],
            #                                                              [False, True]])
            color_jitter = torch_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            color_transforms.insert(0, color_jitter)
            # color_transforms.append(affine)
            # depth_transforms.append(affine)
            # normals_transforms.append(affine)

        color_transforms = torch_transforms.Compose(color_transforms)
        depth_transforms = torch_transforms.Compose(depth_transforms)
        normals_transforms = torch_transforms.Compose(normals_transforms)
        transforms = [color_transforms, depth_transforms, normals_transforms]
        return transforms

    def setup(self, stage: str = None):
        """
        :param stage: "fit", "test", or "predict"
        """
        shared_params = {'image_size': self.image_size}
        self.data_train = PhongDataSet(**shared_params,
                                       config=self.phong_config,
                                       files=list(zip(*self.split_files['train'].values())),
                                       transforms=self.get_transforms('train'))
        self.data_val = PhongDataSet(**shared_params,
                                     config=self.phong_config,
                                     files=list(zip(*self.split_files['validate'].values())),
                                     transforms=self.get_transforms('validate'))
        self.data_test = PhongDataSet(**shared_params,
                                      config=self.phong_config,
                                      files=list(zip(*self.split_files['test'].values())),
                                      transforms=self.get_transforms('test'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.image_utils import matplotlib_show
    from utils.loss import PhongLoss

    color_dir = '/Users/peter/Desktop/bladder_dataset_filtered/color'
    depth_dir = '/Users/peter/Desktop/bladder_dataset_filtered/depth'
    normals_dir = '/Users/peter/Desktop/bladder_dataset_filtered/normals'
    phong = PhongConfig(attenuation=0.01, material_shininess=100)
    dm = PhongDataModule(batch_size=4,
                         color_image_directory=color_dir,
                         depth_image_directory=depth_dir,
                         normals_image_directory=normals_dir,
                         split={'train': .9, 'validate': 0.05, 'test': 0.05},
                         phong_config=phong)
    dm.setup('fit')
    loader = dm.train_dataloader()
    loader_iter = iter(loader)
    loss = PhongLoss(phong, device='cpu')
    while True:
        sample = next(loader_iter)
        loss_value, prediction = loss((sample[2], sample[3]), sample[1])
        print(loss_value)
        sample[-1] = sample[-1]*0.5 + 0.5
        matplotlib_show(*sample)
        matplotlib_show(prediction)
        plt.show(block=False)
        plt.pause(5)
        input("")
        for i in plt.get_fignums():
            plt.close(i)
