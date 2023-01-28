from typing import *
from torchvision import transforms as torch_transforms
from data.image_dataset import ImageDataset
import data.data_transforms as d_transforms
from data.general_data_module import FileLoadingDataModule


class EndoDepthDataModule(FileLoadingDataModule):
    def __init__(self,
                 batch_size,
                 data_roles: List[str],
                 data_directories: List[Union[str, List[str]]],
                 split: dict = None,
                 image_size: int = 256,
                 workers_per_loader: int = 6,
                 depth_scale_factor: float = 1e3,
                 inverse_depth: bool = False):
        """ A Data Module for loading rendered endoscopic images with corresponding depth maps. The color images should
        be stored in a different directory as the depth images. See the split parameter for thoughts on how best to
        set up your data structure for use with this module. The images will be made square and a circular mask applied
        to simulate actual endoscopic images.
        TODO: set up to use a configuration for control of all transforms and other hyperparams.

        :param batch_size: batch sized to use for training.
        :param data_directories: paths to the corresponding data roles. Will be searched recursively.
        :param split: see parent class FileLoadingDataModule.
        :param image_size: final `square` image size to return for training.
        :param workers_per_loader: cpu threads to use for each data loader.
        :param depth_scale_factor: factor to scale the depth values by. Useful for switching between meters and mm.
        :param inverse_depth: whether to invert the depth values (inf depth = 0)  TODO: implement...
        """

        directories = dict(zip(data_roles, data_directories))
        super().__init__(batch_size, directories, split, workers_per_loader)
        self.save_hyperparameters("batch_size")
        self.image_size = image_size
        self.scale_factor = depth_scale_factor
        self.invert_depth = inverse_depth

    def get_transforms(self, stage: str) -> List[torch_transforms.Compose]:
        """ get the list of transforms for each data channel (i.e. image, label)

            :param stage: one of 'train', 'val', 'test'
        """
        num_synchros = 2

        normalize = d_transforms.ImageNetNormalization()
        img_squarify = d_transforms.Squarify(image_size=self.image_size, clamp_values=True)
        depth_squarify = d_transforms.Squarify(image_size=self.image_size)
        mask = d_transforms.SynchronizedTransform(transform=d_transforms.EndoMask(radius_factor=[0.9, 1.0]),
                                                  num_synchros=num_synchros, additional_args=[[None], [0], [0]])
        affine_transform = d_transforms.SynchronizedTransform(transform=d_transforms.RandomAffine(degrees=(0, 360),
                                                                                                  translate=(.1, .1)),
                                                              num_synchros=num_synchros, additional_args=[[True],
                                                                                                          [False]])
        color_jitter = torch_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        to_mm = d_transforms.ElementWiseScale(self.scale_factor)
        channel_slice = d_transforms.TensorSlice((0, ...))  # depth exr saves depth in each RGB channel
        depth_transforms = [channel_slice, to_mm, mask, depth_squarify]
        if self.invert_depth:
            depth_transforms.insert(2, d_transforms.DepthInvert())

        color_transforms = [mask, img_squarify]

        if stage.lower() == 'train':
            depth_transforms.append(affine_transform)
            color_transforms.insert(0, color_jitter)
            color_transforms.append(affine_transform)
        color_transforms.append(normalize)

        composed_color_transforms = torch_transforms.Compose(color_transforms)
        composed_depth_transforms = torch_transforms.Compose(depth_transforms)
        transforms = [composed_color_transforms, composed_depth_transforms]
        return transforms

    def setup(self, stage: str = None):
        self.data_train = ImageDataset(files=list(zip(*self.split_files['train'].values())),
                                       transforms=self.get_transforms('train'))
        self.data_val = ImageDataset(files=list(zip(*self.split_files['validate'].values())),
                                     transforms=self.get_transforms('validate'))
        self.data_test = ImageDataset(files=list(zip(*self.split_files['test'].values())),
                                      transforms=self.get_transforms('test'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.image_utils import matplotlib_show
    from data_transforms import ImageNetNormalization

    denorm = ImageNetNormalization(inverse=True)
    color_dir = r'/Users/peter/isys/2023_01_25/color'
    depth_dir = r'/Users/peter/isys/2023_01_25/depth'
    dm = EndoDepthDataModule(batch_size=3,
                             data_roles=['color', 'depth'],
                             data_directories=[color_dir, depth_dir],
                             split={'train': .6, 'validate': 0.4, 'test': ".*00015.*"},
                             inverse_depth=True)
    dm.setup('fit')
    loader = dm.train_dataloader()
    samples = next(iter(loader))
    samples[0] = denorm(samples[0])
    matplotlib_show(*samples)
    plt.show(block=True)
