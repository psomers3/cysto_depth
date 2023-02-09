from config.training_config import SyntheticTrainingConfig, GANTrainingConfig
from typing import *
from gan_model import GAN
from torch import Tensor


class WGAN(GAN):
    """ A Wasserstein GAN """
    def __init__(self,
                 synth_config: SyntheticTrainingConfig,
                 gan_config: GANTrainingConfig,
                 image_gan: bool = False,
                 wasserstein_lambda: float = 10):
        super().__init__(synth_config, gan_config, image_gan)
        self.wasserstein_lamba = wasserstein_lambda

    @staticmethod
    def adversarial_loss(y_hat, y) -> Tensor:
        """ Wasserstein loss """
        return (y_hat * y).mean()

    def __call__(self, *args, **kwargs) -> Tuple[Union[Tensor, List[Tensor]], ...]:
        return super(WGAN, self).__call__(*args, **kwargs)
