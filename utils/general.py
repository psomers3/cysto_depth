import inspect
from config.training_config import CallbackConfig
from typing import *
import pytorch_lightning as pl


def get_default_args(func) -> dict:
    """
    Get expected arguments and their defaults for a function

    :param func: function to get defaults from
    :return: dictionary of function argument names and default values
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_callbacks(configuration: CallbackConfig) -> List[pl.Callback]:
    """
    Parses a CallbackConfig object to return the desired list of training callbacks

    :param configuration: the configuration for multiple callbacks
    :return: a list of pytorch-lightning callbacks
    """
    callbacks = []
    if configuration.early_stop_patience:
        callbacks.append(pl.callbacks.EarlyStopping(monitor=configuration.early_stop_metric,
                                                    patience=configuration.early_stop_patience,
                                                    verbose=configuration.early_stop_verbose))
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor=configuration.ckpt_metric,
                                                  save_top_k=configuration.ckpt_save_top_k,
                                                  every_n_epochs=configuration.ckpt_every_n_epochs))
    if configuration.lr_monitoring:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval=configuration.lr_monitor_interval))
    return callbacks
