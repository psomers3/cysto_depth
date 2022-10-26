from multiprocessing.pool import ThreadPool
import subprocess
import sys
from typing import *
from argparse import ArgumentParser
from config import MainConfig
from omegaconf import OmegaConf, DictConfig


def extract_system_arguments() -> Tuple[List[str], bool]:
    """
    Provides the user passed arguments that come after "--" when running a python script through blender.
    Also provides whether the script was called as headless

    :return: parsed_args, headless
    """
    idx = 0
    try:
        idx = sys.argv.index("--")
        cli_arguments = True
    except ValueError:
        cli_arguments = False
    arg_string = sys.argv[idx + 1:] if cli_arguments else ""

    gui_enabled = False
    try:
        gui_enabled = bool(sys.argv.index('-b'))
    except ValueError:
        pass

    return arg_string, not gui_enabled


if __name__ == '__main__':
    arguments, headless = extract_system_arguments()
    parser = ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--config', default='config/config.yaml', type=str, help='path to config file')
    parser.add_argument('--gpu', type=int, default=-1, help='specify gpu to use. defaults to all available')
    args, unknown_args = parser.parse_known_args(arguments)
    cli_conf = OmegaConf.from_cli(unknown_args)  # assume any additional args are config overrides
    cfg = DictConfig(OmegaConf.load(args.config))
    config: MainConfig = OmegaConf.merge(OmegaConf.structured(MainConfig()), cfg, cli_conf)

    num_processes = args.num_processes
    config_file = args.config
    gpu = args.gpu

    def rendering_process(samples_per_model: int):
        worker = subprocess.Popen(['blender', '-b', '--python', 'blender_rendering.py', '--',
                                   '--render',
                                   '--config', config_file,
                                   f'samples_per_model={samples_per_model}'],
                                  shell=False,
                                  stdout=subprocess.PIPE)
        print(worker.args)
        line = True
        while line:
            out_line = worker.stdout.readline()
            # look for something in the output if you'd like


    tp = ThreadPool(num_processes)
    samples = config.samples_per_model // num_processes
    for _ in range(num_processes):
        tp.apply_async(rendering_process, (samples,))
    tp.close()
    tp.join()






