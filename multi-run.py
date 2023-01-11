import sys
import time
from threading import Thread
from threading import Event
import subprocess
import os
from argparse import ArgumentParser
from config.blender_config import MainConfig
from omegaconf import OmegaConf, DictConfig

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--config', default='config/config.yaml', type=str, help='path to config file')
    parser.add_argument('--gpu', type=int, default=-1, help='specify gpu to use. defaults to all available')
    parser.add_argument('--shell', type=str, default='bash', help='The shell to use. Will source the rc file before '
                                                                  'running blender')
    args, unknown_args = parser.parse_known_args()
    cli_conf = OmegaConf.from_cli(unknown_args)  # assume any additional args are config overrides
    cfg = DictConfig(OmegaConf.load(args.config))
    config: MainConfig = OmegaConf.merge(OmegaConf.structured(MainConfig()), cfg, cli_conf)
    num_processes = args.num_processes
    config_file = args.config
    gpu = args.gpu
    stop = Event()

    def rendering_process(samples_per_model: int, threads_per_process: int, thread_id: int):
        worker = subprocess.Popen([f'{args.shell}', '-i', '-c', ' '.join(
            ['blender', '-b', '--python', 'blender_rendering.py', '--',
             '--render',
             '--config', config_file,
             '--gpu', f"{gpu}",
             '--id_offset', f'{thread_id * samples_per_model}',
             f'samples_per_model={samples_per_model}',
             f'blender.render.threads={threads_per_process}'])],
                                  shell=False,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        print(worker.args)
        # line = worker.stdout.readline().decode('ascii')
        while worker.poll() is None:
            # print(thread_id)
            if stop.isSet():
                worker.terminate()
                stop.set()
                break
            worker.stdout.flush()
            time.sleep(0.1)
        if worker.poll() == 2:  # ctrl + c caught by underlying process
            stop.set()

    threads = []
    threads_to_use = config.blender.render.threads // num_processes
    samples = config.samples_per_model // num_processes
    for i in range(num_processes):
        threads.append(Thread(target=rendering_process, args=(samples, threads_to_use, i), daemon=True))
        threads[-1].start()

    while not all([not t.is_alive() for t in threads]):
        try:
            [t.join(.1) for t in threads]
        except KeyboardInterrupt as e:
            print(e)
            stop.set()


