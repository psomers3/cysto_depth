import torch


class TorchPicklableGenerator:
    def __init__(self, seed: int = None):
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __setstate__(self, state):
        self.rng = torch.Generator(state['device'])
        self.rng.set_state(state['rng_state'])

    def __getstate__(self):
        return {'rng_state': self.rng.get_state(),
                'device': self.rng.device}


if __name__ == '__main__':
    import dill
    rng = TorchPicklableGenerator()
    dill.pickles(rng)