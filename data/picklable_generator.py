import torch


class TorchPicklableGenerator:
    def __init__(self, seed: int = None):
        self.__dict__['_wrapped_rng'] = torch.Generator()
        if seed is not None:
            self.manual_seed(seed)

    def __setstate__(self, state):
        self.__dict__['_wrapped_rng'] = torch.Generator(state['device'])
        self.set_state(state['rng_state'])

    def __getstate__(self):
        return {'rng_state': self.get_state(),
                'device': self.device}

    def __getattr__(self, item):
        return getattr(self.__dict__['_wrapped_rng'], item)

    def __setattr__(self, name, value):
        setattr(self.__dict__['_wrapped_rng'], name, value)

    def __call__(self, *args, **kwargs) -> torch.Generator:
        return self.__dict__['_wrapped_rng']


if __name__ == '__main__':
    import dill
    rng = TorchPicklableGenerator()
    dill.pickles(rng)
    rng.get_state()
    torch.rand(1, generator=rng())