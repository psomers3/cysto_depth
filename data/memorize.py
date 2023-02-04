from torch.utils.data import Dataset


class MemorizeCheck(Dataset):
    def __init__(self, batch, length):
        self.batch = batch
        self.length = length
        self.i = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.i == len(self.batch):
            self.i = 0
        self.i += 1
        return self.batch[0][self.i - 1], self.batch[1][self.i - 1]
