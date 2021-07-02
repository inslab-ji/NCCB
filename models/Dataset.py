from torch.utils.data import Dataset


class NLPDataset(Dataset):
    def __init__(
        self,
        data
    ):
        self.data = data

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, index):
        return self.data[index]


