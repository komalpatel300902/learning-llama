import torch
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__" :
    word_map = {
        'PAD': 0, 'SOS': 1, 'EOS': 2,
        'one': 3, 'two': 4, 'three': 5,
        'un': 6, 'deux': 7, 'trois': 8
    }
    inv_word_map = {v: k for k, v in word_map.items()}

    data = [
        ([3], [6]),  # one → un
        ([4], [7]),  # two → deux
        ([5], [8])  # three → trois
    ]


    class NumberDataset(Dataset):
        def __init__(self, pairs):
            self.data = pairs

        def __len__(self): return len(self.data)

        def __getitem__(self, i):
            src, tgt = self.data[i]
            src = [1] + src + [2]  # SOS ... EOS
            tgt = [1] + tgt + [2]
            return torch.tensor(src), torch.tensor(tgt)


    dataset = NumberDataset(data)
    for x in dataset:
        print(x)