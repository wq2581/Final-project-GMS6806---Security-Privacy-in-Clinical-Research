

from torch.utils.data import Dataset, DataLoader

import os
import numpy as np

class embeds(Dataset):
    def __init__(self, data_path):
        super(embeds, self).__init__()
        print('loading dataset...')
        ebd = np.load(data_path)



        self.embeds = ebd

        self.length = len(self.embeds)
    def __getitem__(self, item):
        return self.embeds[item]

    def __len__(self):
        return self.length









