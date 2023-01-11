import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SeismicDataet(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.CLASS = ['earthquakes', 'glacial']
        self.num_class = len(self.CLASS)
        earthquakes = os.listdir('../data/earthquakes')
        earthquakes = [f'../data/earthquakes/{file}' for file in earthquakes]
        glacial = os.listdir('../data/glacial')
        glacial = [f'../data/glacial/{file}' for file in glacial]
        dirs = earthquakes + glacial
        train_dirs, test_dirs = train_test_split(dirs, test_size=.2)
        train_dirs, val_dirs = train_test_split(dirs, test_size=.3)
        self.train_dirs, self.val_dirs, self.test_dirs = train_dirs, val_dirs, test_dirs

    def __getitem__(self, index):
        if self.mode == 'test':
            path = self.test_dirs[index]
        elif self.mode == 'train':
            path = self.train_dirs[index]
        else:
            path = self.train_dirs[index]
        data = torch.FloatTensor(np.fromfile(path, dtype=np.float64).reshape(129, 7200))
        words = path.split('/')
        label = words[2]
        indice = self.CLASS.index(label)
        target = torch.FloatTensor(indice)
        return data, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_dirs) 
        elif self.mode == 'eval':
            return len(self.val_dirs)
        else:
            return len(self.train_dirs)

def SeismicDataLoader(mode, batch_size):
    dataset = SeismicDataet(mode)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=(False if mode=='test' else True),
        batch_size=batch_size,
        pin_memory=True
    )
    return dataloader