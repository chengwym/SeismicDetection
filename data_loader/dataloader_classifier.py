import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

from config.constant import RANDOM_STATE

class SeismicDataet(Dataset):
    def __init__(self, mode, path):
        self.mode = mode
        self.CLASS = ['earthquakes', 'glacial']
        self.num_class = len(self.CLASS)
        earthquakes = os.listdir(f'{path}/earthquakes')
        earthquakes = [f'{path}/earthquakes/{file}' for file in earthquakes]
        glacial = os.listdir(f'{path}/glacial')
        glacial = [f'{path}/glacial/{file}' for file in glacial]
        dirs = earthquakes + glacial
        train_dirs, test_dirs = train_test_split(dirs, test_size=.2, random_state=RANDOM_STATE)
        train_dirs, val_dirs = train_test_split(dirs, test_size=.3, random_state=RANDOM_STATE)
        self.train_dirs, self.val_dirs, self.test_dirs = train_dirs, val_dirs, test_dirs

    def __getitem__(self, index):
        if self.mode == 'test':
            path = self.test_dirs[index]
        elif self.mode == 'train':
            path = self.train_dirs[index]
        else:
            path = self.val_dirs[index]
        data = torch.FloatTensor(mpimg.imread(path))
        # print(data.shape)
        words = path.split('/')
        label = words[2]
        # print(label)
        indice = self.CLASS.index(label)
        target = torch.FloatTensor([indice])
        # print(indice)
        # print(target)
        return data, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_dirs) 
        elif self.mode == 'eval':
            return len(self.val_dirs)
        else:
            return len(self.train_dirs)

def SeismicDataLoader(mode, batch_size, path):
    dataset = SeismicDataet(mode, path)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=(False if mode=='test' else True),
        batch_size=batch_size,
        pin_memory=True,
    )
    return dataloader