__author__  = "Jingbo Cheng"

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

from utils.parse_catalog import get_earthquakes_dir, get_glacial_dir
from config.constant import RANDOM_STATE

class SeismicRegDataset(Dataset):
    def __init__(self, mode, path, task):
        """
        mode: test, train, eval
        path: the path to the data folder
        task: earthquakes, glacial
        """
        self.mode = mode
        self.task = task
        earthquakes = os.listdir(f'{path}/earthquakes')
        earthquakes = [f'{path}/earthquakes/{file}' for file in earthquakes]
        glacial = os.listdir(f'{path}/glacial')
        glacial = [f'{path}/glacial/{file}' for file in glacial]
        if task == 'earthquakes':
            self.target_dict = get_earthquakes_dir(f'{path}/earthquakes_catalog.txt')
            dirs = earthquakes
        elif task == 'glacial':
            self.target_dict = get_glacial_dir(f'{path}/glacial_catalog.txt')
            dirs = glacial
        train_dirs, test_dirs = train_test_split(dirs, test_size=.2, random_state=RANDOM_STATE)
        train_dirs, val_dirs = train_test_split(train_dirs, test_size=.3, random_state=RANDOM_STATE)
        self.train_dirs, self.val_dirs, self.test_dirs = train_dirs, val_dirs, test_dirs

    def __getitem__(self, index):
        if self.mode == 'test':
            path = self.test_dirs[index]
        elif self.mode == 'train':
            path = self.train_dirs[index]
        else:
            path = self.val_dirs[index]
        data = torch.FloatTensor(np.transpose(mpimg.imread(path).copy(), [2, 0, 1]))
        words = path.split('/')
        date = words[3].split('_')[1].split('.')[0]
        target = self.target_dict[date]
        target = torch.FloatTensor([target])
        return data, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_dirs) 
        elif self.mode == 'eval':
            return len(self.val_dirs)
        else:
            return len(self.train_dirs)

def SeismicRegDataLoader(mode, batch_size, path, task):
    dataset = SeismicRegDataset(mode, path, task)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=(False if mode=='test' else True),
        batch_size=batch_size,
        pin_memory=True,
    )
    return dataloader