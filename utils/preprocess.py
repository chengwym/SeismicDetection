import os
from sklearn.model_selection import train_test_split

def get_dataset():
    earthquakes = os.listdir('../data/earthquakes')
    earthquakes = [f'../data/earthquakes/{file}' for file in earthquakes]
    glacial = os.listdir('../data/glacial')
    glacial = [f'../data/glacial/{file}' for file in glacial]
    dirs = earthquakes + glacial
    train_dirs, test_dirs = train_test_split(dirs, test_size=.2)
    train_dirs, val_dirs = train_test_split(dirs, test_size=.3)
    return train_dirs, val_dirs, test_dirs