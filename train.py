__author__ = "Jingbo Cheng"

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet152
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader.dataloader_classifier import SeismicDataLoader
from config.config import model, path

def 

def check_accuracy(dataloader: DataLoader,
                   model: nn.Module,
):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / float(num_samples)
        print(f'Got {num_correct}/{num_samples} correct {acc * 100}%')
    return acc
    

def train(model: nn.Module, 
          optimizer: optim.Optimizer,
          epoches: int,
):
    model.train()
    for epoch in tqdm(range(epoches)):
        loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            loss += _loss.item()
        train_acc = check_accuracy(train_dataloader, model)
        eval_acc = check_accuracy(eval_dataloader, model)
        writer.add_scalar('eval_acc', eval_acc, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
    
    

if __name__ == '__main__':
    batch_size = model['batch_size']
    device = model['device']
    epoches = model['epoches']
    tensorboard_path = path['tensorboard_path']
    model = resnet152()
    model = model.to(device)
    optimizer = optim.Adam()
    writer = SummaryWriter(tensorboard_path)
    train_dataloader = SeismicDataLoader('train', batch_size)
    eval_dataloader = SeismicDataLoader('eval', batch_size)
    train(model, optimizer, epoches)