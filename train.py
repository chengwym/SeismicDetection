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
from config.config import model, path, opt

def valid_part(dataloader: DataLoader,
               model: nn.Module,
):
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _loss = F.cross_entropy(x, scores)
            loss += _loss.item()
    return loss

def check_accuracy(dataloader: DataLoader,
                   model: nn.Module,
                   print_result: bool = True,
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
        if print_result == True:
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
        train_acc = check_accuracy(train_dataloader, model, False)
        eval_acc = check_accuracy(eval_dataloader, model, False)
        writer.add_scalar('acc/eval_acc', eval_acc, global_step=epoch)
        writer.add_scalar('acc/train_acc', train_acc, global_step=epoch)
        eval_loss = valid_part(eval_dataloader, model)
        train_loss = loss
        writer.add_scalar('loss/train_loss', train_loss, global_step=epoch)
        writer.add_scalar('loss/eval_loss', eval_loss, global_step=epoch)
    

if __name__ == '__main__':
    batch_size = model['batch_size']
    device = model['device']
    epoches = model['epoches']

    tensorboard_path = path['tensorboard_path']
    model_path = path['model_path']
    
    alpha = opt['alpha']
    beta1 = opt['beta1']
    beta2 = opt['beta2']
    epsilon = opt['epsilon']

    model = resnet152()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha, betas=(beta1, beta2), eps=epsilon)
    writer = SummaryWriter(tensorboard_path)
    train_dataloader = SeismicDataLoader('train', batch_size)
    eval_dataloader = SeismicDataLoader('eval', batch_size)
    test_dataloader = SeismicDataLoader('test', batch_size)
    print('datas are ready')
    train(model, optimizer, epoches)
    print('the train part has been done')
    torch.save(model.state_dict(), model_path)
    check_accuracy(test_dataloader, model, True)