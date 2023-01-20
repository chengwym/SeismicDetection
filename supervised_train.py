__author__ = "Jingbo Cheng"

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import *
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys

from config.constant import RANDOM_STATE
from data_loader.dataloader_classifier import SeismicBinaryDataLoader
from data_loader.dataloader_regression import SeismicRegDataLoader
from config.config import model_dict, path, opt

def valid_part(dataloader: DataLoader,
               model: nn.Module,
               loss_function):
    """
    Giving the dataloader, model and loss funtion, compute the loss.
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _loss = loss_function(scores, y)
            loss += _loss.item()
    return loss

def check_accuracy_classifier(dataloader: DataLoader,
                              model: nn.Module,
                              print_result: bool = True,
                              device=torch.device('cuda')):
    """
    The accuracy of the task calssifier.
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, preds = scores.max(1)
            _, y_indice = y.max(1)
            num_correct += (preds == y_indice).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / float(num_samples)
        if print_result == True:
            print(f'Got {num_correct}/{num_samples} correct {acc * 100}%')
    return acc

def check_accuracy_regress(dataloader: DataLoader,
                           model: nn.Module,
                           print_result: bool = True,
                           task: str = 'earthquakes',
                           device=torch.device('cuda')):
    """
    The accuracy of the task regression.
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    threshold = 1 if task == 'earthquakes' else 0.1
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            num_correct += ((scores - y) <= threshold).sum()
            num_samples += y.size()[0]
        acc = float(num_correct) / float(num_samples)
        if print_result == True:
            print(f'Got {num_correct}/{num_samples} correct {acc * 100}%')
    return acc
            
    

def train(train_dataloader: DataLoader,
          eval_dataloader: DataLoader,
          model: nn.Module, 
          loss_function,
          optimizer: optim.Optimizer,
          epoches: int,
          task: str,
):
    """
    The train part of the supervised learning.
    """
    model.train()
    if task == 'reg':
        check_accuracy = check_accuracy_regress
    elif task == 'binary':
        check_accuracy = check_accuracy_classifier
    for epoch in tqdm(range(epoches)):
        loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _loss = loss_function(scores, y)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            loss += _loss.item()
        global_step.append(epoch)
        
        train_acc = check_accuracy(train_dataloader, model, False)
        eval_acc = check_accuracy(eval_dataloader, model, False)
        acc_train.append(train_acc)
        acc_valid.append(eval_acc)
        eval_loss = valid_part(eval_dataloader, model, loss_function)
        train_loss = loss
        loss_train.append(train_loss)
        loss_valid.append(eval_loss)


if __name__ == '__main__':
    params = sys.argv
    task = params[1]
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    
    batch_size = model_dict['batch_size']
    device = model_dict['device']
    epoches = model_dict['epoches']

    tensorboard_path = path['tensorboard_path']
    model_path = path['model_path']
    image_path = path['image_path']
    
    alpha = opt['alpha']
    beta1 = opt['beta1']
    beta2 = opt['beta2']
    epsilon = opt['epsilon']
    weight_decay = opt['weight_decay']

    if task == 'binary':
        model = resnet50(num_classes=2)
        loss_function = F.cross_entropy
        train_dataloader = SeismicBinaryDataLoader('train', batch_size, './data')
        eval_dataloader = SeismicBinaryDataLoader('eval', batch_size, './data')
        test_dataloader = SeismicBinaryDataLoader('test', batch_size, './data')
    elif task == 'reg':
        model = resnet50(num_classes=1)
        loss_function = F.mse_loss
        train_dataloader = SeismicRegDataLoader('train', batch_size, './data', params[2])
        eval_dataloader = SeismicRegDataLoader('eval', batch_size, './data', params[2])
        test_dataloader = SeismicRegDataLoader('test', batch_size, './data', params[2])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    global_step = []
    acc_train = []
    acc_valid = []
    loss_train = []
    loss_valid = []

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    
    print('datas are ready')
    train(train_dataloader, eval_dataloader, model, loss_function, optimizer, epoches, params[1])
    print('the train part has been done')

    plt.plot(global_step, acc_train, 'b', label='train acc')
    plt.plot(global_step, acc_valid, 'r', label='valid acc')
    plt.title('Train and Valid Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'{image_path}/{params[1]}_acc.png')

    plt.figure()
    plt.plot(global_step, loss_train, 'b', label='train loss')
    plt.plot(global_step, loss_valid, 'r', label='valid loss')
    plt.title('Train and Valid Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{image_path}/{params[1]}_loss.png')

    torch.save(model.state_dict(), f'{model_path}/model.pt')
    if params[1] == 'binary':      
        check_accuracy_classifier(test_dataloader, model)
    elif params[1] == 'reg':
        check_accuracy_regress(test_dataloader, model)