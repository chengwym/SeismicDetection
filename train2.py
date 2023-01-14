__author__ = "Tianyi Zhang"

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet152
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from config.constant import RANDOM_STATE
from data_loader.dataloader_predict_offset import SeismicDataLoader
from config.config import model_dict, path, opt

def valid_part(dataloader: DataLoader,
               model: nn.Module,
):
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, y, path in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _loss = F.mse_loss(y, scores)
            loss += _loss.item()
    return loss

def check_accuracy(dataloader: DataLoader,
                   model: nn.Module,
                   print_result: bool = True,
                   device=torch.device('cuda'),
):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y, path in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            print(path, f"true distance={y}", f"predict distance={scores}")
            num_correct += 1 if abs(scores - y) <= 1 else 0
            num_samples += 1
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
        for x, y, path in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _loss = F.mse_loss(scores, y)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            loss += _loss.item()
        global_step.append(epoch)
        train_acc = check_accuracy(train_dataloader, model, False)
        eval_acc = check_accuracy(eval_dataloader, model, False)
        acc_train.append(train_acc)
        acc_valid.append(eval_acc)
        eval_loss = valid_part(eval_dataloader, model)
        train_loss = loss
        loss_train.append(train_loss)
        loss_valid.append(eval_loss)


if __name__ == '__main__':
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

    model = resnet152(num_classes=1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    global_step = []
    acc_train = []
    acc_valid = []
    loss_train = []
    loss_valid = []

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha, betas=(beta1, beta2), eps=epsilon)
    train_dataloader = SeismicDataLoader('train', batch_size, './data')
    eval_dataloader = SeismicDataLoader('eval', batch_size, './data')
    test_dataloader = SeismicDataLoader('test', batch_size, './data')
    print('datas are ready')
    train(model, optimizer, epoches)
    print('the train part has been done')

    plt.plot(global_step, acc_train, 'b', label='train acc')
    plt.plot(global_step, acc_valid, 'r', label='valid acc')
    plt.title('Train and Valid Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'{image_path}/acc.png')

    plt.figure()
    plt.plot(global_step, loss_train, 'b', label='train loss')
    plt.plot(global_step, loss_valid, 'r', label='valid loss')
    plt.title('Train and Valid Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{image_path}/loss.png')

    torch.save(model.state_dict(), f'{model_path}/model2.pt')
    check_accuracy(test_dataloader, model, True)