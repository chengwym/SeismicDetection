__author__ = "Jingbo Cheng"

import sys
import torch
import torch.nn as nn
from torchvision.models import resnet152

from config.config import path, model_dict
from supervised_train import check_accuracy_classifier, check_accuracy_regress
from data_loader.dataloader_classifier import SeismicDataLoader


if __name__ == '__main__':
    parmas = sys.argv
    
    model_path = path['model_path']
    batch_size = model_dict['batch_size']
    device = model_dict['device']
    test_dataloader = SeismicDataLoader('test', batch_size, './data')

    if parmas[1] == 'binary':
        model = resnet152(num_classes=2)
        check_accuracy = check_accuracy_classifier
    elif parmas[1] == 'reg':
        model = resnet152(num_classes=1)
        check_accuracy = check_accuracy_regress
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f'{model_path}/{parmas[1]}_model.pt', map_location=device))  
    model = model.to(device)
    check_accuracy(test_dataloader, model)