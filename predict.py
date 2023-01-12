import torch
import torch.nn as nn
from torchvision.models import resnet152

from config.config import path, model_dict
from train import check_accuracy
from data_loader.dataloader_classifier import SeismicDataLoader


if __name__ == '__main__':
    model_path = path['model_path']
    batch_size = model_dict['batch_size']
    device = model_dict['device']
    test_dataloader = SeismicDataLoader('test', batch_size, './data')
    model = resnet152(num_classes=2)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=device))  
    model = model.to(device)
    check_accuracy(test_dataloader, model, True)