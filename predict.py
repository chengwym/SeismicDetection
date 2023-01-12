import torch
from torchvision.models import resnet152

from config.config import path
from train import check_accuracy
from data_loader.dataloader_classifier import SeismicDataLoader


if __name__ == '__main__':
    model_path = path['model_path']
    test_dataloader = SeismicDataLoader('test')
    model = resnet152()
    model.load_state_dict(torch.load(f'{model_path}/model.pt'))
    check_accuracy(test_dataloader, model, True)