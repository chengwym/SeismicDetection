from train import check_accuracy
from data_loader.dataloader_classifier import SeismicDataLoader


if __name__ == '__main__':
    test_dataloader = SeismicDataLoader('test')
    
    check_accuracy(test_dataloader, model)