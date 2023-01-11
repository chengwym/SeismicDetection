import torch

path = {
    'model_path': './saving/model/',
    'image_path': './saving/image',
    'tensorboard_path': '/saving/logs'
}

model = {
    'device': torch.device('cuda'),
    'batch_size': 64,
    'epoches': 10,
}

