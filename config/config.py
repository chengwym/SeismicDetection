import torch

path = {
    'model_path': './saving/model/',
    'image_path': './saving/image',
    'tensorboard_path': '/saving/logs'
}

model_dict = {
    'device': torch.device('cuda'),
    'batch_size': 128,
    'epoches': 1000,
}

opt = {
    'alpha': 9e-7,
    'beta1': 0.9,
    'beta2': 0.98,
    'epsilon': 1e-10,
    'weight_decay': 9e-9
}