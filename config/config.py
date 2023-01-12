import torch

path = {
    'model_path': './saving/model/',
    'image_path': './saving/image',
    'tensorboard_path': '/saving/logs'
}

model_dict = {
    'device': torch.device('cuda'),
    'batch_size': 16,
    'epoches': 10,
}

opt = {
    'alpha': 1e-2,
    'beta1': 0.9,
    'beta2': 0.9,
    'epsilon': 1e-8
}
