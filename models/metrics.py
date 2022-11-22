import torch
from ignite.metrics import SSIM, PSNR

def mse(fake, real):
    return torch.sum((fake-real)**2) / torch.numel(real)

def psnr(fake, real):
    metric = PSNR(data_range=255)
    metric.attach(default_evaluator, 'psnr')
    return default_evaluator.run([[fake, real]])

def ssim(fake, real):
    metric = SSIM(data_range=255)
    metric.attach(default_evaluator, 'ssim')
    return default_evaluator.run([[fake, real]])



