import torch
from skimage.metrics import structural_similarity as skimg_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def mse(fake, real):
    MSE = torch.sum((fake-real)**2) / torch.numel(real)
    return MSE

def psnr(fake, real):
    MSE = mse(fake, real)
    if MSE == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = (20 * torch.log10(PIXEL_MAX / torch.sqrt(MSE)))
    return PSNR

def ssim(fake, real):
    fake, real = fake.squeeze(0), real.squeeze(0)
    SSIM = skimg_ssim(real.cpu().numpy(), fake.cpu().numpy(), data_range=255, channel_axis=0)
    return SSIM

def lpimp(fake, real, net_type):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(fake.device)
    LPIMPS = lpips(fake, real)
    return LPIMPS




