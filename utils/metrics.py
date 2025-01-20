import torch

def psnr(gt_image, target_image):
    return 10.0 * torch.log10(1.0 / torch.mean((gt_image - target_image) ** 2))