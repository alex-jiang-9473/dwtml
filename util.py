import numpy as np
import torch
from torch._C import dtype
from typing import Dict
from math import log10, sqrt

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

def to_coordinates_and_coeffs_features(coeffs):
    """Converts DWT coefficients to a set of coordinates and features.

    Args:
        coeffs (torch.Tensor): Shape (1, height, width) for single channel.
    """
    # Create a meshgrid of coordinates
    h, w = coeffs.shape[1], coeffs.shape[2]
    y = torch.linspace(-1, 1, h)
    x = torch.linspace(-1, 1, w)
    y, x = torch.meshgrid(y, x, indexing='ij')
    
    # Stack coordinates into (num_points, 2) tensor
    coordinates = torch.stack([x.flatten(), y.flatten()], dim=1)
    
    # Reshape coefficients to (num_points, 1)
    features = coeffs.reshape(-1, 1)
    
    return coordinates, features

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels

def psnr(img1, img2):
    """Calculates PSNR between two tensors.

    Args:
        img1 (torch.Tensor): First tensor
        img2 (torch.Tensor): Second tensor
    """
    mse = (img1 - img2).detach().pow(2).mean()
    if mse == 0:
        return float('inf')
    max_val = max(img1.max(), img2.max())
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true coefficients and reconstructed coefficients.
    For wavelet coefficients, we don't need to clamp to [0,1] range.

    Args:
        img (torch.Tensor): Ground truth coefficients (2D or 3D tensor)
        img_recon (torch.Tensor): Reconstructed coefficients (2D or 3D tensor)
    """
    # Ensure same shape
    if img.dim() != img_recon.dim():
        if img.dim() == 3 and img_recon.dim() == 2:
            img = img.squeeze(0)
        elif img.dim() == 2 and img_recon.dim() == 3:
            img_recon = img_recon.squeeze(0)
    return psnr(img, img_recon)


def mean(list_):
    return np.mean(list_)


# Calculate PSNR
def calc_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
