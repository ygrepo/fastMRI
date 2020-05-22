# numpy
import numpy as np
# torch
import torch

# fastMRI libraries
from data import transforms


def apply_mask(device, mask_func, kspace, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        kspace (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(kspace.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    mask = mask.to(device)
    kspace = kspace.to(device)
    return torch.where(mask == 0, torch.Tensor([0]).to(device), kspace), mask


def resize(hparams, image, target):
    smallest_width = min(hparams.resolution, image.shape[-2])
    smallest_height = min(hparams.resolution, image.shape[-3])
    if target is not None:
        smallest_width = min(smallest_width, target.shape[-1])
        smallest_height = min(smallest_height, target.shape[-2])
    crop_size = (smallest_height, smallest_width)
    image = transforms.complex_center_crop(image, crop_size)
    # Absolute value
    image_abs = transforms.complex_abs(image)
    # Apply Root-Sum-of-Squares if multicoil data
    if hparams.challenge == "multicoil":
        image_abs = transforms.root_sum_of_squares(image_abs)
    # Normalize input
    image_abs, mean, std = transforms.normalize_instance(image_abs, eps=1e-11)
    image_abs = image_abs.clamp(-6, 6)

    # Normalize target
    if target is not None:
        target = transforms.to_tensor(target)
        target = transforms.center_crop(target, crop_size)
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
    else:
        target = torch.Tensor([0])
    return image, image_abs, target, mean, std


def forward_adjoint_helper(device, hparams, mask_func, kspace, target=None):
    masked_kspace, _ = apply_mask(device, mask_func, kspace, hparams.seed)
    if not torch.is_tensor(masked_kspace):
        kspace_tensor = transforms.to_tensor(masked_kspace)
    else:
        kspace_tensor = masked_kspace
    image = transforms.ifft2(kspace_tensor)
    image, image_abs, _, _, _ = resize(hparams, image, target)
    return image, image_abs


def X_operator(img):
    return transforms.fft2(img)


def gramian_helper(device, hparams, mask_func, img, target=None):
    kspace = X_operator(img)
    new_img, new_img_abs = forward_adjoint_helper(device, hparams, mask_func, kspace, target)
    return new_img, new_img_abs

def X_I_operator(img):
    return transforms.fft2(img)


def gramian_I_helper(device, hparams, mask_func, img, target=None):
    return img, transforms.complex_abs(img)
