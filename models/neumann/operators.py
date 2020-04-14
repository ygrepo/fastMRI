import torch
from torch import nn

from data import transforms


class ForwardOperator(nn.Module):
    def __init__(self, resolution, which_challenge, mask_func, use_seed=False):

        super(ForwardOperator, self).__init__()
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, kspace, target, fname):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        #kspace = transforms.to_tensor(kspace)
        # Apply mask
        print("Applying mask")
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image to given resolution if larger
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])
        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        return image


class GramianModel(nn.Module):

    def __init__(self, forward_operator):
        super(GramianModel, self).__init__()
        self.forward_operator = forward_operator

    def forward(self, image, target, fname):
        volume_kspace = torch.rfft(image, 2)
        #volume_kspace = transforms.fft2(image)
        return self.forward_operator(volume_kspace, target, fname)
