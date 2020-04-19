import numpy as np
import torch
import torch.nn as nn

from common.subsample import create_mask_for_mask_type
from data import transforms



class NeumannNetwork(nn.Module):

    def __init__(self, reg_network=None, hparams=None):
        super(NeumannNetwork, self).__init__()
        self.hparams = hparams
        self.device = "cuda"
        if hparams.gpus == 0:
            self.device = "cpu"
        self.mask_func = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                                   self.hparams.accelerations)
        self.reg_network = reg_network
        self.n_blocks = hparams.n_blocks
        self.eta = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.preconditioned = False

    def apply_mask(self, data, mask_func, seed=None):
        """
        Subsample given k-space by multiplying with a mask.

        Args:
            data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
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
        #print(data.shape)
        shape = np.array(data.shape)
        shape[:-3] = 1
        mask = mask_func(shape, seed)
        mask = mask.to(self.device)
        data = data.to(self.device)
        return torch.where(mask == 0, torch.Tensor([0]).to(self.device), data), mask

    def resize(self, image, challenge: str="singlecoil"):
        smallest_width = min(self.hparams.resolution, image.shape[-2])
        smallest_height = min(self.hparams.resolution, image.shape[-3])
        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)
        # Absolute value
        image_abs = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if challenge == "multicoil":
            image_abs= transforms.root_sum_of_squares(image_abs)
        # Normalize input
        image_abs, mean, std = transforms.normalize_instance(image_abs, eps=1e-11)
        image_abs = image_abs.clamp(-6, 6)
        return image, image_abs

    def forward_adjoint_helper(self, kspace):
        masked_kspace, _ = self.apply_mask(kspace, self.mask_func, self.hparams.seed)
        if not torch.is_tensor(masked_kspace):
            kspace_tensor = transforms.to_tensor(masked_kspace)
        else:
            kspace_tensor = masked_kspace
        image = transforms.ifft2(kspace_tensor)
        image, image_abs = self.resize(image)
        return image, image_abs

    def X_operator(self, img):
        #kspace = 1j * (img[..., 1].detach().cpu().numpy())
        #kspace += (img[..., 0].detach().cpu().numpy())
        #kspace = transforms.to_tensor(kspace)
        return transforms.fft2(img)

    def gramian_helper(self, img):
        kspace = self.X_operator(img)
        new_img, new_img_abs = self.forward_adjoint_helper(kspace)
        return new_img, new_img_abs

    def forward(self, kspace):

        runner_img, runner_img_abs = self.forward_adjoint_helper(kspace)
        neumann_sum = runner_img

        # unrolled gradient iterations
        for i in range(self.n_blocks):
            #print(f"\nNNeumann Iteration:{i}")
            new_img, new_img_abs = self.gramian_helper(runner_img)
            linear_component = runner_img - self.eta * new_img
            #print(runner_img_abs.shape)
            learned_component = -self.reg_network(new_img_abs)
            learned_component = torch.rfft(learned_component, 1, onesided=False).float()

            runner_img = linear_component + learned_component
            neumann_sum = neumann_sum + runner_img

        return transforms.complex_abs(neumann_sum)

    def parameters(self):
        return list([self.eta]) + list(self.reg_network.parameters())
        # return list([self.eta, self.lambda_param]) + list(self.reg_network.parameters())
