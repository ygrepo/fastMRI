import numpy as np
import torch
import torch.nn as nn

from common.subsample import create_mask_for_mask_type
from data import transforms



class NeumannNetwork(nn.Module):

    def __init__(self, reg_network=None, hparams=None):
        super(NeumannNetwork, self).__init__()
        self.hparams = hparams
        self.mask_func = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                                   self.hparams.accelerations)
        self.reg_network = reg_network
        self.n_blocks = hparams.n_blocks
        self.eta = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.preconditioned = False

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
        masked_kspace, _ = transforms.apply_mask(kspace, self.mask_func, self.hparams.seed)
        if not torch.is_tensor(masked_kspace):
            kspace_tensor = transforms.to_tensor(masked_kspace)
        else:
            kspace_tensor = masked_kspace
        image = transforms.ifft2(kspace_tensor)
        image, image_abs = self.resize(image)
        return image, image_abs

    def X_operator(self, img):
        kspace = 1j * (img[..., 1].detach().numpy())
        kspace += (img[..., 0].detach().numpy())
        kspace = transforms.to_tensor(kspace)
        return transforms.fft2(kspace)

    def gramian_helper(self, img):
        kspace = self.X_operator(img)
        new_img, new_img_abs = self.forward_adjoint_helper(kspace)
        return new_img, new_img_abs

    def forward(self, kspace):

        runner_img, runner_img_abs = self.forward_adjoint_helper(kspace)
        neumann_sum = runner_img

        # unrolled gradient iterations
        for i in range(self.n_blocks):
            print(f"\nNNeumann Iteration:{i}")
            new_img, new_img_abs = self.gramian_helper(runner_img)
            linear_component = runner_img - self.eta * new_img
            learned_component = -self.reg_network(runner_img_abs)
            learned_component = transforms.to_tensor(np.fft.fft2(learned_component.detach().numpy())).float()

            runner_img = linear_component + learned_component
            neumann_sum = neumann_sum + runner_img

        return transforms.complex_abs(neumann_sum)

    def parameters(self):
        return list([self.eta]) + list(self.reg_network.parameters())
        # return list([self.eta, self.lambda_param]) + list(self.reg_network.parameters())
