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

    def forward_adjoint_helper(self, kspace):
        masked_kspace, _ = transforms.apply_mask(kspace, self.mask_func, self.hparams.seed)
        if not torch.is_tensor(masked_kspace):
            kspace_tensor = transforms.to_tensor(masked_kspace)
        else:
            kspace_tensor = masked_kspace
        image = transforms.ifft2(kspace_tensor)
        image_abs = transforms.complex_abs(image)
        return image, image_abs

    def X_operator(self, img):
        kspace = 1j * (img[..., 1].numpy())
        kspace += (img[..., 0].numpy())
        kspace2 = transforms.to_tensor(kspace)
        new_kspace = transforms.fft2(kspace2)
        return new_kspace

    def gramian_helper(self, img):
        kspace = self.X_operator(img)
        new_img, new_img_abs = self.forward_adjoint_helper(kspace)
        return new_img, new_img_abs

    def forward(self, kspace):

        runner_img, runner_img_abs = self.forward_adjoint_helper(kspace)
        neumann_sum = runner_img

        # unrolled gradient iterations
        for i in range(self.n_blocks):
            new_img, new_img_abs = self.gramian_helper(runner_img)
            linear_component = runner_img - self.eta * new_img
            learned_component = -self.reg_network(runner_img_abs)
            learned_component = transforms.to_tensor(np.fft.fft2(learned_component.numpy())).float()

            runner_img = linear_component + learned_component
            neumann_sum = neumann_sum + runner_img

        return neumann_sum

    def parameters(self):
        return list([self.eta]) + list(self.reg_network.parameters())
        # return list([self.eta, self.lambda_param]) + list(self.reg_network.parameters())
