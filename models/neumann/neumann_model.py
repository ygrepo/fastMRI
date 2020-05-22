import torch
import torch.nn as nn

from common.subsample import create_mask_for_mask_type
from models.neumann.operators import forward_adjoint_helper, gramian_helper


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

    def forward(self, kspace):

        runner_img, runner_img_abs = forward_adjoint_helper(self.device, self.hparams, self.mask_func, kspace,
                                                            target=None)
        runner_img_abs = runner_img_abs.unsqueeze(1)
        runner_img_abs = self.eta * runner_img_abs

        # unrolled gradient iterations
        for i in range(self.n_blocks):
            # print(f"\nNNeumann Iteration:{i}")
            tmp = torch.rfft(runner_img_abs, 1, onesided=False).float().squeeze()
            gramian_img, gramian_img_abs = gramian_helper(self.device, self.hparams, self.mask_func, tmp)
            gramian_img_abs = gramian_img_abs.unsqueeze(1)
            linear_component = runner_img_abs - self.eta * gramian_img_abs
            learned_component = self.reg_network(runner_img_abs)

            runner_img_abs = linear_component - learned_component

        return runner_img_abs

    def parameters(self):
        return list([self.eta]) + list(self.reg_network.parameters())
        # return list([self.eta, self.lambda_param]) + list(self.reg_network.parameters())
