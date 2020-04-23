from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn

from common.subsample import create_mask_for_mask_type
from data import transforms
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
        self.id_sequ = 0

    def forward(self, kspace):

        runner_img, runner_img_abs = forward_adjoint_helper(self.device, self.hparams, self.mask_func, kspace,
                                                            target=None)
        self.save_image(runner_img_abs, "init_runner")

        neumann_sum = runner_img

        # unrolled gradient iterations
        for i in range(self.n_blocks):
            # print(f"\nNNeumann Iteration:{i}")
            gramian_img, gramian_img_abs = gramian_helper(self.device, self.hparams, self.mask_func, runner_img)
            self.save_image(gramian_img_abs, "gramian")
            linear_component = runner_img - self.eta * gramian_img
            self.save_image(linear_component, "linear_component")
            gramian_img_abs = gramian_img_abs.unsqueeze(0).permute((1, 0, 2, 3))
            learned_component = -self.reg_network(gramian_img_abs)
            self.save_image(learned_component, "learned_component")
            learned_component = torch.rfft(learned_component, 2, onesided=False).float().squeeze()

            runner_img = linear_component + learned_component
            self.save_image(transforms.complex_abs(runner_img), "runner_img")
            neumann_sum = neumann_sum + runner_img
            self.save_image(transforms.complex_abs(neumann_sum), "neumann_img")

        return transforms.complex_abs(neumann_sum)

    def parameters(self):
        return list([self.eta]) + list(self.reg_network.parameters())
        # return list([self.eta, self.lambda_param]) + list(self.reg_network.parameters())


    def save_image(self, img, tag):
        # img_path = Path(self.hparams.exp_dir) / self.hparams.exp / "image_debugging"
        # img_path.mkdir(parents=True, exist_ok=True)
        # img_path = img_path / (tag + str(self.id_sequ))
        img = img.squeeze(0)
        img_np = img.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        print(img_np.shape, img_np.dtype)
        img_pil = Image.fromarray(img_np.astype('uint8'))
        # try:
        #     img_pil.save(img_path, format="PNG")
        # except ValueError as e:
        #     print(e)
        # self.id_sequ += 1
