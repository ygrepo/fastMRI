"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from torch.nn import functional as F
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from common import evaluate
from common.args import Args
from common.utils import save_reconstructions
from data import transforms
from data.mri_data import SliceData
from models.neumann.neumann_model import NeumannNetwork
from models.unet.unet_model import UnetModel

def default_collate(batch):
    """
    Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512

    We need our own collate function that wraps things up (imge, mask, label).

    In this setup,  batch is a list of tuples (the result of calling: img, mask, label = Dataset[i].
    The output of this function is four elements:
        . data: a pytorch tensor of size (batch_size, c, h, w) of float32 . Each sample is a tensor of shape (c, h_,
        w_) that represents a cropped patch from an image (or the entire image) where: c is the depth of the patches (
        since they are RGB, so c=3),  h is the height of the patch, and w_ is the its width.
        . mask: a list of pytorch tensors of size (batch_size, 1, h, w) full of 1 and 0. The mask of the ENTIRE image (no
        cropping is performed). Images does not have the same size, and the same thing goes for the masks. Therefore,
        we can't put the masks in one tensor.
        . target: a vector (pytorch tensor) of length batch_size of type torch.LongTensor containing the image-level
        labels.
    :param batch: list of tuples (img, mask, label)
    :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
    """

    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    kspaces = [item[2] for item in batch]
    means = [item[3] for item in batch]
    stds = [item[4] for item in batch]
    fnames = [item[5] for item in batch]
    slice_infos = [item[6] for item in batch]

    # data = torch.stack([item[0] for item in batch])
    # mask = [item[1] for item in batch]  # each element is of size (1, h*, w*). where (h*, w*) changes from mask to another.
    # target = torch.LongTensor([item[2] for item in batch])  # image labels.

    return images, targets, kspaces, means, stds, fnames, slice_infos

def resize(image, target, resolution, challenge):
    smallest_width = min(resolution, image.shape[-2])
    smallest_height = min(resolution, image.shape[-3])
    if target is not None:
        smallest_width = min(smallest_width, target.shape[-1])
        smallest_height = min(smallest_height, target.shape[-2])
    crop_size = (smallest_height, smallest_width)
    image = transforms.complex_center_crop(image, crop_size)
    # Absolute value
    image = transforms.complex_abs(image)
    # Apply Root-Sum-of-Squares if multicoil data
    if challenge == 'multicoil':
        image = transforms.root_sum_of_squares(image)
    # Normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    # Normalize target
    if target is not None:
        target = transforms.to_tensor(target)
        target = transforms.center_crop(target, crop_size)
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
    else:
        target = torch.Tensor([0])
    return image, target, mean, std


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice_info):
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
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        if self.mask_func:
            print("Applying mask")
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image to given resolution if larger
        image, target, mean, std = resize(image, target, self.resolution, self.which_challenge)
        #print(f"image:{image.shape}, target:{target.shape}, mean:{mean}, {std}, {fname}, {slice_info}")
        return image, target, kspace, mean, std, fname, slice_info


class NeumannMRIModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # reg_model = REDNet20(num_features= self.hparams.resolution)
        self.hparams = hparams
        reg_model = UnetModel(
            in_chans=1,
            out_chans=1,
            chans=hparams.num_chans,
            num_pool_layers=hparams.num_pools,
            drop_prob=hparams.drop_prob
        )
        self.neumann = NeumannNetwork(reg_network=reg_model, hparams=hparams)

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or self.hparams.sample_rate
        dataset = SliceData(
            root=self.hparams.data_path / f'{self.hparams.challenge}_{data_partition}',
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=self.hparams.challenge
        )
        sampler = RandomSampler(dataset)
        # sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
            sampler=sampler,
            #collate_fn=default_collate
        )

    def forward(self, input):
        return self.neumann(input.unsqueeze(1)).squeeze(1)

    def train_dataloader(self):
        return self._create_data_loader(self.train_data_transform(), data_partition='train')

    def training_step(self, batch, batch_idx):
        #print(f"Training step, batch_idx:{batch_idx}")
        image, target, kspace, mean, std, fname, slice = batch
        output = self.forward(kspace)
        loss = F.mse_loss(output, target)
        logs = {"loss": loss.item()}
        print(f"loss:{loss}")
        return dict(loss=loss, log=logs)

    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition='val')

    def validation_step(self, batch, batch_idx):
        #print(f"Validation step, batch_idx:{batch_idx}")
        image, target, kspace, mean, std, fname, slice = batch
        output = self.forward(kspace)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            'fname': fname,
            'slice': slice,
            'output': (output * std + mean).cpu().numpy(),
            'target': (target * std + mean).cpu().numpy(),
            'val_loss': F.mse_loss(output, target),
        }

    def _evaluate(self, val_logs):
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in val_logs:
            losses.append(log["val_loss"].cpu().numpy())
            for i, (fname, slice) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname].append((slice, log["output"][i]))
                targets[fname].append((slice, log["target"][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics["nmse"].append(evaluate.nmse(target, output))
            metrics["ssim"].append(evaluate.ssim(target, output))
            metrics["psnr"].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        self.logger.log_metrics(metrics)
        return dict(log=metrics, **metrics)

    def _visualize(self, val_logs):
        def _normalize(image):
            image = image[np.newaxis]
            image -= image.min()
            return image / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(torch.Tensor(image), nrow=4, pad_value=1)
            grid_path = Path(self.hparams.exp_dir) / self.hparams.exp / "image_validation_step"
            grid_path.mkdir(parents=True, exist_ok=True)
            grid_path = grid_path / tag
            grid_np = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            grid_pil = Image.fromarray(grid_np)
            try:
                grid_pil.save(grid_path, format="PNG")
            except ValueError as e:
                print(e)

        # Only process first size to simplify visualization.
        visualize_size = val_logs[0]['output'].shape
        val_logs = [x for x in val_logs if x['output'].shape == visualize_size]
        num_logs = len(val_logs)
        num_viz_images = 16
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets = [], []
        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_logs[i]['output'][0]))
            targets.append(_normalize(val_logs[i]['target'][0]))
        outputs = np.stack(outputs)
        targets = np.stack(targets)
        _save_image(targets, 'Target')
        _save_image(outputs, 'Reconstruction')
        _save_image(np.abs(targets - outputs), 'Error')

    def validation_epoch_end(self, val_logs):
        self._visualize(val_logs)
        return self._evaluate(val_logs)

    def test_dataloader(self):
        return self._create_data_loader(self.test_data_transform(), data_partition='test', sample_rate=1.)

    def test_step(self, batch, batch_idx):
        image, target, kspace, mean, std, fname, slice = batch
        output = self.forward(kspace)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            'fname': fname,
            'slice': slice,
            'output': (output * std + mean).cpu().numpy(),
        }

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(list)
        for log in test_logs:
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
        save_reconstructions(outputs, self.hparams.exp_dir / self.hparams.exp / 'reconstructions')
        return dict()

    def configure_optimizers(self):
        optim = RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_step_size, self.hparams.lr_gamma)
        return [optim], [scheduler]

    def train_data_transform(self):
        return DataTransform(self.hparams.resolution, self.hparams.challenge)

    def val_data_transform(self):
        return DataTransform(self.hparams.resolution, self.hparams.challenge)

    def test_data_transform(self):
        return DataTransform(self.hparams.resolution, self.hparams.challenge)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
        parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
        parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
        parser.add_argument('--n_blocks', type=int, default=1, help='Number of Neumann Network blocks')
        parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--lr-step-size', type=int, default=40,
                            help='Period of learning rate decay')
        parser.add_argument('--lr-gamma', type=float, default=0.1,
                            help='Multiplicative factor of learning rate decay')
        parser.add_argument('--weight-decay', type=float, default=0.,
                            help='Strength of weight decay regularization')
        return parser


def create_trainer(args, logger):
    return Trainer(
        logger=logger,
        default_save_path=args.exp_dir,
        checkpoint_callback=True,
        max_nb_epochs=args.num_epochs,
        gpus=args.gpus,
        #distributed_backend="ddp",
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        early_stop_callback=False
    )


def main(args):
    if args.mode == 'train':
        load_version = 0 if args.resume else None
        logger = TestTubeLogger(save_dir=args.exp_dir, name=args.exp, version=load_version)
        trainer = create_trainer(args, logger)
        model = NeumannMRIModel(args)
        trainer.fit(model)
    else:  # args.mode == 'test'
        assert args.checkpoint is not None
        trainer = create_trainer(args, logger=False)
        model = NeumannMRIModel.load_from_checkpoint(str(args.checkpoint))
        model.hparams.sample_rate = 1.
        trainer.test(model)


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--exp-dir', type=pathlib.Path, default='experiments',
                        help='Path where model and results should be saved')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. ')
    parser = NeumannMRIModel.add_model_specific_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
