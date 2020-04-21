"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from common import evaluate
from common.utils import save_reconstructions
from data.mri_data import SliceData


class MRIModel(pl.LightningModule):
    """
    Abstract super class for Deep Learning based reconstruction models.
    This is a subclass of the LightningModule class from pytorch_lightning, with
    some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

    To implement a new reconstruction model, inherit from this class and implement the
    following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation and testing respectively
        - configure_optimizers:
            Create and return the optimizers
    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

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
            pin_memory=False,
            sampler=sampler,
        )

    def train_data_transform(self):
        raise NotImplementedError

    @pl.data_loader
    def train_dataloader(self):
        return self._create_data_loader(self.train_data_transform(), data_partition='train')

    def val_data_transform(self):
        raise NotImplementedError

    @pl.data_loader
    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition='val')

    def test_data_transform(self):
        raise NotImplementedError

    @pl.data_loader
    def test_dataloader(self):
        return self._create_data_loader(self.test_data_transform(), data_partition='test', sample_rate=1.)

    def _evaluate(self, val_logs):
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in val_logs:
            losses.append(log['val_loss'].cpu().numpy())
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        # save the metrics data
        metric_file_path = Path(self.hparams.exp_dir) / self.hparams.exp / "validation_metrics"
        metric_file_path.mkdir(parents=True, exist_ok=True)
        metric_file_path = metric_file_path / "metrics.csv"
        df = pd.DataFrame([metrics])
        if metric_file_path.exists():
            df.to_csv(metric_file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(metric_file_path, mode="w", header=True, index=False)
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

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(list)
        for log in test_logs:
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
        save_reconstructions(outputs, self.hparams.exp_dir / self.hparams.exp / 'reconstructions')
        return dict()
