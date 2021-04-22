import math
import os
from os.path import join
from argparse import ArgumentParser
from typing import Callable, Optional, Any
import glob
import sys
import random
import tempfile
import zipfile
import uuid
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torchvision.datasets.folder import default_loader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.core.datamodule import LightningDataModule
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVEvalDataTransform, SwAVTrainDataTransform)
from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization)
from pl_bolts.models.self_supervised import SwAV

from pytorch_models.utils import save_json, s3_sync, batch_submit, S3SyncCallback


class CustomDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.img_paths = glob.glob(
            join(data_dir, '**', split, 'img', '*.png'), recursive=True)

        self.img_paths.sort()
        random.seed(1234)
        random.shuffle(self.img_paths)

    def __getitem__(self, ind):
        img_path = self.img_paths[ind]
        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.img_paths)


class CustomDataModule(LightningDataModule):
    name = 'custom'

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = None

    def prepare_data(self) -> None:
        ds = CustomDataset(self.data_dir, split='train')
        self.num_samples = len(ds)
        self.sz = ds[0][0].size

    def size(self):
        return self.sz

    def train_dataloader(self) -> DataLoader:
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        dataset = CustomDataset(self.data_dir, split='train', transform=transforms)
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        transforms = self.train_transform() if self.val_transforms is None else self.val_transforms
        dataset = CustomDataset(self.data_dir, split='valid', transform=transforms)
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


def save_backbone(swav_model, output_path):
    state_dict = swav_model.model.state_dict()
    keys_to_remove = [
        key for key in state_dict
        if key.startswith('projection_head') or key.startswith('prototypes')]

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key not in keys_to_remove:
            new_key = 'backbone.' + key
            new_state_dict[new_key] = value

    torch.save(new_state_dict, output_path)


class DataPlotter():
    def __init__(self, dm):
        self.dm = dm

    def plot_sample(self, ax, x):
        x = x.permute([1, 2, 0])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        x = (x * std) + mean
        ax.imshow(x.numpy())

    def plot_batch(self, x):
        x = x[1]
        batch_sz = x.shape[0]
        nrows = ncols = math.ceil(math.sqrt(batch_sz))
        fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(2*nrows, 2*ncols))
        axs = axs.flatten()
        for ind, ax in enumerate(axs):
            if ind < batch_sz:
                ax = axs[ind]
                self.plot_sample(ax, x[ind])
            ax.axis('off')

        fig.tight_layout()
        return fig

    def plot_dl(self, dl, out_path=None):
        x, _ = next(iter(dl))
        fig = self.plot_batch(x)
        if out_path is not None:
            fig.savefig(out_path)

    def plot_dataloaders(self, out_dir=None):
        os.makedirs(out_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            out_path = None if out_dir is None else join(out_dir, f'{split}.png')
            dl = getattr(self.dm, f'{split}_dataloader')()
            self.plot_dl(dl, out_path=out_path)

def main(args):
    os.makedirs('/opt/data/tmp/', exist_ok=True)
    with tempfile.TemporaryDirectory(dir='/opt/data/tmp/') as tmp_dir:
        root_dir = args.default_root_dir
        orig_root_dir = root_dir
        if root_dir.startswith('s3://'):
            new_root_dir = join(tmp_dir, 'output')
            s3_sync(root_dir, new_root_dir)
            root_dir = new_root_dir
            args.default_root_dir = root_dir
        else:
            os.makedirs(root_dir, exist_ok=True)

        data_dir = args.data_dir
        if data_dir.startswith('s3://'):
            new_data_dir = join(tmp_dir, 'data')
            s3_sync(data_dir, new_data_dir)
            data_dir = new_data_dir
        proc_data_dir = join(tmp_dir, 'processed_data')
        os.makedirs(proc_data_dir, exist_ok=True)
        zip_uris = glob.glob(join(data_dir, '*.zip'))
        for zip_uri in zip_uris:
            with zipfile.ZipFile(zip_uri, 'r') as zipf:
                _proc_data_dir = join(proc_data_dir, str(uuid.uuid4()))
                zipf.extractall(_proc_data_dir)
        args.data_dir = proc_data_dir

        args.gpus = torch.cuda.device_count()
        args.dataset = 'imagenet'
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.size_crops = [224, 96]
        args.nmb_crops = [2, 6]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1., 0.14]
        args.gaussian_blur = True
        args.jitter_strength = 1.

        args.lars_wrapper = True
        args.nmb_prototypes = 3000
        args.online_ft = False

        dm = CustomDataModule(
            data_dir=args.data_dir, batch_size=args.batch_size,
            num_workers=args.num_workers)
        dm.prepare_data()

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        dm.train_transforms = SwAVTrainDataTransform(
            normalize=normalization,
            size_crops=args.size_crops,
            nmb_crops=args.nmb_crops,
            min_scale_crops=args.min_scale_crops,
            max_scale_crops=args.max_scale_crops,
            gaussian_blur=args.gaussian_blur,
            jitter_strength=args.jitter_strength
        )

        dm.val_transforms = SwAVEvalDataTransform(
            normalize=normalization,
            size_crops=args.size_crops,
            nmb_crops=args.nmb_crops,
            min_scale_crops=args.min_scale_crops,
            max_scale_crops=args.max_scale_crops,
            gaussian_blur=args.gaussian_blur,
            jitter_strength=args.jitter_strength
        )

        plotter = DataPlotter(dm)
        plot_dir = join(root_dir, 'dataloaders')
        plotter.plot_dataloaders(plot_dir)

        init_weights = args.init_weights
        if init_weights:
            model = SwAV.load_from_checkpoint(init_weights, strict=True, **args.__dict__)
        else:
            model = SwAV(**args.__dict__)

        csv_logger = pl.loggers.CSVLogger(root_dir, 'csv')
        tb_logger = pl.loggers.TensorBoardLogger(root_dir, 'tb')
        loggers = [csv_logger, tb_logger]
        checkpoint_callback = ModelCheckpoint(root_dir, 'last_epoch')
        lr_monitor_callback = LearningRateMonitor()
        callbacks = [checkpoint_callback, lr_monitor_callback]
        if orig_root_dir.startswith('s3://'):
            callbacks.append(
                S3SyncCallback(root_dir, orig_root_dir))

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            max_steps=None if args.max_steps == -1 else args.max_steps,
            gpus=args.gpus,
            num_nodes=args.num_nodes,
            distributed_backend='ddp' if args.gpus > 1 else None,
            sync_batchnorm=True if args.gpus > 1 else False,
            precision=32 if args.fp32 else 16,
            fast_dev_run=args.fast_dev_run,
            default_root_dir=args.default_root_dir,
            callbacks=callbacks,
            logger=loggers,
            progress_bar_refresh_rate=args.progress_bar_refresh_rate
        )

        trainer.fit(model, datamodule=dm)

        backbone_path = join(root_dir, 'backbone.pth')
        save_backbone(model, backbone_path)

        if orig_root_dir.startswith('s3://'):
            s3_sync(root_dir, orig_root_dir)


def get_arg_parser():
    parser = ArgumentParser()
    parser = SwAV.add_model_specific_args(parser)
    parser.add_argument('--default_root_dir', type=str, default=os.getcwd())
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1)
    parser.set_defaults(hide_progress_bar=False)
    return parser


if __name__ == '__main__':
    command = sys.argv
    if '--aws_batch' in command:
        command.remove('--aws_batch')
        command.insert(0, 'python')
        command.extend(['--progress_bar_refresh_rate', '100'])
        batch_submit(command)
        exit()

    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
