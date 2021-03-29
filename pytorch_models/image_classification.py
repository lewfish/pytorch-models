from argparse import ArgumentParser
import math
from os.path import join, isfile
import os
import tempfile
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule

from pytorch_models.utils import save_json, s3_sync, batch_submit, S3SyncCallback

class ImageClassification(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        model_cls = getattr(torchvision.models, self.hparams.backbone)
        model = model_cls(
            pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        # This model modification is needed for 32x32 images. It makes
        # accuracy go from about 85% to 93%. This should not be used if working with
        # large images!
        # Taken from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/notebooks/07-cifar10-baseline.ipynb
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def eval_step(self, batch, batch_idx, split):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log(f'{split}_loss', loss)
        self.log(f'{split}_acc', acc)

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        total_devices = max(1, self.hparams.gpus * self.hparams.num_nodes)
        train_batches = self.hparams.num_samples // total_devices
        train_steps = (self.hparams.max_epochs * train_batches) // self.hparams.accumulate_grad_batches

        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay)
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer, self.hparams.learning_rate, total_steps=train_steps),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--backbone', type=str, default='resnet18')
        parser.add_argument('--val_split', type=float, default=0.2)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--pretrained', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--data_dir', type=str, default='/opt/data/torch-cache/')
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                            metavar='W', help='weight decay')
        parser.add_argument('--predict_only', action='store_true', dest='predict_only')
        parser.set_defaults(predict_only=False)
        parser.add_argument('--sync_min_interval', type=int, default=180)

        return parser


class ImageClassificationPlotter():
    def __init__(self, model, dm, classes=None):
        self.model = model
        self.dm = dm
        self.classes = classes

    def plot_sample(self, ax, x, y, z=None):
        cmap = 'gray' if x.shape[0] == 1 else None
        if x.shape[0] == 1:
            x = x[0]
        else:
            x = x.permute([1, 2, 0])
        ax.imshow(x.numpy(), cmap=cmap)

        y = self.classes[y] if self.classes else y
        title = str(y)
        if z is not None:
            z = self.classes[z] if self.classes else z
            title += f' / {z}'
        ax.set_title(title)

    def plot_batch(self, x, y, z=None):
        batch_sz = x.shape[0]
        nrows = ncols = math.ceil(math.sqrt(batch_sz))
        fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(2*nrows, 2*ncols))
        axs = axs.flatten()
        for ind, ax in enumerate(axs):
            if ind < batch_sz:
                ax = axs[ind]
                _z = None if z is None else z[ind].item()
                self.plot_sample(ax, x[ind], y[ind].item(), _z)
            ax.axis('off')

        fig.tight_layout()
        return fig

    def plot_dl(self, dl, out_path=None):
        x, y = next(iter(dl))
        fig = self.plot_batch(x, y)
        if out_path is not None:
            fig.savefig(out_path)

    def plot_dataloaders(self, out_dir=None):
        os.makedirs(out_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            out_path = None if out_dir is None else join(out_dir, f'{split}.png')
            dl = getattr(self.dm, f'{split}_dataloader')()
            self.plot_dl(dl, out_path=out_path)

    def plot_predictions(self, out_path=None):
        dl = self.dm.val_dataloader()
        self.model.eval()
        with torch.no_grad():
            x, y = next(iter(dl))
            x = x.to(self.model.device)
            z = self.model(x).argmax(1).cpu()
            x = x.cpu()
        fig = self.plot_batch(x, y, z)
        if out_path is not None:
            fig.savefig(out_path)


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

        args.gpus = torch.cuda.device_count()

        checkpoint_path = join(root_dir, 'last.ckpt')
        if isfile(checkpoint_path):
            args.resume_from_checkpoint = checkpoint_path

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, val_split=args.val_split,
            num_workers=args.num_workers, batch_size=args.batch_size)
        dm.train_transforms = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        dm.val_transforms = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        dm.prepare_data()
        dm.setup(stage='fit')
        dm.setup(stage='test')

        args.num_samples = dm.num_samples
        args.num_classes = dm.num_classes
        classes = dm.dataset_cls(args.data_dir, download=False).classes

        pl.seed_everything(args.seed)
        if args.resume_from_checkpoint:
            model = ImageClassification.load_from_checkpoint(args.resume_from_checkpoint)
        else:
            model = ImageClassification(args)

        plotter = ImageClassificationPlotter(model, dm, classes=classes)

        csv_logger = pl.loggers.CSVLogger(root_dir, 'csv')
        tb_logger = pl.loggers.TensorBoardLogger(root_dir, 'tb')
        checkpoint_callback = ModelCheckpoint(
            root_dir, 'final_epoch', save_last=True)
        lr_monitor_callback = LearningRateMonitor()
        callbacks = [checkpoint_callback, lr_monitor_callback]
        if orig_root_dir.startswith('s3://'):
            callbacks.append(
                S3SyncCallback(root_dir, orig_root_dir, args.sync_min_interval))

        trainer = Trainer.from_argparse_args(
            args, logger=[csv_logger, tb_logger],
            callbacks=callbacks)

        if not args.predict_only:
            out_dir = join(root_dir, 'dataloaders')
            plotter.plot_dataloaders(out_dir)
            trainer.fit(model, datamodule=dm)

        test_metrics = trainer.test(model, datamodule=dm)
        out_path = join(root_dir, 'test-metrics.json')
        save_json(test_metrics, out_path)

        out_path = join(root_dir, 'predictions.png')
        plotter.plot_predictions(out_path)

        if orig_root_dir.startswith('s3://'):
            s3_sync(root_dir, orig_root_dir)

def get_arg_parser():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = ImageClassification.add_model_specific_args(parser)

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
