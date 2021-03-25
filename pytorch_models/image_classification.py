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
from torch.optim.lr_scheduler import CyclicLR
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl

from pytorch_models.utils import save_json, s3_sync, batch_submit, S3SyncCallback

class ImageClassification(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        model_cls = getattr(torchvision.models, self.hparams.backbone)
        model = model_cls(pretrained=self.hparams.pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.hparams.num_classes)
        self.model = model

        self.prepared_data = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def setup(self, stage):
        if stage == 'fit':
            total_devices = max(1, self.hparams.gpus * self.hparams.num_nodes)
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.hparams.max_epochs * train_batches) // self.hparams.accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        step_size_up = self.train_steps // 2
        step_size_down = self.train_steps - step_size_up
        one_cycle_scheduler = CyclicLR(
            optimizer,
            base_lr=self.hparams.learning_rate / 10,
            max_lr=self.hparams.learning_rate,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=False)
        scheduler = {
            'scheduler': one_cycle_scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]

    def prepare_data(self):
        if not self.prepared_data:
            train_transform = T.Compose(
                [
                    T.RandomCrop(32, padding=4, padding_mode='reflect'),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ]
            )
            self.full_dataset = CIFAR10(
                self.hparams.data_dir, train=True, download=True)
            self.classes = self.full_dataset.classes
            train_len = int(round(len(self.full_dataset) * self.hparams.train_ratio))
            inds = torch.randperm(len(self.full_dataset))
            train_inds = inds[0:train_len]
            val_inds = inds[train_len:]
            self.train_ds = Subset(CIFAR10(
                self.hparams.data_dir, train=True, download=True, transform=train_transform),
                train_inds)
            self.val_ds = Subset(CIFAR10(
                self.hparams.data_dir, train=True, download=True, transform=train_transform),
                val_inds)
            self.test_ds = CIFAR10(
                self.hparams.data_dir, train=False, download=True,
                transform=T.ToTensor())

            self.prepared_data = True

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, shuffle=True, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)

    def plot_sample(self, ax, x, y, z=None):
        cmap = 'gray' if x.shape[0] == 1 else None
        if x.shape[0] == 1:
            x = x[0]
        else:
            x = x.permute([1, 2, 0])
        ax.imshow(x, cmap=cmap)

        y = self.classes[y]
        title = str(y)
        if z is not None:
            z = self.classes[z]
            title += f' / {z}'
        ax.set_title(title)

    def raw2pred(self, z):
        return z.argmax(1)

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
            dl = getattr(self, f'{split}_dataloader')()
            self.plot_dl(dl, out_path=out_path)

    def plot_predictions(self, out_path=None):
        dl = self.val_dataloader()
        self.eval()
        with torch.no_grad():
            x, y = next(iter(dl))
            x = x.to(self.device)
            z = self(x).cpu()
            z = self.raw2pred(z)
            x = x.cpu()
        fig = self.plot_batch(x, y, z)
        if out_path is not None:
            fig.savefig(out_path)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--backbone', type=str, default='resnet18')
        parser.add_argument('--train_ratio', type=float, default=0.8)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--pretrained', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--data_dir', type=str, default='/opt/data/torch-cache/')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

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

        checkpoint_path = join(root_dir, 'last_epoch.ckpt')
        if isfile(checkpoint_path):
            args.resume_from_checkpoint = checkpoint_path

        pl.seed_everything(args.seed)
        if args.resume_from_checkpoint:
            model = ImageClassification.load_from_checkpoint(args.resume_from_checkpoint)
        else:
            model = ImageClassification(args)
        model.prepare_data()

        csv_logger = pl.loggers.CSVLogger(root_dir, 'csv')
        tb_logger = pl.loggers.TensorBoardLogger(root_dir, 'tb')
        checkpoint_callback = ModelCheckpoint(root_dir, 'last_epoch')
        lr_monitor_callback = LearningRateMonitor()
        callbacks = [checkpoint_callback, lr_monitor_callback]
        if orig_root_dir.startswith('s3://'):
            callbacks.append(S3SyncCallback(root_dir, orig_root_dir))

        trainer = Trainer.from_argparse_args(
            args, logger=[csv_logger, tb_logger],
            callbacks=callbacks)

        if not args.predict_only:
            out_dir = join(root_dir, 'dataloaders')
            model.plot_dataloaders(out_dir)
            trainer.fit(model)

        test_metrics = trainer.test(model, test_dataloaders=model.val_dataloader())
        out_path = join(root_dir, 'val-metrics.json')
        save_json(test_metrics, out_path)

        out_path = join(root_dir, 'predictions.png')
        model.plot_predictions(out_path)

        if orig_root_dir.startswith('s3://'):
            s3_sync(root_dir, orig_root_dir)

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--predict_only', type=bool, default=False)
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
