# This trains a MoCo model on CIFAR10. It was adapted from:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

from os.path import join, basename
import os
import tempfile
import sys
from datetime import datetime
from functools import partial
from typing import OrderedDict
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.models import resnet
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_models.utils import (
    s3_cp, save_json, s3_sync, batch_submit, S3SyncCallback, unzip)


# From https://github.com/facebookresearch/moco/blob/master/moco/loader.py
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, ind):
        x, y = self.dataset[ind]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)


def setup_data(args):
    os.makedirs(args.data_dir, exist_ok=True)
    crop_size = 32 if args.dataset_type == 'cifar10' else 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    train_transform = TwoCropsTransform(train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    if args.dataset_type == 'cifar10':
        train_data = CIFAR10(root=args.data_dir, train=True, transform=train_transform, download=True)
        memory_data = CIFAR10(root=args.data_dir, train=True, transform=test_transform, download=True)
        test_data = CIFAR10(root=args.data_dir, train=False, transform=test_transform, download=True)
    else:
        data = ImageFolder(root=args.dataset_uri)
        inds = torch.randperm(len(data))
        nb_train = int(len(data) * args.train_ratio)
        train_data = TransformedDataset(Subset(data, inds[0:nb_train]), train_transform)
        train_data.classes = data.classes
        memory_data = TransformedDataset(Subset(data, inds[0:nb_train]), test_transform)
        memory_data.classes = data.classes
        test_data = TransformedDataset(Subset(data, inds[nb_train:]), test_transform)
        test_data.classes = data.classes

    if args.fast_test:
        use_inds = list(range(args.batch_size))
        classes = train_data.classes

        train_data = Subset(train_data, use_inds)
        memory_data = Subset(memory_data, use_inds)
        test_data = Subset(test_data, use_inds)

        train_data.classes = classes
        memory_data.classes = classes
        test_data.classes = classes

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, memory_loader, test_loader


class DataPlotter():
    def __init__(self, train_loader, memory_loader, test_loader):
        self.train_loader = train_loader
        self.memory_loader = memory_loader
        self.test_loader = test_loader

    def plot_sample(self, ax, x):
        x = x.permute([1, 2, 0])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        x = (x * std) + mean
        ax.imshow(x.numpy())

    def plot_batch(self, x, y, classes=None):
        batch_sz = x.shape[0]
        nrows = ncols = math.ceil(math.sqrt(batch_sz))
        fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(2*nrows, 2*ncols))
        axs = axs.flatten()
        for ind, ax in enumerate(axs):
            if ind < batch_sz:
                ax = axs[ind]
                self.plot_sample(ax, x[ind])
                if classes:
                    ax.set_title(classes[y[ind]])
            ax.axis('off')

        fig.tight_layout()
        return fig

    def plot_ssl_batch(self, x1, x2, y=None, classes=None):
        batch_sz = x1.shape[0]
        nrows = ncols = math.ceil(math.sqrt(batch_sz * 2))
        fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(2*nrows, 2*ncols))
        axs = axs.flatten()
        for ind in range(batch_sz):
            ax = axs[ind*2]
            self.plot_sample(ax, x1[ind])
            if classes and y is not None:
                ax.set_title(classes[y[ind]])
            ax.axis('off')

            ax = axs[ind*2 + 1]
            self.plot_sample(ax, x2[ind])
            ax.axis('off')

        fig.tight_layout()
        return fig

    def plot_dl(self, dl, out_path=None, batch_lim=16):
        x, y = next(iter(dl))
        classes = dl.dataset.classes

        if isinstance(x, tuple) or isinstance(x, list):
            x1, x2 = x
            if x1.shape[0] > batch_lim:
                x1 = x1[0:batch_lim]
                x2 = x2[0:batch_lim]
            fig = self.plot_ssl_batch(x1, x2, y, classes)
        else:
            if x.shape[0] > batch_lim:
                x = x[0:batch_lim]
            fig = self.plot_batch(x, y, classes)
        if out_path is not None:
            fig.savefig(out_path)

    def plot_dataloaders(self, out_dir=None, batch_lim=16):
        os.makedirs(out_dir, exist_ok=True)
        for split in ['train', 'memory', 'test']:
            out_path = None if out_dir is None else join(out_dir, f'{split}.png')
            dl = getattr(self, f'{split}_loader')
            self.plot_dl(dl, out_path=out_path, batch_lim=batch_lim)


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)

def build_backbone(feature_dim=128, arch=None, bn_splits=16, cifar_mode=False):
    norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
    resnet_arch = getattr(resnet, arch)
    model = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)
    if cifar_mode:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model

class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8,
                 symmetric=True, cifar_mode=False, mlp=False, pretrained_uri=None):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = build_backbone(
            feature_dim=dim, arch=arch, bn_splits=bn_splits, cifar_mode=cifar_mode)
        self.encoder_k = build_backbone(
            feature_dim=dim, arch=arch, bn_splits=bn_splits, cifar_mode=cifar_mode)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if pretrained_uri:
            state_dict = torch.load(pretrained_uri, map_location='cpu')['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.encoder_q.', '')] = v
            self.encoder_q.load_state_dict(new_state_dict)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(x.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(im_q.device)

        loss = nn.CrossEntropyLoss().to(im_q.device)(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss

def train(optimizer, net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for (im_1, im_2), _ in train_bar:
        im_1, im_2 = im_1.to(args.device, non_blocking=True), im_2.to(args.device, non_blocking=True)

        loss = net(im_1, im_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(net, memory_data_loader, test_data_loader, epoch, args):
    # test using a knn monitor
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.to(args.device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(torch.cat(feature_labels), device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def main(args, tmp_dir):
    train_loader, memory_loader, test_loader = setup_data(args)
    if args.plot_dl:
        plotter = DataPlotter(train_loader, memory_loader, test_loader)
        plot_dir = join(args.root_dir, 'dataloaders')
        plotter.plot_dataloaders(plot_dir)

    model = ModelMoCo(
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
            arch=args.arch,
            bn_splits=args.bn_splits,
            symmetric=args.symmetric,
            cifar_mode=args.cifar_mode,
            mlp=args.mlp,
            pretrained_uri=args.pretrained_uri
        ).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # load model if resume
    epoch_start = 1
    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.root_dir):
        os.mkdir(args.root_dir)
    # dump args
    with open(args.root_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in range(epoch_start, args.epochs + 1):
        if args.skip_train:
            train_loss = 0.0
        else:
            train_loss = train(optimizer, model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.root_dir + '/log.csv', index_label='epoch')
        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.root_dir + '/model_last.pth')


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

    parser.add_argument('-a', '--arch', default='resnet18')

    # lr: 0.06 for batch 512 (or 0.03 for batch 256)
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

    parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

    parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

    # knn monitor
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--root-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--data-dir', default='/opt/data/data-cache/', type=str, metavar='PATH', help='path to dataset cache (default: data)')
    parser.add_argument('--fast-test', action='store_true', help='run a fast test')
    parser.add_argument('--cifar-mode', action='store_true', help='use resnet modified for cifar-sized images')
    parser.add_argument('--mlp', action='store_true', help='use mlp for last layer aka mocov2')
    parser.add_argument('--pretrained-uri', default='', type=str, metavar='PATH', help='URI of pretrained model')
    parser.add_argument('--dataset-type', default='cifar10', type=str, metavar='DATASET_TYPE', help='type of dataset')
    parser.add_argument('--dataset-uri', default='', type=str, metavar='DATASET_URI', help='URI of dataset')
    parser.add_argument('--train-ratio', default=0.8, help='ratio of dataset to use for training')
    parser.add_argument('--plot-dl', action='store_true', help='plot dataloaders')
    parser.add_argument('--skip-train', action='store_true', help='skip training and just do eval')

    return parser

if __name__ == '__main__':
    command = sys.argv
    if '--aws-batch' in command:
        command.remove('--aws-batch')
        command.insert(0, 'python')
        batch_submit(command)
        exit()

    parser = get_arg_parser()
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    if args.fast_test:
        args.batch_size = 8
        args.epochs = 1
        args.knn_k = 1

    os.makedirs('/opt/data/tmp/', exist_ok=True)
    with tempfile.TemporaryDirectory(dir='/opt/data/tmp/') as tmp_dir:
        root_dir = args.root_dir
        orig_root_dir = root_dir
        if root_dir.startswith('s3://'):
            new_root_dir = join(tmp_dir, 'output')
            s3_sync(root_dir, new_root_dir)
            root_dir = new_root_dir
            args.root_dir = root_dir

        pretrained_uri = args.pretrained_uri
        if pretrained_uri.startswith('s3://'):
            new_pretrained_uri = join(tmp_dir, basename(pretrained_uri))
            s3_cp(pretrained_uri, new_pretrained_uri)
            args.pretrained_uri = new_pretrained_uri

        dataset_uri = args.dataset_uri
        if dataset_uri and dataset_uri.startswith('s3://'):
            new_dataset_uri = join(tmp_dir, basename(dataset_uri))
            s3_cp(dataset_uri, new_dataset_uri)
            dataset_dir = join(args.data_dir, 'dataset')
            unzip(new_dataset_uri, dataset_dir)
            # TODO: rm new_dataset_uri
            args.dataset_uri = dataset_dir

        main(args, tmp_dir)

        if orig_root_dir.startswith('s3://'):
            s3_sync(root_dir, orig_root_dir)
