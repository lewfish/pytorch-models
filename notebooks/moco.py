# %%
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
import torch
from pytorch_models.moco import get_arg_parser, main

# %%
# cifar10
parser = get_arg_parser()
args = parser.parse_args('')

args.data_dir = '/opt/data/data-cache/'
args.root_dir = '/opt/data/research/ssl/moco/'
args.fast_test = True
args.mlp = True
args.arch = 'resnet50'
args.cifar_mode = False
args.pretrained_uri = '/opt/data/research/ssl/checkpoints/moco_v2_800ep_pretrain.pth.tar'
args.plot_dl = True

args.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
if args.fast_test:
    args.batch_size = 16
    args.epochs = 1
    args.knn_k = 1
tmp_dir = '/opt/data/tmp/tmp'
main(args, tmp_dir)

# %%

# resisc45

parser = get_arg_parser()
args = parser.parse_args('')

args.root_dir = '/opt/data/research/ssl/moco/'
args.fast_test = True
args.mlp = True
args.arch = 'resnet50'
args.cifar_mode = False
args.dataset_type = 'image_folder'
# args.dataset_uri = '/opt/data/research/ssl/resisc45/'
args.dataset_uri = 's3://research-lf-dev/ssl/datasets/resisc45.zip'
args.pretrained_uri = '/opt/data/research/ssl/checkpoints/moco_v2_800ep_pretrain.pth.tar'
args.plot_dl = True

args.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
if args.fast_test:
    args.batch_size = 16
    args.epochs = 1
    args.knn_k = 1
tmp_dir = '/opt/data/tmp/tmp'
main(args, tmp_dir)

# %%
data_dir = '/opt/data/research/ssl/resisc45/'
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

ds = ImageFolder(data_dir, transform=train_transform)
ds_subset = Subset(ds, list(range(10)))
# %%
