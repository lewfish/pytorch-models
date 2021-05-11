# %%
from IPython import get_ipython

from pytorch_models.image_classification import get_arg_parser, main

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
parser = get_arg_parser()
args = parser.parse_args(args=[])
args.default_root_dir = '/opt/data/lightning/cifar10/'
# args.default_root_dir = 's3://raster-vision-lf-dev/research/lightning/cifar10/'
args.data_dir = '/opt/data/torch-cache'
args.num_workers = 4
args.batch_size = 16
args.max_steps = 10
args.limit_val_batches = 10
args.limit_test_batches = 10
args.fast_dev_run = False
args.max_epochs = 1

# args.predict_only = True
# args.resume_from_checkpoint = '/opt/data/lightning/mnist/last_epoch.ckpt'

# %%
parser = get_arg_parser()
args = parser.parse_args(args=[])
args.default_root_dir = '/opt/data/research/ssl/moco/fine-tune/output/'
args.data_dir = '/opt/data/torch-cache'
args.num_workers = 4
args.batch_size = 16
args.max_steps = 10
args.limit_val_batches = 10
args.limit_test_batches = 10
args.fast_dev_run = False
args.max_epochs = 1

args.backbone = 'resnet50'
args.cifar_mode = False
args.dataset_type = 'image_folder'
# args.dataset_uri = '/opt/data/research/ssl/resisc45/'
args.dataset_uri = 's3://research-lf-dev/ssl/datasets/resisc45.zip'
args.pretrained_uri = '/opt/data/research/ssl/moco/backbones/resisc_moco.pth'
args.train_ratio = 0.8
args.train_sz = 1000
args.num_classes = 45

# %%
main(args)

# %%
