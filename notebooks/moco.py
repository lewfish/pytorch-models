# %%
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
import torch
from pytorch_models.moco import get_arg_parser, main

# %%
parser = get_arg_parser()
args = parser.parse_args('')

args.data_dir = '/opt/data/data-cache/'
args.results_dir = '/opt/data/research/ssl/moco/'
args.fast_test = True

args.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
if args.fast_test:
    args.batch_size = 8
    args.epochs = 1
    args.knn_k = 1
main(args)

# %%
