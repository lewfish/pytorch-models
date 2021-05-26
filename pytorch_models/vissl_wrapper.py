import sys
import argparse

from pytorch_models.utils import s3_sync, s3_cp, batch_submit


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Run VISSL')
    parser.add_argument('--config', default='')
    parser.add_argument('--output-uri', default='')
    parser.add_argument('--init-weights', default='')
    return parser


def main(args):
    # download data zip file and unzip to a tmp_dir
    # make an output dir
    # download init weights
    cmd = [
        'python',
        '/tmp/vissl/tools/run_distributed_engines.py',
        'hydra.verbose=true',
        f'config={args.config}',
        f'config.DATA.TRAIN.DATA_PATHS=["{data_dir}"]',
        f'config.CHECKPOINT.DIR="{output_dir}"]',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE="{weights_path}"'
    ]
    # sync output_dir to output_uri

if __name__ == '__main__':
    command = sys.argv
    if '--aws_batch' in command:
        command.remove('--aws_batch')
        command.insert(0, 'python')
        batch_submit(command)
        exit()

    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
