import sys
import argparse
import os
import tempfile
import shutil
from os.path import join, basename
import subprocess

from pytorch_models.utils import s3_sync, s3_cp, batch_submit, unzip


def run_vissl(config, dataset_dir, output_dir, pretrained_path=None):
    cmd = [
        'python',
        '/opt/vissl/vissl/tools/run_distributed_engines.py',
        'hydra.verbose=true',
        f'config={config}',
        f'config.DATA.TRAIN.DATA_PATHS=["{dataset_dir}"]',
        f'config.CHECKPOINT.DIR="{output_dir}"]']
    if pretrained_path:
        cmd.append(f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE="{pretrained_path}"')
    subprocess.run(cmd)
    # TODO


def main(args):
    os.makedirs('/opt/data/tmp/', exist_ok=True)
    with tempfile.TemporaryDirectory(dir='/opt/data/tmp/') as tmp_dir:
        output_uri = args.output_uri
        orig_output_uri = output_uri
        if output_uri.startswith('s3://'):
            new_output_uri = join(tmp_dir, 'output')
            s3_sync(output_uri, new_output_uri)
            output_uri = new_output_uri

        pretrained_uri = args.pretrained_uri
        if pretrained_uri:
            if pretrained_uri.startswith('s3://'):
                new_pretrained_uri = join(tmp_dir, basename(pretrained_uri))
                s3_cp(pretrained_uri, new_pretrained_uri)
                pretrained_uri = new_pretrained_uri
            else:
                pretrained_uri = None

        dataset_uri = args.dataset_uri
        if dataset_uri.startswith('s3://'):
            new_dataset_uri = join(tmp_dir, basename(dataset_uri))
            s3_cp(dataset_uri, new_dataset_uri)
            dataset_dir = join(args.data_dir, 'dataset')
            unzip(new_dataset_uri, dataset_dir)
            shutil.rmtree(new_dataset_uri)
            dataset_uri = dataset_dir

    run_vissl(args.config, dataset_uri, output_uri, pretrained_path=pretrained_uri)

    if orig_output_uri.startswith('s3://'):
        s3_sync(output_uri, orig_output_uri)


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Run VISSL')
    parser.add_argument('--config', default='')
    parser.add_argument('--output-uri', default='')
    parser.add_argument('--dataset-uri', default='')
    parser.add_argument('--pretrained-uri', default='')
    return parser


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
