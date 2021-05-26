import sys
import argparse
import os
import tempfile
import shutil
from os.path import join, basename, isfile, splitext, isdir
import subprocess

from pytorch_models.utils import s3_sync, s3_cp, batch_submit, unzip


# From https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, raise_error=False):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code and raise_error:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_vissl(config, dataset_dir, output_dir, pretrained_path=None):
    train_dir = join(dataset_dir, 'train')
    val_dir = join(dataset_dir, 'val')
    cmd = [
        'python',
        '/opt/vissl/vissl/tools/run_distributed_engines.py',
        'hydra.verbose=true',
        f'config={config}',
        f'config.DATA.TRAIN.DATA_PATHS=["{train_dir}"]',
        f'config.DATA.TEST.DATA_PATHS=["{val_dir}"]',
        f'config.CHECKPOINT.DIR="{output_dir}"']
    if pretrained_path:
        cmd.append(f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE="{pretrained_path}"')

    for line in execute(cmd):
        print(line, end='')


def main(args):
    os.makedirs('/opt/data/tmp/', exist_ok=True)
    data_cache_dir = '/opt/data/data-cache/'
    os.makedirs(data_cache_dir, exist_ok=True)

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
                new_pretrained_uri = join(data_cache_dir, basename(pretrained_uri))
                if not isfile(new_pretrained_uri):
                    s3_cp(pretrained_uri, new_pretrained_uri)
                else:
                    print(f'Using cached pretrained_uri in {new_pretrained_uri}')
                pretrained_uri = new_pretrained_uri

        dataset_uri = args.dataset_uri
        if dataset_uri.startswith('s3://'):
            new_dataset_uri = join(data_cache_dir, basename(dataset_uri))
            new_dataset_dir = join(data_cache_dir, splitext(basename(dataset_uri))[0])
            if not isdir(new_dataset_dir):
                if not isfile(new_dataset_uri):
                    s3_cp(dataset_uri, new_dataset_uri)
                unzip(new_dataset_uri, new_dataset_dir)
                os.remove(new_dataset_uri)
            else:
                print(f'Using cached dataset in {new_dataset_dir}')
            dataset_uri = new_dataset_dir

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
    if '--aws-batch' in command:
        command.remove('--aws-batch')
        command.insert(0, 'python')
        batch_submit(command)
        exit()

    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
