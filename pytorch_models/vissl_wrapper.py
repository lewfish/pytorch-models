import sys
from pytorch_models.utils import save_json, s3_sync, batch_submit, S3SyncCallback

def main(args):
    cmd = [
        'python',
        '/tmp/vissl/tools/run_distributed_engines.py',
        'hydra.verbose=true',
        f'config={args.config}',
        f'config.DATA.TRAIN.DATA_PATHS=["{data_dir}"]',
        f'config.CHECKPOINT.DIR="{output_dir}"]',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE="{weights_path}"'
    ]

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
