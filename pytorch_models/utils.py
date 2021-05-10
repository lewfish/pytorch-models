import json
import subprocess
import uuid
import boto3
import os
import time

from torch.utils.data import Dataset
from pytorch_lightning.callbacks import Callback

def save_json(x, out_path):
    with open(out_path, 'w') as out_file:
        json.dump(x, out_file)

def s3_sync(from_uri, to_uri):
    cmd = ['aws', 's3', 'sync', from_uri, to_uri]
    print(f'Syncing from {from_uri} to {to_uri}...')
    subprocess.run(cmd)

def s3_cp(from_uri, to_uri):
    cmd = ['aws', 's3', 'cp', from_uri, to_uri]
    print(f'Copying from {from_uri} to {to_uri}...')
    subprocess.run(cmd)

def unzip(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    cmd = ['unzip', zip_path, '-q', '-d', dest_dir]
    subprocess.run(cmd)

def batch_submit(command, attempts=3):
    job_def = os.environ['JOB_DEF']
    job_queue = os.environ['JOB_QUEUE']
    client = boto3.client('batch')
    job_name = 'pytorch-models-{}'.format(uuid.uuid4())

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
        'containerOverrides': {
            'command': command
        },
        'retryStrategy': {
            'attempts': attempts
        }
    }

    job_id = client.submit_job(**kwargs)['jobId']
    msg = 'submitted job with jobName={} and jobId={}'.format(
        job_name, job_id)
    print(command)
    print(msg)
    return job_id

class S3SyncCallback(Callback):
    def __init__(self, from_uri, to_uri, min_interval=60):
        super().__init__()
        self.from_uri = from_uri
        self.to_uri = to_uri
        self.min_interval = min_interval

        self.last_sync = time.time()

    def on_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.last_sync
        if elapsed > self.min_interval:
            s3_sync(self.from_uri, self.to_uri)
            self.last_sync = time.time()

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, ind):
        x, y = self.dataset[ind]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)
