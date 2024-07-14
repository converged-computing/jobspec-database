#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import shutil
import signal
import sys
import textwrap
from pathlib import Path
from typing import Union

import ray
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from split_dataset import autosplit


class PrepareDataset:

    def __init__(self,
                 project_id: int,
                 dataset_path: str = './dataset',
                 weights: Union[list, tuple] = (0.8, 0.2, 0.0)):
        self.project_id = project_id
        self.dataset_path = Path(dataset_path)
        self.weights = weights

    @staticmethod
    def keyboard_interrupt_handler(sig: int, _) -> None:
        print(f'\nKeyboardInterrupt (id: {sig}) has been caught...')
        print('Terminating the session gracefully...')
        sys.exit(1)

    def get_project_data(self) -> None:

        @ray.remote
        def download(task: dict):
            img_url = task['data']['image']
            if img_url.startswith('s3://') and s3_endpoint:
                img_url = img_url.replace('s3://', s3_endpoint)
            fname = self.dataset_path / 'images' / Path(img_url).name
            if fname.exists():
                return
            with open(fname, 'wb') as fp:
                res = requests.get(img_url)
                fp.write(res.content)

        s3_endpoint = os.getenv('S3_ENDPOINT')
        if s3_endpoint:
            s3_endpoint = s3_endpoint.rstrip('/') + '/'

        headers = {
            'Authorization': f'Token {os.environ["LABEL_STUDIO_TOKEN"]}'
        }
        label_studio_host = os.environ['LABEL_STUDIO_HOST'].rstrip('/') + '/'

        print('Requesting project data in YOLO format...')
        r = requests.get(
            f'{label_studio_host}/api/projects/{self.project_id}/export?exportType=YOLO',  # noqa E501
            headers=headers)
        r.raise_for_status()

        with open(f'{self.dataset_path}.zip', 'wb') as f:
            f.write(r.content)

        shutil.unpack_archive(f'{self.dataset_path}.zip', self.dataset_path)
        Path(f'{self.dataset_path}.zip').unlink()

        print('Requesting project data in JSON format...')
        r = requests.get(
            f'{label_studio_host}/api/projects/{self.project_id}/export?exportType=JSON',  # noqa E501
            headers=headers)
        r.raise_for_status()
        data = r.json()

        with open(f'{self.dataset_path}/annotated_tasks.json', 'w') as j:
            json.dump(data, j)

        futures = [download.remote(task) for task in tqdm(data)]
        results_nums = [ray.get(future) for future in tqdm(futures)]

    def create_dataset_config(self):

        with open(self.dataset_path / 'classes.txt') as f:
            classes = f.read().splitlines()
        num_classes = len(classes)

        content = f'''\
        path: {self.dataset_path.absolute()}
        train: autosplit_train.txt
        val: autosplit_val.txt
        test:
        nc: {num_classes}
        names: {classes}\n'''

        with open(self.dataset_path / 'dataset_config.yml', 'w') as f:
            f.write(textwrap.dedent(content))

    def run_pipeline(self):
        signal.signal(signal.SIGINT, self.keyboard_interrupt_handler)

        self.get_project_data()

        splits = [
            'autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt'
        ]

        autosplit(self.dataset_path, self.weights)

        for split in splits:
            if Path(split).exists():
                if (self.dataset_path / split).exists():
                    Path(self.dataset_path / split).unlink()
                shutil.move(split, self.dataset_path)

        self.create_dataset_config()


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--project-id',
                        help='Label-studio project id',
                        type=int,
                        required=True)
    parser.add_argument('-d',
                        '--dataset-path',
                        help='Path to the output dataset '
                        '(if existing, dataset will be updated)',
                        type=str,
                        default='./dataset')

    parser.add_argument(
        '-w',
        '--weights',
        help='Split weights: train val test (default: 0.8 0.2 0.0)',
        type=float,
        default=[0.8, 0.2, 0.0],
        nargs=3)
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()
    pd = PrepareDataset(project_id=args.project_id,
                        dataset_path=args.dataset_path,
                        weights=args.weights)
    pd.run_pipeline()
