#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from progress_manager import Progress


class GPUNotAvailable(Exception):
    pass


if '-h' not in sys.argv and '--help' not in sys.argv:
    import tensorflow as tf  # noqa
    sys.path.insert(0, f'{os.getcwd()}/ai4eutils')
    sys.path.insert(0, f'{os.getcwd()}/CameraTraps')
    try:
        from CameraTraps.detection import run_tf_detector_batch  # noqa
        from CameraTraps.visualization import visualize_detector_output  # noqa
    except RuntimeError as e:
        logger.exception(e)
        print('ERROR in loading local modules...')
        sys.exit(1)


class MegaDetector:

    def __init__(self,
                 images_dir=None,
                 resume=False,
                 cpu=False,
                 ckpt=None,
                 progress_file='progress.json',
                 confidence_threshold=0.1,
                 verbose=True):
        self.images_dir = images_dir
        self.resume = resume
        self.cpu = cpu
        self.ckpt = ckpt
        self.progress_file = progress_file
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

    @staticmethod
    def setup_dirs(folder):
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images_list = sum([glob(f'{folder}/*{ext}') for ext in img_extensions],
                          [])
        images_list_len = len(images_list)
        if not images_list_len:
            logger.warning(f'No images in the current folder: {folder} '
                           '(subdirs are not included)')
            return None
        logger.info(f'Number of images in the folder: {images_list_len}')

        logger.info(f'Will process {len(images_list)} images')
        logger.debug(f'Images folder: {folder}')

        output_folder = f'{folder}/output'
        Path(output_folder).mkdir(exist_ok=True)
        output_file_path = output_folder + \
            f'/data_{Path(folder).name}.json'
        return images_list, output_folder, output_file_path

    def restore_checkpoint(self, folder):
        ckpt_path = f'{folder}/output/ckpt.json'
        restored_results = []

        if self.resume:
            logger.info('Resuming from checkpoint...')
            try:
                if Path(ckpt_path).exists():
                    if self.ckpt:
                        ckpt_path = self.ckpt
                        logger.info(
                            'Resuming from custom checkpoint path instead'
                            ' of default...')
                    with open(ckpt_path) as f:
                        saved = json.load(f)

                    assert 'images' in saved, \
                        'The file saved as checkpoint does not have the ' \
                        'correct fields; cannot be restored'

                    restored_results = saved['images']
                    logger.info(f'Restored {len(restored_results)} '
                                f'entries from the checkpoint')
            except AssertionError as err:
                logger.exception(err)
        else:
            logger.info('Processing from the start...')

        return ckpt_path, restored_results

    def predict_folder(self, folder):
        logger.debug(tf.__version__)
        logger.debug(f'GPU available: {tf.test.is_gpu_available()}')

        if not tf.test.is_gpu_available():
            if not self.cpu:
                raise GPUNotAvailable(f'No available GPUs. Terminating... '
                                      f'Folder of terminated job: {folder}')

        try:
            images_list, output_folder, output_file_path = \
                self.setup_dirs(folder)
        except TypeError:
            return

        ckpt_path, restored_results = self.restore_checkpoint(folder)

        logger.info(f'Number of images in folder: {len(images_list)}')

        results = run_tf_detector_batch.load_and_run_detector_batch(
            model_file='megadetector_v4_1_0.pb',
            image_file_names=images_list,
            checkpoint_path=ckpt_path,
            confidence_threshold=0.1,
            checkpoint_frequency=100,
            results=restored_results,
            n_cores=0,
            use_image_queue=False)

        logger.debug('Finished running '
                     '`run_tf_detector_batch.load_and_run_detector_batch`')

        run_tf_detector_batch.write_results_to_file(results,
                                                    output_file_path,
                                                    relative_path_base=None)

        logger.debug(
            'Finished running `run_tf_detector_batch.write_results_to_file`')
        logger.info(f'Data file path: {output_file_path}')
        Path(f'{folder}/output/_complete').touch()
        return

    def run_detector(self):
        Path('logs').mkdir(exist_ok=True)

        assert Path(
            self.progress_file).exists(), '`progress.json` does not exist!'
        progress = Progress(data_dir=self.images_dir,
                            progress_file=self.progress_file,
                            verbose=self.verbose)

        if self.images_dir:
            folders = [
                x for x in glob(f'{self.images_dir}/**/*', recursive=True)
                if Path(x).is_dir()
            ]
            if len(folders) > 1:
                logger.debug('Detected multiple subdirs!')
            else:
                folders = [self.images_dir]
        else:
            with open(self.progress_file) as j:
                folders = list(json.load(j).keys())
                logger.info(f'Will process the following folders: '
                            f'{json.dumps(folders, indent=4)}')
                time.sleep(5)

        for folder in tqdm(folders):
            if progress.status(folder) is True:
                logger.warning(f'Finished folder {folder}! Skipping...')
                continue
            if progress.status(folder) == 'started':
                logger.warning(
                    f'Started folder {folder}, but it\'s either still in '
                    'progress or needs to be resumed. Check logs and pass '
                    '`--resume` if it needs to be resumed.')
                if not self.resume:
                    continue
            else:
                logger.info(f'Starting folder: {folder}...')

            progress.update_progress({folder: 'started'})

            if Path(f'{folder}/output/_complete').exists():
                logger.warning(f'`{folder}` is already completed! '
                               f'Skipping...')
                continue

            try:
                logger.debug(f'Current folder: {folder}')
                assert Path(folder).exists(), f'{folder} does not exist'
            except AssertionError as err:
                logger.exception(err)
                logger.error(f'Skipping {folder}...')
                continue

            self.predict_folder(folder)

            progress.update_progress({folder: True})


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir',
                        type=str,
                        help='Path to the source images folder (local)')
    parser.add_argument('--resume',
                        action='store_true',
                        help='Resume from the last checkpoint')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='Use CPU if GPU is not available')
    parser.add_argument('--ckpt',
                        type=str,
                        help='Path to a checkpoint file other than default')
    parser.add_argument('--progress-file',
                        default='progress.json',
                        help='Path to the progress JSON file',
                        type=str)
    parser.add_argument('--confidence-threshold',
                        default=0.1,
                        help='Confidence threshold (default: 0.1)',
                        type=float)
    parser.add_argument('--verbose',
                        help='Print lots more stuff',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    logger.add(f'logs/logs.log')
    args = opts()

    mega_detector = MegaDetector(
        images_dir=args.images_dir,
        resume=args.resume,
        cpu=args.cpu,
        ckpt=args.ckpt,
        progress_file=args.progress_file,
        confidence_threshold=args.confidence_threshold,
        verbose=args.verbose)
    mega_detector.run_detector()
