#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
# import random
import shutil
import sys
from glob import glob
from pathlib import Path

import numpy as np
# import ipyplot
import ray
from loguru import logger
from ray.exceptions import RayTaskError
from tqdm import tqdm
# from tqdm.notebook import tqdm

sys.path.insert(0, f'{os.getcwd()}/ai4eutils')
sys.path.insert(0, f'{os.getcwd()}/CameraTraps')

from CameraTraps.visualization import visualize_detector_output  # noqa


def process_data(jsonfiles):
    _data = []

    for json_file in jsonfiles:
        with open(json_file) as _j:
            d = json.load(_j)
            _data.append(d['images'])
    _data = sum(_data, [])
    return _data


def split_data(__data):
    _detections = []
    _no_detections = []
    _failed = []

    for x in __data:
        if x.get('detections'):
            _detections.append(x)
        elif x.get('failure'):
            _failed.append(x)
        else:
            _no_detections.append(x)

    logger.debug(f'Detections: {len(_detections)}\nNo'
                 f'detections: {len(_no_detections)}')
    return _detections, _no_detections, _failed


def create_conf_levels_dict(_detections):
    D = {round(x, 1): [] for x in np.arange(0.1, 1, 0.1)}

    for x in tqdm(_detections):
        for k in D:
            if x['max_detection_conf'] >= k:
                D[k].append(x)

    for k in D:
        assert any([
            x['max_detection_conf'] for x in D[k]
            if x['max_detection_conf'] >= k
        ])

    logger.debug('Data size in each conf value:')

    count_dict = {}
    for k, v in D.items():
        count_dict.update({k: len(v)})
        logger.debug(f'{k}: {len(v)}')

    with open('detections_per_conf_lvl.json', 'w') as js:
        json.dump(D, js, indent=4)

    with open('detections_per_conf_lvl_count.json', 'w') as js:
        json.dump(count_dict, js, indent=4)
    return D


def _sort_files(x):
    if x['detections']:
        cat_str = 'detections'
    else:
        cat_str = 'no_detections'

    if not Path(x['file']).exists():
        return x['file']

    _file = Path('/'.join(Path(x['file']).parts[1:]))
    out_file = f'{args.output_dir}/{cat_str}/{_file}'
    Path(Path(out_file).parent).mkdir(exist_ok=True, parents=True)

    shutil.copy(x["file"], out_file)


@ray.remote
def sort_files(x):
    _sort_files(x)


# def sample_detections(detections,
#                       sample_size_per_level=300,
#                       output_image_width=1280):
#     random.seed(8)
#     random_D = {}
#
#     for k, v in D.items():
#         random_D.update({k: random.sample(v, sample_size_per_level)})
#
#     names = {'1': 'animal', '2': 'person', '3': 'vehicle'}
#
#     ND = random.sample([x['file'] for x in no_detections],
#                        sample_size_per_level)
#     Path('no_detections_sample').mkdir(exist_ok=True)
#     for x in ND:
#         shutil.copy2(x, f'no_detections_sample/{Path(x).name}')
#
#     for level in np.arange(0.1, 1, 0.1):
#         level = round(level, 1)
#         level_dir_path = f'levels/{level}'
#         Path(level_dir_path).mkdir(exist_ok=True, parents=True)
#
#         visualize_detector_output.visualize_detector_output(
#             detector_output_path=random_D[level],
#             out_dir=level_dir_path,
#             confidence=level,
#             images_dir='.',
#             is_azure=False,
#             sample=-1,
#             output_image_width=output_image_width,
#             random_seed=None,
#             render_detections_only=True)
#
#     display_width = output_image_width / 2
#     zoom_scale = output_image_width / display_width
#
#     levels_folders = glob(f'levels/*')
#
#     images = [glob(f'{level}/*') for level in levels_folders]
#     labels = [[Path(Path(x).parent).name for x in y] for y in images]
#
#     plot = ipyplot.plot_class_tabs(images,
#                                    labels,
#                                    max_imgs_per_tab=sample_size_per_level,
#                                    img_width=display_width,
#                                    zoom_scale=zoom_scale,
#                                    hide_images_url=False,
#                                    display=False)
#
#     with open('random_detections_sample.html', 'w') as f:
#         f.write(plot)
#     logger.info('Exported sample to `random_detections_sample.html`')


def opts():
    parser = argparse.ArgumentParser()
    # parser.add_argument( '-d', '--results-dir', type=str, help='Path to
    # the results folder with the *.json files. The results ' 'directory
    # should be in the same parent directory as the directory of ' 'the
    # images data (e.g., `./parent/results_dir`, `./parent/images`)',
    # required=True)
    parser.add_argument('-d',
                        '--data-dir',
                        type=str,
                        help='Data directory',
                        required=True)
    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        help='Name of the directory to save the results to',
                        default='filtered_data')
    parser.add_argument(
        '-c',
        '--max-detection-conf-threshold',
        type=float,
        help='Max. detection confidence threshold for detections to be kept',
        required=True)
    parser.add_argument('--disable-ray',
                        action='store_true',
                        help='Disable ray module')
    return parser.parse_args()


if __name__ == '__main__':
    args = opts()
    Path(args.output_dir).mkdir(exist_ok=True)
    json_files = [
        x for x in glob(f'{args.data_dir}/**/*.json', recursive=True)
        if Path(x).name != 'ckpt.json'
    ]
    logger.debug(f'Data file path example: {json_files[0]}')

    data = process_data(json_files)
    logger.debug(f'Number of data items: {len(data)}')

    detections, no_detections, failed = split_data(data)

    with open('failed.json', 'w') as j:
        json.dump(failed, j, indent=4)

    logger.error(
        'Files that MegaDetector failed to predict: '
        f'{json.dumps(failed, indent=4)}')

    max_detection_conf = round(args.max_detection_conf_threshold, 1)
    detections = create_conf_levels_dict(detections)[max_detection_conf]

    for ITEM in [(no_detections, 'no_detections'), (detections, 'detections')]:
        futures = []
        not_found = []

        try:
            for future in tqdm(ITEM[0]):
                if args.disable_ray:
                    futures.append(_sort_files(future))
                else:
                    futures.append(sort_files.remote(future))

            for future in tqdm(futures):
                if args.disable_ray:
                    not_found.append(future)
                else:
                    not_found.append(ray.get(future))

            not_found = [x for x in not_found if x]

        except (KeyboardInterrupt, RayTaskError, TypeError) as e:
            logger.error(e)
            ray.shutdown()

        if not_found:
            logger.error('Number of items that were not found in '
                         f'{ITEM[1]}: {len(not_found)}')
            logger.error('These items belong to these directories:\n'
                         f'{list(set([Path(x).parent for x in not_found]))}')

    ray.shutdown()
