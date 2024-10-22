""" Training script for steps_with_decay policy"""

import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import random
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats


# Fix random seed
#np.random.seed(999)
#random.seed(999)

torch.cuda.manual_seed(999)
torch.cuda.manual_seed_all(999)
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True


# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

DEBUG = False

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=20, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 2
   
    # ADE20k as a detection dataset
    elif args.dataset == "ade_train":
        cfg.TRAIN.DATASETS = ('ade_train',)
        cfg.MODEL.NUM_CLASSES = 446
    
    # Noisy CS6+WIDER datasets
    elif args.dataset == 'cs6_noise020+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise020','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise050+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise050','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise080+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise080','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise085+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise085','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise090+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise090','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise095+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise095','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise100+WIDER':
        cfg.TRAIN.DATASETS = ('cs6_noise100','wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    
    # Just Noisy CS6 datasets
    elif args.dataset == 'cs6_noise020':
        cfg.TRAIN.DATASETS = ('cs6_noise020',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise030':
        cfg.TRAIN.DATASETS = ('cs6_noise030',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise040':
        cfg.TRAIN.DATASETS = ('cs6_noise040',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise050':
        cfg.TRAIN.DATASETS = ('cs6_noise050',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise060':
        cfg.TRAIN.DATASETS = ('cs6_noise060',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise070':
        cfg.TRAIN.DATASETS = ('cs6_noise070',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise080':
        cfg.TRAIN.DATASETS = ('cs6_noise080',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise085':
        cfg.TRAIN.DATASETS = ('cs6_noise085',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise090':
        cfg.TRAIN.DATASETS = ('cs6_noise090',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise095':
        cfg.TRAIN.DATASETS = ('cs6_noise095',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'cs6_noise100':
        cfg.TRAIN.DATASETS = ('cs6_noise100',)
        cfg.MODEL.NUM_CLASSES = 2


    # Cityscapes 7 classes
    elif args.dataset == "cityscapes":
        cfg.TRAIN.DATASETS = ('cityscapes_train',)
        cfg.MODEL.NUM_CLASSES = 8
   
    # BDD 7 classes
    elif args.dataset == "bdd_any_any_any":
        cfg.TRAIN.DATASETS = ('bdd_any_any_any_train',)
        cfg.MODEL.NUM_CLASSES = 8
    elif args.dataset == "bdd_any_any_daytime":
        cfg.TRAIN.DATASETS = ('bdd_any_any_daytime_train',)
        cfg.MODEL.NUM_CLASSES = 8
    elif args.dataset == "bdd_clear_any_daytime":
        cfg.TRAIN.DATASETS = ('bdd_clear_any_daytime_train',)
        cfg.MODEL.NUM_CLASSES = 8
   
    # Cistyscapes Pedestrian sets
    elif args.dataset == "cityscapes_peds":
        cfg.TRAIN.DATASETS = ('cityscapes_peds_train',)
        cfg.MODEL.NUM_CLASSES = 2

    # Cityscapes Car sets
    elif args.dataset == "cityscapes_cars_HPlen3+kitti_car_train":
        cfg.TRAIN.DATASETS = ('cityscapes_cars_HPlen3','kitti_car_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cityscapes_cars_HPlen5+kitti_car_train":
        cfg.TRAIN.DATASETS = ('cityscapes_cars_HPlen5','kitti_car_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cityscapes_car_train+kitti_car_train":
        cfg.TRAIN.DATASETS = ('cityscapes_car_train','kitti_car_train')
        cfg.MODEL.NUM_CLASSES = 2

    # KITTI Car set
    elif args.dataset == "kitti_car_train":
        cfg.TRAIN.DATASETS = ('kitti_car_train',)
        cfg.MODEL.NUM_CLASSES = 2

    # BDD pedestrians sets
    elif args.dataset == "bdd_peds":
        cfg.TRAIN.DATASETS = ('bdd_peds_train',) # bdd peds: clear_any_daytime
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "bdd_peds_full":
        cfg.TRAIN.DATASETS = ('bdd_peds_full_train',) # bdd peds: any_any_any
        cfg.MODEL.NUM_CLASSES = 2
    # Pedestrians with constraints
    elif args.dataset == "bdd_peds_not_clear_any_daytime":
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train',)
        cfg.MODEL.NUM_CLASSES = 2
    # Ashish's  20k samples videos
    elif args.dataset == "bdd_peds_not_clear_any_daytime_20k":
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_20k_train',)
        cfg.MODEL.NUM_CLASSES = 2
    # Source domain + Target domain detections
    elif args.dataset == "bdd_peds+DETS_20k":
        cfg.TRAIN.DATASETS = ('bdd_peds_dets_20k_target_domain','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    # Source domain + Target domain detections -- same 18k images as HP18k
    elif args.dataset == "bdd_peds+DETS18k":
        cfg.TRAIN.DATASETS = ('bdd_peds_dets18k_target_domain','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    # Only Dets
    elif args.dataset == "DETS20k":
        cfg.TRAIN.DATASETS = ('bdd_peds_dets_20k_target_domain',)
        cfg.MODEL.NUM_CLASSES = 2
    # Only Dets18k - same images as HP18k
    elif args.dataset == 'DETS18k':
        cfg.TRAIN.DATASETS = ('bdd_peds_dets18k_target_domain',)
        cfg.MODEL.NUM_CLASSES = 2
    # Only HP
    elif args.dataset == 'HP':
        cfg.TRAIN.DATASETS = ('bdd_peds_HP_target_domain',)
        cfg.MODEL.NUM_CLASSES = 2
    # Only HP 18k videos
    elif args.dataset == 'HP18k':
        cfg.TRAIN.DATASETS = ('bdd_peds_HP18k_target_domain',)
        cfg.MODEL.NUM_CLASSES = 2
    # Source domain + Target domain HP
    elif args.dataset == 'bdd_peds+HP':
        cfg.TRAIN.DATASETS = ('bdd_peds_train','bdd_peds_HP_target_domain')
        cfg.MODEL.NUM_CLASSES = 2
    # Source domain + Target domain HP 18k videos
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k':
        cfg.TRAIN.DATASETS = ('bdd_peds_HP18k_target_domain','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    #### Source domain + Target domain with different conf threshold theta ####
    elif args.dataset == 'bdd_peds+HP18k_thresh-050':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_thresh-050','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k_thresh-060':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_thresh-060','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k_thresh-070':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_thresh-070','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k_thresh-090':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_thresh-090','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    ##############################
    
    #### Data distillation on BDD -- for rebuttal
    elif args.dataset == 'bdd_peds+bdd_data_dist_small':
        cfg.TRAIN.DATASETS = ('bdd_data_dist_small','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+bdd_data_dist_mid':
        cfg.TRAIN.DATASETS = ('bdd_data_dist_mid','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+bdd_data_dist':
        cfg.TRAIN.DATASETS = ('bdd_data_dist','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    ##############################


    #### Source domain + **Labeled** Target domain with varying number of images
    elif args.dataset == 'bdd_peds+labeled_100':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_100','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+labeled_075':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_075','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+labeled_050':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_050','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+labeled_025':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_025','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+labeled_010':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_010','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+labeled_005':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_005','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+labeled_001':
        cfg.TRAIN.DATASETS = ('bdd_peds_not_clear_any_daytime_train_001','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    ##############################

    # Source domain + Target domain HP tracker bboxes only
    elif args.dataset == 'bdd_peds+HP18k_track_only':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_track_only','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    
    ##### subsets of bdd_HP18k with different constraints
    # Source domain + HP tracker images at NIGHT
    elif args.dataset == 'bdd_peds+HP18k_any_any_night':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_any_any_night','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    
    elif args.dataset == 'bdd_peds+HP18k_rainy_any_daytime':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_rainy_any_daytime','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k_rainy_any_night':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_rainy_any_night','bdd_peds_train')

        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k_overcast,rainy_any_daytime':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_overcast,rainy_any_daytime','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2 
    elif args.dataset == 'bdd_peds+HP18k_overcast,rainy_any_night':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_overcast,rainy_any_night','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    
    elif args.dataset == 'bdd_peds+HP18k_overcast,rainy,snowy_any_daytime':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_overcast,rainy,snowy_any_daytime','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    #############  end of bdd constraned subsets  #####################
    
    # Source domain + Target domain HP18k -- after histogram matching 
    elif args.dataset == 'bdd_peds+HP18k_remap_hist':
        cfg.TRAIN.DATASETS = ('bdd_peds_HP18k_target_domain_remap_hist','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+HP18k_remap_cityscape_hist':
        cfg.TRAIN.DATASETS = ('bdd_peds_HP18k_target_domain_remap_cityscape_hist','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    # Source domain + Target domain HP18k -- after histogram matching 
    elif args.dataset == 'bdd_peds+HP18k_remap_random':
        cfg.TRAIN.DATASETS = ('bdd_peds_HP18k_target_domain_remap_random','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    # Source+Noisy Target domain -- prevent domain adv from using HP roi info
    elif args.dataset == 'bdd_peds+bdd_HP18k_noisy_100k':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_noisy_100k','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+bdd_HP18k_noisy_080':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_noisy_080','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+bdd_HP18k_noisy_060':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_noisy_060','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'bdd_peds+bdd_HP18k_noisy_070':
        cfg.TRAIN.DATASETS = ('bdd_HP18k_noisy_070','bdd_peds_train')
        cfg.MODEL.NUM_CLASSES = 2

    elif args.dataset == "wider_train":
        cfg.TRAIN.DATASETS = ('wider_train',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-subset":
        cfg.TRAIN.DATASETS = ('cs6-subset',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-subset-score":
        cfg.TRAIN.DATASETS = ('cs6-subset-score',)
    elif args.dataset == "cs6-subset-gt":
        cfg.TRAIN.DATASETS = ('cs6-subset-gt',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-3013-gt":
        cfg.TRAIN.DATASETS = ('cs6-3013-gt',) # DEBUG: overfit on one video annots
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-subset-gt+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-subset-gt', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-subset+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-subset', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-gt":
        cfg.TRAIN.DATASETS = ('cs6-train-gt',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-gt-noisy-0.3":
        cfg.TRAIN.DATASETS = ('cs6-train-gt-noisy-0.3',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-gt-noisy-0.5":
        cfg.TRAIN.DATASETS = ('cs6-train-gt-noisy-0.5',)
        cfg.MODEL.NUM_CLASSES = 2

    elif args.dataset == "cs6-train-det-score":
        cfg.TRAIN.DATASETS = ('cs6-train-det-score',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-det-score-0.5":
        cfg.TRAIN.DATASETS = ('cs6-train-det-score-0.5',)
    elif args.dataset == "cs6-train-det":
        cfg.TRAIN.DATASETS = ('cs6-train-det',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-det-0.5":
        cfg.TRAIN.DATASETS = ('cs6-train-det-0.5',)
        cfg.MODEL.NUM_CLASSES = 2
        

    elif args.dataset == "cs6-train-hp":
        cfg.TRAIN.DATASETS = ('cs6-train-hp',)
        cfg.MODEL.NUM_CLASSES = 2

    elif args.dataset == "cs6-train-easy-gt":
        cfg.TRAIN.DATASETS = ('cs6-train-easy-gt',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-easy-gt-sub":
        cfg.TRAIN.DATASETS = ('cs6-train-easy-gt-sub',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-easy-hp":
        cfg.TRAIN.DATASETS = ('cs6-train-easy-hp',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-easy-det":
        cfg.TRAIN.DATASETS = ('cs6-train-easy-det',)
        cfg.MODEL.NUM_CLASSES = 2

        # Joint training with CS6 and WIDER
    elif args.dataset == "cs6-train-easy-gt-sub+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-train-easy-gt-sub', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cs6-train-gt+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-train-gt', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2


    elif args.dataset == "cs6-train-hp+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-train-hp', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2

    elif args.dataset == "cs6-train-dummy+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-train-dummy', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2

    elif args.dataset == "cs6-train-det+WIDER":
        cfg.TRAIN.DATASETS = ('cs6-train-det', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2

    # Dets dataset created by removing tracker results from the HP json
    elif args.dataset == "cs6_train_det_from_hp+WIDER": 
        cfg.TRAIN.DATASETS = ('cs6_train_det_from_hp', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    # Dataset created by removing det results from the HP json -- HP tracker only
    elif args.dataset == "cs6_train_hp_tracker_only+WIDER":
        cfg.TRAIN.DATASETS = ('cs6_train_hp_tracker_only', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    # HP dataset with noisy labels: used to prevent DA from getting any info from HP
    elif args.dataset == "cs6_train_hp_noisy_100+WIDER":
        cfg.TRAIN.DATASETS = ('cs6_train_hp_noisy_100', 'wider_train')
        cfg.MODEL.NUM_CLASSES = 2
    
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    effective_batch_size = args.iter_size * args.batch_size
    print('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

    ### Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

    ### Adjust solver steps
    step_scale = original_batch_size / effective_batch_size
    old_solver_steps = cfg.SOLVER.STEPS
    old_max_iter = cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
    print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                                  old_max_iter, cfg.SOLVER.MAX_ITER))

    # Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
    # of `collect_and_distribute_fpn_rpn_proposals.py`
    #
    # post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
              '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma
    assert_and_infer_cfg()

    timers = defaultdict(Timer)

    """def _init_fn(worker_id):
        random.seed(999)
        np.random.seed(999)
        torch.cuda.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        torch.manual_seed(999)
        torch.backends.cudnn.deterministic = True
    """

    ### Dataset ###
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)
    if cfg.TRAIN.JOINT_TRAINING:
        if len(cfg.TRAIN.DATASETS)==2:
            print('Joint training on two datasets')
        else:
            raise NotImplementedError

        joint_training_roidb = []
        for i, dataset_name in enumerate(cfg.TRAIN.DATASETS):
            # ROIDB construction
            timers['roidb'].tic()
            roidb, ratio_list, ratio_index = combined_roidb_for_training(
                (dataset_name), cfg.TRAIN.PROPOSAL_FILES)
            timers['roidb'].toc()
            roidb_size = len(roidb)
            logger.info('{:d} roidb entries'.format(roidb_size))
            logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

            if i == 0:
                roidb_size = len(roidb)

            batchSampler = BatchSampler(
                sampler=MinibatchSampler(ratio_list, ratio_index),
                batch_size=args.batch_size,
                drop_last=True
            )
            dataset = RoiDataLoader(
                roidb,
                cfg.MODEL.NUM_CLASSES,
                training=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batchSampler,
                num_workers=cfg.DATA_LOADER.NUM_THREADS,
                collate_fn=collate_minibatch)
                #worker_init_fn=_init_fn)
                # decrease num-threads when using two dataloaders
            dataiterator = iter(dataloader)

            joint_training_roidb.append(
                {'dataloader': dataloader, 'dataiterator': dataiterator,
                 'dataset_name': dataset_name}
            )
    else:
        roidb, ratio_list, ratio_index = combined_roidb_for_training(
            cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
        timers['roidb'].toc()
        roidb_size = len(roidb)
        logger.info('{:d} roidb entries'.format(roidb_size))
        logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

        batchSampler = BatchSampler(
            sampler=MinibatchSampler(ratio_list, ratio_index),
            batch_size=args.batch_size,
            drop_last=True
        )
        dataset = RoiDataLoader(
            roidb,
            cfg.MODEL.NUM_CLASSES,
            training=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batchSampler,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            collate_fn=collate_minibatch)
            #worker_init_fn=init_fn)
        dataiterator = iter(dataloader)


    # Effective training sample size for one epoch
    train_size = roidb_size // args.batch_size * args.batch_size

    if cfg.TRAIN.JOINT_SELECTIVE_FG:
        orig_fg_batch_ratio = cfg.TRAIN.FG_FRACTION



    ### Model ###
    maskRCNN = Generalized_RCNN()

    if cfg.CUDA:
        maskRCNN.cuda()

    ### Optimizer ###
    gn_param_nameset = set()
    for name, module in maskRCNN.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_param_nameset.add(name+'.weight')
            gn_param_nameset.add(name+'.bias')
    gn_params = []
    gn_param_names = []
    bias_params = []
    bias_param_names = []
    nonbias_params = []
    nonbias_param_names = []
    nograd_param_names = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
                bias_param_names.append(key)
            elif key in gn_param_nameset:
                gn_params.append(value)
                gn_param_names.append(key)
            else:
                nonbias_params.append(value)
                nonbias_param_names.append(key)
        else:
            nograd_param_names.append(key)
    assert (gn_param_nameset - set(nograd_param_names) - set(bias_param_names)) == set(gn_param_names)

    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]
    # names of paramerters for each paramter
    param_names = [nonbias_param_names, bias_param_names, gn_param_names]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])
        if args.resume:
            args.start_step = checkpoint['step'] + 1
            if 'train_size' in checkpoint:  # For backward compatibility
                if checkpoint['train_size'] != train_size:
                    print('train_size value: %d different from the one in checkpoint: %d'
                          % (train_size, checkpoint['train_size']))

            # reorder the params in optimizer checkpoint's params_groups if needed
            # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            # optimizer.load_state_dict(checkpoint['optimizer'])
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

    ### Training Setups ###
    args.run_name = misc_utils.get_run_name() + '_' + str(cfg.TRAIN.DATASETS)  + '_step'
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)



    ### Training Loop ###
    maskRCNN.train()

    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] >= args.start_step:
            decay_steps_ind = i
            break
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)
    try:
        logger.info('Training starts !')
        step = args.start_step
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

            """
            random.seed(cfg.RNG_SEED)
            np.random.seed(cfg.RNG_SEED)
            torch.cuda.manual_seed(cfg.RNG_SEED)
            torch.cuda.manual_seed_all(cfg.RNG_SEED)
            torch.manual_seed(cfg.RNG_SEED)
            torch.backends.cudnn.deterministic = True
            """

            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR

            # Learning rate decay
            if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                    step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                decay_steps_ind += 1

            training_stats.IterTic()
            optimizer.zero_grad()

            for inner_iter in range(args.iter_size):
                # use a iter counter for optional alternating batches
                if args.iter_size == 1:
                    iter_counter = step
                else:
                    iter_counter = inner_iter

                if cfg.TRAIN.JOINT_TRAINING:
                    # alternate batches between dataset[0] and dataset[1]                    
                    if iter_counter % 2 == 0:
                        if True: #DEBUG:
                            print('Dataset: %s' % joint_training_roidb[0]['dataset_name'])
                        dataloader = joint_training_roidb[0]['dataloader']
                        dataiterator = joint_training_roidb[0]['dataiterator']
                        # NOTE: if available FG samples cannot fill minibatch 
                        # then batchsize will be smaller than cfg.TRAIN.BATCH_SIZE_PER_IM. 
                    else:
                        if True: #DEBUG:
                            print('Dataset: %s' % joint_training_roidb[1]['dataset_name'])
                        dataloader = joint_training_roidb[1]['dataloader']
                        dataiterator = joint_training_roidb[1]['dataiterator']

                try:
                    input_data = next(dataiterator)
                except StopIteration:
                    # end of epoch for dataloader
                    dataiterator = iter(dataloader)
                    input_data = next(dataiterator)
                    if cfg.TRAIN.JOINT_TRAINING:
                        if iter_counter % 2 == 0:
                            joint_training_roidb[0]['dataiterator'] = dataiterator
                        else:
                            joint_training_roidb[1]['dataiterator'] = dataiterator

                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))
                
                net_outputs = maskRCNN(**input_data)
                training_stats.UpdateIterStats(net_outputs, inner_iter)
                loss = net_outputs['total_loss']
                # [p.data.get_device() for p in maskRCNN.parameters()]
                # [(name, p.data.get_device()) for name, p in maskRCNN.named_parameters()]
                loss.backward()

            optimizer.step()
            training_stats.IterToc()

            training_stats.LogIterStats(step, lr)

            if (step+1) % CHECKPOINT_PERIOD == 0:
                save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

        # ---- Training ends ----
        # Save last checkpoint
        save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

    except (RuntimeError, KeyboardInterrupt):
        del dataiterator
        logger.info('Save ckpt on exception ...')
        save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


if __name__ == '__main__':
    main()
