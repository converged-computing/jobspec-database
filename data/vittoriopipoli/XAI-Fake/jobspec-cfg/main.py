import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import random
import time
from datetime import date

import numpy as np
import torch
from torchvision import transforms

from hashlib import shake_256
from munch import munchify, unmunchify
import wandb

from model.model_manager import ModelManager
from dataloader.Caltech import Caltech
from torch.utils.data import Subset, DataLoader
from dataloader.Transforms import create_transforms
from data.fake_dataset_dloader import dataset_splitter
from xai_gradcam import plot_activation
from data.fake_dataset_dloader import dataset_splitter, AiorNotDataset
from dataset_sampling.dataset_statistics import compute_spectr_BW


# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5)  # output len: 2*5=10
    return h.upper()


def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="config.yaml",
                            help="the config file to be used to run the experiment", required=True)
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    arg_parser.add_argument("--project_dir", default="/mnt/beegfs/work/H2020DeciderFicarra/vpipoli/xai_fake/results",
                            type=str, help="Folder where to store the execution")
    arg_parser.add_argument("--xai_grad", action='store_true', help="if evaluate checkpoint with grad cam", default=False)
    arg_parser.add_argument("--annotation_file", default="/mnt/beegfs/work/H2020DeciderFicarra/vpipoli/xai_fake/XAI-Fake/dataset_sampling/dataset_10000.csv")
    arg_parser.add_argument("--aiornot", action='store_true', help="If you want to use dataset of aiornot competition", default=False)
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # start wandb
    wandb.init(
        project="XAI-Fake",
        entity="xai_fake",
        config=unmunchify(config),
        mode=config.wandb.mode
    )

    # Check if project_dir exists
    if not os.path.exists(args.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # check if preprocessing is set and file exists
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preprocessing = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preprocessing = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preprocessing_file:
            preprocessing = yaml.load(preprocessing_file, yaml.FullLoader)
            preprocessing = munchify(preprocessing)

    # check if augmentation is set and file exists
    logging.info(f'loading augmentation')
    if config.data_loader.augmentation is None:
        augmentation = []
    elif not os.path.exists(config.data_loader.augmentation):
        logging.error("augmentation file does not exist: {}".format(config.data_loader.augmentation))
        augmentation = []
    else:
        with open(config.data_loader.augmentation, 'r') as augmentation_file:
            augmentation = yaml.load(augmentation_file, yaml.FullLoader)
            augmentation = munchify(augmentation)
    # make title unique to avoid overriding
    todays_date = date.today()
    config.title = f'{config.title}_YY{todays_date.year}-MM{str(todays_date.month).zfill(2)}-DD{str(todays_date.day).zfill(2)}_{timehash()}'
    parent_directory = os.path.join(args.project_dir, config.title)
    checkpoint_model = os.path.join(parent_directory, 'checkpoint.pt')
    os.makedirs(parent_directory, exist_ok=True)
    logging.info(f'project directory: {parent_directory}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(parent_directory, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(parent_directory, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    mm = ModelManager(config)
    transf_train = create_transforms(preprocessing, augmentation, config, eval=False, compose=True)
    transf_eval = create_transforms(preprocessing, augmentation, config, eval=True, compose=True)
    # THE FOLLOWING TRANSFORMATIONS MUST BE CREATED ACCORDINGLY TO THE DATALOADER/TRANSFORMS.PY, PREPROCESSING, AUGMENTATIONS AND CONFIG(DATALOADER.NORMALIZE) YAML FILES
    # THE FOLLOWING IS A TOY DATASET 
    # MOST OF THE FOLLOWING INSTRUCTIONS MUST BE WRAPPED IN A DATALOADER CLASS
    load_spectrum = True
    if load_spectrum == True:
        fft_img_real_mean = np.load('/mnt/beegfs/work/H2020DeciderFicarra/vpipoli/xai_fake/XAI-Fake/fft_img_real_mean.npy')
        fft_img_fake_mean = np.load('/mnt/beegfs/work/H2020DeciderFicarra/vpipoli/xai_fake/XAI-Fake/fft_img_fake_mean.npy')
        
    else:
        logging.info("COMPUTING SPECTR AND MEANS...")
        fft_img_real_mean, fft_img_fake_mean = compute_spectr_BW(args.annotation_file)
        np.save("/mnt/beegfs/work/H2020DeciderFicarra/vpipoli/xai_fake/XAI-Fake/fft_img_real_mean.npy", fft_img_real_mean, allow_pickle=True, fix_imports=True)
        np.save("/mnt/beegfs/work/H2020DeciderFicarra/vpipoli/xai_fake/XAI-Fake/fft_img_fake_mean.npy", fft_img_fake_mean, allow_pickle=True, fix_imports=True)
        logging.info("SPECTR COMPUTED!")

    # spectrum_shift_eval={"fft_img_real_mean":fft_img_real_mean, "fft_img_fake_mean":fft_img_fake_mean}
    spectrum_shift_eval=False
    if not args.aiornot:
        train_dataset, test_dataset, eval_dataset = dataset_splitter(args.annotation_file, split_size=0.8,
                                                                     transform_train=transf_train,
                                                                     transform_test=transf_eval, 
                                                                     spectrum_shift_eval=spectrum_shift_eval,
                                                                     seed=config.seed)
    else:
        ds = AiorNotDataset(train_transform=transf_train,test_transform=transf_eval, seed=config.seed, split_size=0.8)
        train_dataset, test_dataset, eval_dataset = ds.aiornot_dataset()
    # CREATE DATALOADERS
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.data_loader.batch_size, 
                                  shuffle=True,
                                  num_workers=config.data_loader.num_workers, drop_last=True)
    val_dataloader = DataLoader(eval_dataset, 
                                batch_size=config.data_loader.batch_size, 
                                shuffle=False, num_workers=config.data_loader.num_workers)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config.data_loader.batch_size, 
                                 shuffle=False,
                                 num_workers=config.data_loader.num_workers)

    if config.trainer.reload and not os.path.exists(config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {config.trainer.checkpoint}')
        raise SystemExit
    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        mm.train(train_dataloader, val_dataloader, debug=args.debug, checkpoint=checkpoint_model)
        mm.evaluate(test_dataloader, checkpoint=checkpoint_model, best=True)

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        mm.evaluate(test_dataloader, checkpoint=config.trainer.checkpoint, best=True)

    # Test the model
    if config.trainer.do_inference:
        logging.info('Inference...')
        mm.evaluate(test_dataloader, checkpoint=config.trainer.checkpoint)

    if args.xai_grad:
        plot_activation(mm.net, test_dataset, parent_directory, config)
