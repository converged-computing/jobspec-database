''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser

def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='WT64',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)') 
  parser.add_argument(
    '--norm_path', type=str, default='./',
    help='Default location where norm values file is stored (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser

def run(config):
  # Get loader
  config['drop_last'] = False
  loaders = utils.get_data_loaders(**config)

  # Load inception net
  net = inception_utils.load_inception_net(parallel=config['parallel'])
  pool, logits, labels = [], [], []
  pool_iwt, logits_iwt, labels_iwt = [], [], []

  device = 'cuda'
  filters = utils.create_filters(device=device)
  inv_filters = utils.create_inv_filters(device=device)

  norm_dict = utils.load_norm_dict(config['norm_path'])
  shift, scale = torch.from_numpy(norm_dict['shift']).to(device), torch.from_numpy(norm_dict['scale']).to(device)

  for i, (x, y) in enumerate(tqdm(loaders[0])):
    x = utils.wt(x.to(device), filters, levels=2)[:, :, :64, :64]
    x_full = x.clone()
    x = utils.normalize(x, shift, scale)

    x_full = utils.zero_pad(x_full, 256, device) # Full image size hard-coded
    x_full = utils.iwt(x_full, inv_filters, levels=2)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]

      pool_val_iwt, logits_val_iwt = net(x_full)
      pool_iwt += [np.asarray(pool_val_iwt.cpu())]
      logits_iwt += [np.asarray(F.softmax(logits_val_iwt, 1).cpu())]
      labels_iwt += [np.asarray(y.cpu())]

  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  pool_iwt, logits_iwt, labels_iwt = [np.concatenate(item, 0) for item in [pool_iwt, logits_iwt, labels_iwt]]
  # uncomment to save pool, logits, and labels to disk
  # print('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  print('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  print('Training data (wt, normalized) from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))

  IS_mean_iwt, IS_std_iwt = inception_utils.calculate_inception_score(logits_iwt)
  print('Training data (iwt, denormalized) from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean_iwt, IS_std_iwt))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default 
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  mu_iwt, sigma_iwt = np.mean(pool_iwt, axis=0), np.cov(pool_iwt, rowvar=False)
  print('Saving calculated means and covariances to disk...')

  # WT is normalized, 64x64
  # IWT is denormalized, 256x256
  np.savez(config['dataset'].strip('_hdf5')+'_wt_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})
  np.savez(config['dataset'].strip('_hdf5')+'_iwt_inception_moments.npz', **{'mu' : mu_iwt, 'sigma' : sigma_iwt})

def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()