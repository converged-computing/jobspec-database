# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
from collections import OrderedDict
import torch
import time
import numpy as np

from src.utils import bool_flag, initialize_exp
from src.models import build_bdma_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_10'
VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_10-S2T-10000'

# main
parser = argparse.ArgumentParser(description='Bidirectional training')
parser.add_argument("--seed", type=int, default=0, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--iterative", type=bool_flag, default=False, help="Iterative building of dictionary.")
parser.add_argument("--bidirectional", type=bool_flag, default=False, help="Bidirectional Manifold Alignment.")
parser.add_argument("--shared", type=bool_flag, default=True, help="Shared reverse parameters.")
parser.add_argument("--vocab_size", type=int, default=0, help="Size of Vocabulary for training.")
parser.add_argument("--descending", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--loss", choices=['m', 'r', 'mr'], default='m', help="Types of losses.")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")

# Data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")

# BDMA Network
parser.add_argument("--n_layers", type=int, default=0, help="BDMA layers")
parser.add_argument("--n_hid_dim", type=int, default=1024, help="BDMA hidden layer dimensions")
parser.add_argument("--n_dropout", type=float, default=0., help="BDMA dropout")
parser.add_argument("--n_rev_beta", type=float, default=1.0, help="BDMA Reverse Loss Learning Rate")
parser.add_argument("--n_input_dropout", type=float, default=0.1, help="BDMA input dropout")
parser.add_argument("--n_steps", type=int, default=5, help="BDMA steps")
parser.add_argument("--n_lambda", type=float, default=1, help="BDMA loss feedback coefficient")
parser.add_argument("--n_smooth", type=float, default=0.1, help="BDMA smooth predictions")
parser.add_argument("--map_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

# Training refinement.
parser.add_argument("--n_refinement", type=int, default=10, help="Number of refinement iterations (0 to disable the refinement procedure)")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="adam,lr=0.0005", help="Mapping optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

# Dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")

# Reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm", help="Normalize embeddings before training")

# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default", "combined"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or params.dico_eval == 'combined' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# Check BDMA Parameters.
assert 0 <= params.n_dropout < 1
assert 0 <= params.n_input_dropout < 1
assert 0 <= params.n_smooth < 0.5
assert params.n_lambda > 0 and params.n_steps > 0
assert 0 < params.lr_shrink <= 1

# Torch random values.
torch.manual_seed(params.seed)
np.random.seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping = build_bdma_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
evaluator = Evaluator(trainer)

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train, size=params.vocab_size,
                           descending=params.descending)

# define the validation metric.
VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
logger.info("Validation metric: %s" % VALIDATION_METRIC)

# Check.
if params.iterative:
    logger.info("Enabling Iterative Training...")
else:
    logger.info("No Iterative Training...")

"""
Learning loop with and without BDMA Iterative Learning.
"""
for n_iter in range(params.n_refinement + 1):
    tic = time.time()
    stats = {'MSE_COSTS': []}
    logger.info('Starting iteration %i...' % n_iter)

    # Build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if params.iterative:
        if n_iter > 0 or not hasattr(trainer, 'dico'):
            trainer.build_dictionary()

    if n_iter == 0 and params.n_layers == 0:
        # NOTE: Perform procrustes operation when its a linear op.
        trainer.bdma_procrustes()

    # Standard MSE loss network.
    f_loss, b_loss, n = trainer.bdma_step(stats)
    logger.info('Average F loss: {}, Average B loss: {}, Num Batches: {}'.format(f_loss, b_loss, n))

    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best_bdma(to_log, VALIDATION_METRIC)
    logger.info('End of iteration %i.\n\n' % n_iter)

    # update the learning rate (stop if too small)
    trainer.update_lr(to_log, VALIDATION_METRIC)
    if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
        logger.info('Learning rate < 1e-6. BREAK.')
        break


# export embeddings
if params.export:
    trainer.reload_best_bdma()
    trainer.export()
