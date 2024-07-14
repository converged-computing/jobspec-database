"""Painter by Numbers.
"""
import tensorflow as tf

from core import boot
from core.utils import str2bool

boot.gpus_with_memory_growth()

from argparse import ArgumentParser

import numpy as np

parser = ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--override", default=False, type=str2bool)
parser.add_argument("--mixed_precision", default=False, type=str2bool)
parser.add_argument("--jit_compile", default=False, type=str2bool)

# Dataset
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--patch_size", default=299, type=int)
parser.add_argument("--data_split", default="frequent", choices=["original", "frequent"])
parser.add_argument("--data_frequent_test_size", default=0.1, type=float)

# Preprocess
parser.add_argument("--preprocess_max_size", default=6000, type=int)
parser.add_argument("--preprocess_workers", default=24, type=int)

# Features (Backbone)
parser.add_argument("--strategy", default="ce", choices=["ce", "supcon", "supcon_mh"])
parser.add_argument("--backbone_architecture", default="InceptionV3")
parser.add_argument("--backbone_features_layer", default="avg_pool")

parser.add_argument("--logs_dir", default="./experiments/logs/")
parser.add_argument("--weights_dir", default="./experiments/weights/")
parser.add_argument("--preds_dir", default="./experiments/predictions/")

## Features (Backbone) Training
parser.add_argument("--backbone_valid_split", type=float, default=0.1)
parser.add_argument("--backbone_valid_seed", type=int, default=581)
parser.add_argument("--backbone_train_workers", type=int, default=8)
parser.add_argument("--backbone_optimizer", default="momentum", type=str, choices=["sgd", "adam", "momentum"])

parser.add_argument("--backbone_train_epochs", default=5, type=int)
parser.add_argument("--backbone_train_lr", default=0.01, type=float)
parser.add_argument("--backbone_finetune_epochs", default=100, type=int)
parser.add_argument("--backbone_freezebn", default=False, type=str2bool)
parser.add_argument("--backbone_finetune_layers", default="all")  # ResNet Arch => first: "conv1_pad" b4: "conv4_block1_preact_bn"
parser.add_argument("--backbone_finetune_lr", default=0.01, type=float)
parser.add_argument("--backbone_train_supcon_temperature", default=0.1, type=float)

## Features (Backbone) Inference
parser.add_argument("--patches_train", default=2, type=int)  # 2 random crops are the default in SupCon (arxiv 2004.11362).
parser.add_argument("--positives_train", default=1, type=int)  # Positives from different sources.
parser.add_argument("--patches_test", default=20, type=int)  # 2 random crops are the default in SupCon (arxiv 2004.11362).
parser.add_argument("--batch_test", default=2, type=int)  # Usually small because `patches_test` is large.
parser.add_argument("--features_parts", default=4, type=int)  # default: 4

# Adversarial Feature Hallucination Network
parser.add_argument("--afhp_problem", default="painter", type=str, choices=["painter", "style", "genre"])
parser.add_argument("--afhp_lr", default=1e-5, type=float)
parser.add_argument("--afhp_epochs", default=100, type=int)
parser.add_argument("--afhp_persist", default=True, type=str2bool)
parser.add_argument("--afhp_gp_w", default=10.0, type=float)
parser.add_argument("--afhp_ac_w", default=1.0, type=float)
parser.add_argument("--afhp_cf_w", default=1.0, type=float)

# Steps
parser.add_argument("--step_preprocess_reduce", default=True, type=str2bool)
parser.add_argument("--step_features_train", default=True, type=str2bool)
parser.add_argument("--step_features_infer", default=True, type=str2bool)
parser.add_argument("--step_afhp_train", default=True, type=str2bool)
parser.add_argument("--step_afhp_test", default=True, type=str2bool)


def run(args):
  import datasets.pbn

  print(__doc__)
  print("=" * 65)
  print(*(f"{k:<20} = {v}" for k, v in vars(args).items()), sep="\n")
  print("=" * 65)

  dataset = datasets.pbn.get_dataset(args)

  if args.step_preprocess_reduce:
    import steps.preprocess_step
    steps.preprocess_step.reduce_massive_images(dataset, args)

  train_info, test_info = datasets.pbn.train_test_split(dataset, args)

  if args.step_features_train or args.step_features_infer:
    if args.strategy == "ce":
      import steps.backbone_step
      steps.backbone_step.vanilla(train_info, dataset, STRATEGY, args)

    elif args.strategy == "supcon":
      import steps.backbone_step
      steps.backbone_step.supcon(train_info, dataset, STRATEGY, args)

    else:
      import steps.backbone_step
      steps.backbone_step.supcon_mh(train_info, dataset, STRATEGY, args)

  if args.step_afhp_train or args.step_afhp_test:
    import steps.afhp_step
    afhp_model, _, test_data = steps.afhp_step.run(train_info, dataset, STRATEGY, args)

    if args.step_afhp_test:
      import steps.fsl_step
      steps.fsl_step.run(args, afhp_model, *test_data)


if __name__ == "__main__":
  args = parser.parse_args()

  np.random.seed(args.seed + 42)

  STRATEGY = boot.appropriate_distributed_strategy()

  if args.mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  run(args)
