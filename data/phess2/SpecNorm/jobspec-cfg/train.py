# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import pickle
import yaml
import json
import socket
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from src.classifiers import *
from corpus.CIFAR100_dataset import *

hostname = socket.gethostname()
seed_everything(1)
torch.set_float32_matmul_precision('medium')


def run_train(args):
    # if (args.config.endswith(".json")):
    #     with open(args.config, 'r') as file:
    #         config = json.load(file)
    # elif (args.config.endswith(".yaml")):
    #     config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    # else:
    #     print("config file type not supported")
    #     print(args.config)
    #     return
    with open('model_configs_12_11.pkl', 'rb') as f:
        config_paths = pickle.load(f)
    con_path = config_paths[args.job_id]

    with open(con_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if 'data' in config.keys():
        if args.gpus > 0:
            config['data']['loader']['batch_size'] = config['data']['loader']['batch_size'] // args.gpus
        else:
            config['data']['loader']['batch_size'] = 1
    else:
        if args.gpus > 0:
            config['loader']['batch_size'] = config['loader']['batch_size'] // args.gpus
        else:
            config['data']['loader']['batch_size'] = 1

    if config['model_type'] == 'ResNet':
        exp_dir = pathlib.Path(f"{config['model_type']}" + "/" + f"{config['model_size']}_{config['norm_type']}")
    else:
        exp_dir = pathlib.Path(f"{config['model_type']}" + "/" + f"{config['num_layers']}_{config['width']}_{config['norm_type']}")
    checkpoint_dir = exp_dir / "checkpoints"

    if args.ckpt_path != '':
        ckpt_path = checkpoint_dir / args.ckpt_path
    else:
        ckpt_path = None

    dm = CIFAR100DataModule(num_workers=args.num_workers, batch_size=config['data']['loader']['batch_size'])

    if config['model_type'] == 'ResNet':
        if ckpt_path is not None:
            model = CIFAR100_Resnet.load_from_checkpoint(checkpoint_path=ckpt_path, model_size=config['model_size'], norm=config['norm_type'], lr=config['hparas']['lr'])
        else:
            model = CIFAR100_Resnet(config['model_size'], config['norm_type'], config['hparas']['lr'])
    else:
        if ckpt_path is not None:
            model = CIFAR100_MLP.load_from_checkpoint(checkpoint_path=ckpt_path, num_layers=config['num_layers'], width=config['width'], norm=config['norm_type'], lr=config['hparas']['lr'])
        else:
            model = CIFAR100_MLP(config['num_layers'], config['width'], config['norm_type'], config['hparas']['lr'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=10, save_top_k=-1, save_weights_only=True)
    trainer = Trainer(default_root_dir=exp_dir, 
                      callbacks=checkpoint_callback, 
                      devices=args.gpus, 
                      num_nodes=args.num_nodes, 
                      max_epochs=config['hparas']['epochs'],
                      accelerator="gpu" if args.gpus > 0 else 'cpu',
                      )
    trainer.fit(model, dm)
    trainer.save_checkpoint(checkpoint_dir / 'final.ckpt')
    trainer.test(model, datamodule = dm)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str, help='Path to experiment config.')
    parser.add_argument('--job_id', type=int, help='Job ID for the experiment.')
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Resume training from this checkpoint."
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
