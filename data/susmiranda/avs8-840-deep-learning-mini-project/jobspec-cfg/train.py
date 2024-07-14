"""Script for training KWT model"""
from argparse import ArgumentParser
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup
import wandb
from utils.trainer import train, evaluate
from src.data.dataset import SpectrogramDataset
#from utils.spectrogram_dataset import SpectrogramDataset
from utils.config_parser import parse_config
from src.models.KWT import KWT
from utils.misc import seed_everything, count_params, get_model, calc_step, log
#from torch.nn.parallel import DataParallel #probably too much overhead, is too slow with even 2 GPUs
import matplotlib.pyplot as plt
import os

""""
Run script with configuration file as argument

command: srun --time=4:00:00 --gres=gpu:1 singularity exec --nv <your_singularity_container_here> python train.py --conf KWT_configs/KWT_config.cfg

srun --gres=gpu:1 singularity exec --nv ~/pytorch-24.01/ python train.py --conf KWT_configs/KWT_config.cfg --ckpt ../models/ssformer_200E_valid.ckpt
"""

def training_pipeline(config):
    """
    Initiates and executes all the steps involved with KWT model training
    :param config: KWT configuration
    """
    #####################################
    # Training initialzation
    #####################################
 # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Initialize KWT
    model = KWT(**config['hparams']['KWT'])
    #model = DataParallel(model) # Wrapping model in DataParallel class
    model.to(device); # Sending device to GPU

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        # print(checkpoint.keys())  # Print out the keys
        # Look for the correct key for the model state di
        model.load_state_dict(checkpoint['state_dict'])

    # Make dataset
    train_set = SpectrogramDataset(config['tr_manifest_path'], config['labels_map'], config['audio_config'])
    val_set = SpectrogramDataset(config['val_manifest_path'], config['labels_map'], config['audio_config'])
    test_set = SpectrogramDataset(config['eval_manifest_path'],config['labels_map'],config['audio_config'])

    if config['dev_mode']:
        train_set.files = train_set.files[:100]
        train_set.labels = train_set.labels[:100]
        train_set.len = len(train_set.files)
        val_set.files = val_set.files[:50]
        val_set.labels = val_set.labels[:50]
        val_set.len = len(val_set.files)
        test_set.files = test_set.files[:50]
        test_set.labels = test_set.labels[:50]
        test_set.len = len(test_set.files)
        config['hparams']['batch_size'] = 25

      # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'], 
    sampler= train_set.weighted_sampler, num_workers=3)
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'],
    sampler= val_set.weighted_sampler, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=config['hparams']['batch_size'])

    #classes = ('COVID-19', 'healthy', 'symptomatic')
    classes = {"COVID-19": 0, "healthy": 1, "symptomatic": 2}
    
    # Cross Entropy loss
    criterion = nn.CrossEntropyLoss() # multi class 
    #criterion = nn.BCEWithLogitsLoss() #multi label classification (FDS50K)

    # optimizer
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=config["hparams"]["optimizer"]["lr"],
                           betas=config["hparams"]["optimizer"]["betas"],
                           eps=config["hparams"]["optimizer"]["eps"],
                           weight_decay=config["hparams"]["optimizer"]["weight_decay"])
    
    #optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    
    # # Make a scheduler
    cosineWarmup = get_cosine_schedule_with_warmup(optimizer, config["hparams"]["scheduler"]["n_warmup"], config["hparams"]["n_epochs"])
    schedulers = {'scheduler': cosineWarmup,
                  'warmup': cosineWarmup}
    

    #####################################
    # Training Run
    #####################################

      # Training loop
    print("Initiating training.")
 
    train(model, optimizer, criterion, train_loader, val_loader, schedulers, config) 
    
    #####################################
    # Final Test
    #####################################

    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(train_loader), len(train_loader) - 1)

    # evaluating the final state (last.pth)
    test_acc, test_loss = evaluate(model, criterion, test_loader, config["hparams"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_acc_last": test_acc
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(
        os.path.join(config["exp"]["save_dir"], "best.pth"), map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_acc, test_loss = evaluate(model, criterion, test_loader, config["hparams"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_acc_best": test_acc
    }
    log(log_dict, final_step, config)


def main(args):
    """
    Calls training pipeline and sets up wandb logging if used
    :param args: input arguments
    
    """

    config = parse_config(args.conf)

    

    # below code is for logging everything in wandb
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id
    config['dev_mode'] = args.dev_mode

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
        
        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")
        
        else:
            wandb.login()

        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"],entity=config["exp"]["entity"]):
            training_pipeline(config)
    else:

        training_pipeline(config)

if __name__ == "__main__":
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
            
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Obtional experiment identifier.", default=None)
    parser.add_argument('--dev_mode', action='store_true', help='Flag to limit the dataset for testing purposes.')
    args = parser.parse_args()

    
    main(args)

    
