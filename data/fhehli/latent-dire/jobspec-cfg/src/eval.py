import argparse
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, ImageFolder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.data_loader import ldire_loader
from src.nn.resnet50 import preprocess_resnet50_pixel, preprocess_resnet50_latent
from src.training import Classifier


def main(args: argparse.Namespace) -> None:
    # Setup Weights & Biases
    wandb_logger = WandbLogger(name=args.name, project="Eval", entity="latent-dire", config=vars(args))

    # Load the data
    if args.model in ["resnet50_latent"]:
        transform = preprocess_resnet50_latent
    elif args.model in ["resnet50_pixel"]:
        transform = preprocess_resnet50_pixel
    if args.type == "images":
        dataset = ImageFolder(args.data_dir, transform=transform)
    elif args.type == "latent":
        dataset = DatasetFolder(args.data_dir, transform=transform, loader=ldire_loader, extensions=(".npz",))
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    clf = Classifier.load_from_checkpoint(args.ckpt)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",  # use all available GPUs
        precision="16-mixed",
        default_root_dir=f"models/{args.name}",
        logger=wandb_logger,
        log_every_n_steps=5,
        fast_dev_run=args.dev_run,
    )
    trainer.test(clf, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev_run", action="store_true", help="Whether to run a test run.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        required="-d" not in sys.argv and "--dev-run" not in sys.argv,
        help="A descriptive name for the run, for wandb.",
    )
    parser.add_argument("--type", type=str, required=True, choices=["images", "latent"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["resnet50_pixel", "resnet50_latent"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for the data loader.")
    args = parser.parse_args()

    main(args)
