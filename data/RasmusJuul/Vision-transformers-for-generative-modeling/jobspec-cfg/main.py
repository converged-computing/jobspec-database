import argparse
import os
import sys

import torch

from src.models.train_model import main as train


def main(
    name: str = "test",
    model_type: str = "ViTVAE",
    max_epochs: int = 10,
    num_workers: int = 0,
    dim: int = 256,
    depth: int = 12,
    heads: int = 16,
    mlp_dim: int = 256,
    lr: float = 5e-5,
    patch_size: int = 16,
    batch_size: int = 256,
    ngf: int = 8,
    generator: str = "ViTVAE"
):
    torch.cuda.empty_cache()

    train(
        name=name,
        model_type=model_type,
        max_epochs=max_epochs,
        num_workers=num_workers,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        lr=lr,
        patch_size=patch_size,
        batch_size=batch_size,
        ngf=ngf,
        generator=generator
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a model (ViT-VAE, ViT, DeepViT) on CelebA"
    )
    parser.add_argument("--name", "-n", type=str, help="Name for wandb")
    parser.add_argument(
        "--model-type",
        "-mt",
        type=str,
        help="Model type to train either ViTVAE or Classifier",
    )
    parser.add_argument("--max-epochs", "-me", type=int, help="Number of max epochs")
    parser.add_argument(
        "--num-workers", "-nw", type=int, help="Number of threads use in loading data"
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="Last dimension of output tensor after linear transformation",
    )
    parser.add_argument("--depth", type=int, help="Number of Transformer blocks")
    parser.add_argument(
        "--heads", type=int, help="Number of heads in Multi-head Attention layer"
    )
    parser.add_argument(
        "--mlp_dim", type=int, help="Dimension of the MLP (FeedForward) layer"
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate"
    )
    parser.add_argument(
        "--patch-size",
        "--ps",
        type=int,
        help="Number of patches. image_size must be divisible by patch_size. The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16. (aka patch_size can't be more than 8 for cifar10",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        help="affects size of decoder",
    )
    parser.add_argument(
        "--gen",
        type=str,
        help="Which generator should be use to generate image for the classifier",
    )
    args = parser.parse_args()

    name = "test"
    model_type = "ViTVAE"
    max_epochs = 10
    num_workers = 0
    dim = 128
    depth = 12
    heads = 16
    mlp_dim = 128
    lr = 5e-5
    patch_size = 16
    batch_size = 256
    ngf = 8
    generator = "ViTVAE"

    if args.name:
        name = args.name
    if args.model_type:
        model_type = args.model_type
    if args.max_epochs:
        max_epochs = args.max_epochs
    if args.num_workers:
        num_workers = args.num_workers
    if args.dim:
        dim = args.dim
    if args.depth:
        depth = args.depth
    if args.heads:
        heads = args.heads
    if args.mlp_dim:
        mlp_dim = args.mlp_dim
    if args.lr:
        lr = args.lr
    if args.patch_size:
        patch_size = args.patch_size
    if args.batch_size:
        batch_size = args.batch_size
    if args.ngf:
        ngf = args.ngf
    if args.gen:
        generator = args.gen

    main(
        name=name,
        model_type=model_type,
        max_epochs=max_epochs,
        num_workers=num_workers,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        lr=lr,
        patch_size=patch_size,
        batch_size=batch_size,
        ngf=ngf,
        generator=generator
    )
