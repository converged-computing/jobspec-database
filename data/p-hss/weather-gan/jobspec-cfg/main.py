from argparse import ArgumentParser, Namespace

from src.configuration import Config
from src.trainer import training

def main(args: Namespace) -> None:
    config = Config()
    training(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=10,
                        help="dimensionality of the latent space")
    hparams = parser.parse_args()

    main(hparams)