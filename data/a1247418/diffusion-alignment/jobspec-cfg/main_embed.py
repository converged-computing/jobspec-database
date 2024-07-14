import argparse
from typing import Any

import numpy as np
import torch

from embedding import embed

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--things_root", type=str, help="path/to/things")
    aa("--data_root", type=str, help="path/to/cwd")
    aa("--path_to_model_dict", type=str, help="path/to/model_dict.json")
    aa("--path_to_caption_dict", type=str, help="path/to/captions_dict.npy", default=None)
    aa(
        "--model",
        type=str,
        help="Which model to use. For multiple models, features will be stacked.",
    )
    aa(
        "--module",
        type=str,
        default="penultimate",
        help="DNN module for which to learn a linear transform. For multiple modules, features will be stacked.",
    )
    aa(
        "--source",
        type=str,
        default="torchvision",
        choices=[
            "custom",
            "torchvision",
            "diffusion",
        ],
    )
    aa(
        "--overwrite",
        action="store_true",
        help="whether or not to overwrite existing results",
    )
    aa(
        "--not_pretrained",
        action="store_true",
        help="whether or not to use pretrained models",
    )
    aa(
        "--cc0",
        action="store_true",
        help="whether or not to use CC0 files",
    )
    aa(
        "--pool",
        action="store_true",
        help="whether or not to pool representations spatially",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()

    embed(
        path_to_embeddings=args.data_root,
        path_to_things=args.things_root,
        path_to_model_dict=args.path_to_model_dict,
        model_name=args.model,
        source=args.source,
        module_type=args.module,
        cc0=args.cc0,
        overwrite=args.overwrite,
        pretrained=not args.not_pretrained,
        pool=args.pool or args.source == "diffusion",
        path_to_caption_dict=args.path_to_caption_dict,
    )
