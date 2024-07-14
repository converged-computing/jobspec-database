import argparse
import os
import time
from typing import Any, Union, List, Optional

import numpy as np
import torch
from torchvision.transforms import ToTensor, Compose, Resize
from sklearn.decomposition import PCA

import helpers
from things import THINGSBehavior

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--things_root", type=str, help="path/to/things")
    aa("--data_root", type=str, help="path/to/cwd")
    aa(
        "--model",
        type=str,
        nargs="+",
        help="Which model to use. For multiple models, features will be stacked.",
    )
    aa(
        "--module",
        type=str,
        default="penultimate",
        nargs="+",
        help="DNN module for which to learn a linear transform. For multiple modules, features will be stacked.",
    )
    aa(
        "--source",
        type=str,
        default="torchvision",
        nargs="+",
        choices=[
            "custom",
            "torchvision",
            "diffusion",
        ],
    )
    aa(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dot"],
        help="distance function used for predicting the odd-one-out",
    )
    aa(
        "--overwrite",
        action="store_true",
        help="whether or not to overwrite existing results",
    )
    aa(
        "--cc0",
        action="store_true",
        help="whether or not to use CC0 files",
    )
    aa("--pca", type=int, help="dimensionality of the PCA", default=None)
    args = parser.parse_args()
    return args


def load_dataset(data_dir: str, transform=None, cc0=False):
    dataset = THINGSBehavior(
        root=data_dir, aligned=False, download=False, transform=transform, cc0=cc0
    )
    return dataset


def eval_alignment(
    sources: List[str],
    models: List[str],
    module_types: List[str],
    things_dir: str,
    path_to_embeddings: str,
    path_to_alignment: str,
    dist: str = "cosine",
    cc0: bool = False,
    save: bool = True,
    overwrite: bool = False,
    pca: Optional[int] = None
):
    stacked_models = "+".join(models)
    stacked_module_types = "+".join(module_types)
    stacked_sources = "+".join(sources) if any([s != sources[0] for s in sources]) else sources[0]

    try:
        existing_results = np.load(path_to_alignment, allow_pickle=True)
        print("Found an existing results file at", path_to_alignment)
    except FileNotFoundError:
        existing_results = None

    skip = False
    if not overwrite and existing_results is not None:
        print("Looking up results for:", stacked_sources, stacked_models, stacked_module_types)
        try:
            acc = existing_results[stacked_sources][stacked_models][stacked_module_types]
            print("Results exist -- skipping.")
            print("Zero-shot odd-one-out accuracy:", acc)
            skip = True
        except KeyError:
            print("Results do not exist -- continuing.")
            pass

    if not skip:
        transform = Compose([Resize(512), ToTensor()])
        dataset = load_dataset(things_dir, transform=transform, cc0=False)
        triplets = dataset.get_triplets()
        print("There are %d triplets to be evaluated." % len(triplets))

        all_features = []  # In case of stacked features
        for source, model_name, module_type in zip(sources, models, module_types):
            print(source, model_name, module_type, "cc0 =", cc0)

            try:
                features = helpers.load_features(
                        path=path_to_embeddings,
                        model_name=model_name,
                        module_type=module_type,
                        source=source,
                        cc0=cc0,
                )
            except FileNotFoundError:
                print("   Features not found. Stopping evaluation.")
                skip = True
                break

            if features[0].dtype == np.float16:
                print("   Converting to float32.")
                features = np.array([np.float32(ft) for ft in features])

            all_features.append(features)

        if not skip:
            features = np.concatenate(all_features, axis=-1)

            if pca is not None:
                print(f"Performing PCA ({pca})")
                pca_algo = PCA(n_components=min(pca, min(features.shape[0], features.shape[1])))
                pca_algo.fit(features)
                features = pca_algo.transform(features)
                print("  New dimensionality:", features.shape)

            choices, probas = helpers.get_predictions(features, triplets, temperature=1.0, dist=dist)
            acc = helpers.accuracy(choices)
            print("Zero-shot odd-one-out accuracy:", acc)

            if save:
                print("Saving to", path_to_alignment)
                helpers.save_pickle(
                    acc, path_to_alignment, stacked_models, stacked_sources, stacked_module_types, per_model_path=False
                )


if __name__ == "__main__":
    args = parseargs()

    path_to_alignment = os.path.join(args.data_root, "alignment.pkl")
    path_to_embeddings = args.data_root

    sources = [args.source] if type(args.source) == str else args.source
    models = [args.model] if type(args.model) == str else args.model
    module_types = [args.module] if type(args.module) == str else args.module

    eval_alignment(
        sources=sources,
        models=models,
        module_types=module_types,
        things_dir=args.things_root,
        path_to_embeddings=path_to_embeddings,
        path_to_alignment=path_to_alignment,
        dist=args.distance,
        cc0=args.cc0,
        save=True,
        overwrite=args.overwrite,
        pca=args.pca
    )
