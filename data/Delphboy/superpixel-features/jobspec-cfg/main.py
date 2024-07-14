import argparse
import logging
import os

import numpy as np
import torch

from features import (
    get_blip_patch_features,
    get_blip_superpixel_features,
    get_blip_whole_img_features,
    get_clip_patch_features,
    get_clip_superpixel_features,
    get_clip_whole_img_features,
    get_resnet_patch_features,
    get_resnet_superpixel_features,
    get_resnet_whole_img_features,
)
from superpixels import get_patches, get_superpixels, load_image
from rag import create_rag_edges

LOGGER = None


def get_logger():
    global LOGGER
    if LOGGER is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        LOGGER = logger
    return LOGGER


def process_superpixels(
    image_dir: str,
    output_dir: str,
    num_superpixels: int,
    model_id: str,
    superpixel_algo: str = "SLIC",
):
    # Get the images in the directory
    images = os.listdir(image_dir)
    dev = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device set to {dev}")
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        scikit_image, torch_image = load_image(os.path.join(image_dir, image))
        superpixels = get_superpixels(
            img_scikit=scikit_image, n_segments=num_superpixels, algo=superpixel_algo
        )

        if model_id == "CLIP":
            features, bounding_boxes = get_clip_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=512,
            )
        elif model_id == "BLIP":
            features, bounding_boxes = get_blip_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=768,
            )
        else:
            features, bounding_boxes = get_resnet_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=2048,
            )

        features = features.squeeze(0).cpu().numpy()
        bounding_boxes = bounding_boxes.squeeze(0).cpu().numpy()
        feats = {"feat": features, "bbox": bounding_boxes}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


def process_rag(
    image_dir: str,
    output_dir: str,
    num_superpixels: int,
    model_id: str,
    superpixel_algo: str = "SLIC",
):
    # Get the images in the directory
    images = os.listdir(image_dir)
    dev = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device set to {dev}")
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        scikit_image, torch_image = load_image(os.path.join(image_dir, image))
        superpixels = get_superpixels(
            img_scikit=scikit_image, n_segments=num_superpixels, algo=superpixel_algo
        )

        if model_id == "CLIP":
            features, bounding_boxes = get_clip_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=512,
            )
        elif model_id == "BLIP":
            features, bounding_boxes = get_blip_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=768,
            )
        else:
            features, bounding_boxes = get_resnet_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=2048,
            )

        features = features.squeeze(0).cpu().numpy()
        bounding_boxes = bounding_boxes.squeeze(0).cpu().numpy()
        edges = create_rag_edges(scikit_image, superpixels.cpu().numpy())
        feats = {"feat": features, "bbox": bounding_boxes, "rag": edges}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


def process_patches(
    image_dir: str,
    output_dir: str,
    model_id: str,
):
    # Get the images in the directory
    images = os.listdir(image_dir)
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        _, torch_image = load_image(os.path.join(image_dir, image))
        patches = get_patches(img_torch=torch_image)

        if model_id == "CLIP":
            features = get_clip_patch_features(
                patches=patches,
                feat_resize_dim=512,
            )
        elif model_id == "BLIP":
            features = get_blip_patch_features(
                patches=patches,
                feat_resize_dim=768,
            )
        else:
            features = get_resnet_patch_features(
                patches=patches,
                feat_resize_dim=2048,
            )

        features = features.squeeze(0).cpu().numpy()
        feats = {"feat": features}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


def process_whole_image(
    image_dir: str,
    output_dir: str,
    model_id: str,
):
    # Get the images in the directory
    images = os.listdir(image_dir)
    dev = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device set to {dev}")
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        _, torch_image = load_image(os.path.join(image_dir, image))

        if model_id == "CLIP":
            features = get_clip_whole_img_features(img=torch_image)
        elif model_id == "BLIP":
            features = get_blip_whole_img_features(img=torch_image)
        else:
            features = get_resnet_whole_img_features(img=torch_image)

        features = features.squeeze(0).cpu().detach().numpy()

        feats = {"feat": features}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image_dir", type=str, required=True, help="Path to image dir")
    args.add_argument("--save_dir", type=str, required=True, help="Path to save dir")

    # Segmentation options
    args.add_argument(
        "--num_superpixels", type=int, default=25, help="Number of superpixels to use"
    )
    args.add_argument(
        "--algorithm",
        type=str,
        default="SLIC",
        choices=["SLIC", "watershed"],
        help="Superpixel algorithm to use",
    )
    args.add_argument(
        "--rag",
        action="store_true",
        help="Add RAG edge features to the superpixel features",
    )

    args.add_argument(
        "--whole_img", action="store_true", help="Generate whole image features"
    )
    args.add_argument(
        "--patches",
        action="store_true",
        help="Generate patch features instead of superpixel features",
    )

    # Model options
    args.add_argument(
        "--feature_extractor",
        type=str,
        default="BLIP",
        choices=["BLIP", "CLIP", "ResNet"],
        help="Which model to use for feature extraction?",
    )

    args = args.parse_args()
    get_logger()
    print("started")

    # Sanity Checks
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory {args.image_dir} not found.")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.whole_img and args.patches:
        raise ValueError(
            "Cannot generate both whole image and patch features at the same time."
        )

    if (args.rag and args.patches) or (args.rag and args.whole_img):
        raise ValueError(
            "Cannot generate RAG features with patches or whole image features."
        )

    if args.whole_img:
        process_whole_image(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            model_id=args.feature_extractor,
        )
    elif args.patches:
        process_patches(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            model_id=args.feature_extractor,
        )
    elif args.rag:
        process_rag(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            num_superpixels=args.num_superpixels,
            model_id=args.feature_extractor,
            superpixel_algo=args.algorithm,
        )
    else:
        process_superpixels(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            num_superpixels=args.num_superpixels,
            model_id=args.feature_extractor,
            superpixel_algo=args.algorithm,
        )
