import pandas as pd
import numpy as np
import torch
import os
import time
from imageio import imread, imwrite
from scipy import linalg
import argparse
import sys
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, "../SemanticStyleGAN")
from models import make_model
from visualize.utils import generate
from torchmetrics.functional import kl_divergence


# import matplotlib.pyplot as plt
##TODO: Remove and import it from another class
# TODO: Refactor code
color_map = {
    0: [0, 0, 0],  # Void
    1: [128, 64, 128],  # Road
    2: [244, 35, 232],  # Side Walk
    3: [70, 70, 70],  # Building
    4: [102, 102, 156],  # Wall
    5: [190, 153, 153],  # Fence
    6: [153, 153, 153],  # pole
    7: [250, 170, 30],  # traffic light
    8: [220, 220, 0],  # Traffic sign
    9: [107, 142, 35],  # Vegitation
    10: [70, 130, 180],  # sky
    11: [220, 20, 60],  # human
    12: [255, 0, 0],  # rider
    13: [0, 0, 142],  # car
    14: [0, 60, 100],  # other vehicles
    15: [0, 0, 230],  # bike and motorcycle
    16: [116, 95, 159],
}


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # msg = ('fid calculation produces singular product; '
        #        'adding %s to diagonal of cov estimates') % eps
        # logger.info(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fsd(real_data, generated_data):
    """Calculates the FSD of two paths"""
    m1 = np.mean(real_data, axis=0)
    s1 = np.cov(real_data, rowvar=False)

    m2 = np.mean(generated_data, axis=0)
    s2 = np.cov(generated_data, rowvar=False)
    fsd_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fsd_value


def calculate_kl(real_data, generated_data):
    m1 = np.mean(real_data, axis=0)
    m2 = np.mean(generated_data, axis=0)

    m1 = torch.tensor(m1).unsqueeze(0)
    m2 = torch.tensor(m2).unsqueeze(0)
    return kl_divergence(m1, m2)


def initalize_model(ckpt, device):
    ckpt = torch.load(ckpt)
    model = make_model(ckpt["args"])
    model.to(device)
    model.eval()
    model.load_state_dict(ckpt["g_ema"])
    return model


# Stupid way of converting from rgb segmentation maps to the original labels
# converting from a label map back to it's label by summing the 3 channels into 1 channel, then
# checking mapping the knows sums to the correct label(index)
def from_rgb_to_label(image, color_map):
    color_map_sum = {}
    new_image = np.zeros((image.shape[0], image.shape[1]))
    for c in color_map:
        color_map_sum[c] = sum(color_map[c])
    image_tensor = torch.tensor(image).to(float)
    image_summed = torch.sum(image_tensor, dim=2)
    for index in color_map_sum:
        mask = image_summed == color_map_sum[index]
        new_image[mask] = index
    return new_image


def calculate_mean_for_one_hot(image):
    sem_seg_tensor = torch.tensor(image)
    sem_seg_unsqueezed = sem_seg_tensor.reshape(-1)
    res = torch.nn.functional.one_hot(sem_seg_unsqueezed.to(torch.int64), 16)
    final_res = res.reshape(
        sem_seg_tensor.shape[0], sem_seg_tensor.shape[1], -1
    ).float()
    mean_val = torch.mean(final_res, dim=(0, 1))
    mean_val = mean_val.unsqueeze(0)
    return mean_val


def calculate_real_cond(dataset_path, save_real_ds, sample_num=None):
    logger.info(f"Calculating values for the real data.")
    logger.info(f"Save real dataset : {save_real_ds}")
    dataset_mean_values = []
    accum = 0
    files_count = (
        sum([len(files) for r, d, files in os.walk(dataset_path)])
        if sample_num is None
        else sample_num
    )
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            accum += 1
            if accum % 10 == 0:
                logger.info(f"Done with {accum} / {files_count} of the real data")
            extra_data = "leftImg8bit" in file
            # In normal dataset, label images contain labelIds prefix.
            if not extra_data and "labelIds" not in file:
                continue
            # In extra dataset, files does not contain _prob prefix.
            if extra_data and "_prob" in file:
                continue
            if len(dataset_mean_values) >= sample_num:
                break
            filepath = subdir + os.sep + file
            image = imread(filepath)
            mean_val = calculate_mean_for_one_hot(image)
            dataset_mean_values.append(mean_val.cpu().numpy())
    real_cond = np.concatenate(dataset_mean_values)
    if save_real_ds:
        np.save("./real_dataset_cond_5k.npy", real_cond)
    return real_cond


def calculate_generated_cond(ckpt, sample, truncation, truncation_mean, batch, device):
    logger.info("Calculating generated cond")
    logger.info(f"Loading model from checkpoint")
    model = initalize_model(ckpt, "cuda")
    logger.info(f"Model initalized successfuly")
    logger.info(f"Style dimension is {model.style_dim}")
    mean_latent = model.style(
        torch.randn(truncation_mean, model.style_dim, device=device)
    ).mean(0)
    start_time = time.time()
    dataset_mean_values = []
    with torch.no_grad():
        n_batch = sample // batch
        resid = sample - (n_batch * batch)
        batch_sizes = [batch] * n_batch + [resid]
        for batch_iter in tqdm(batch_sizes):
            if batch_iter < batch:
                logger.info(f"Skipping batch iteration of size {batch_iter}")
                continue

            styles = model.style(
                torch.randn(batch_iter, model.style_dim, device=device)
            )
            logger.info(f"Styles shape is {styles.shape}")
            styles = truncation * styles + (1 - truncation) * mean_latent.unsqueeze(0)
            images, segs = generate(
                model, styles, mean_latent=mean_latent, batch_size=batch
            )

            for i in range(len(images)):
                converted_seg = from_rgb_to_label(segs[i], color_map)
                mean_val = calculate_mean_for_one_hot(converted_seg)
                dataset_mean_values.append(mean_val.cpu().numpy())
        logger.info(f"Time taken to generate images : {time.time()-start_time}")
        generated_cond = np.concatenate(dataset_mean_values)
    logger.info(f"Average speed: {(time.time() - start_time)/(sample)}s")
    return generated_cond


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, help="path to the model checkpoint")
    parser.add_argument("--dataset", type=str, default=8, help="path for dataset")
    parser.add_argument("--batch", type=int, default=8, help="batch size for inference")
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="number of samples to be generated to calculate FSD",
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation ratio"
    )
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=10000,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="running device for inference"
    )
    parser.add_argument(
        "--real_dataset_values",
        type=str,
        default=None,
        help="saved value for real dataset",
    )
    parser.add_argument(
        "--save_real_dataset",
        type=bool,
        default=False,
        help="saved value for real dataset",
    )
    args = parser.parse_args()
    logger.add("./log_files/logguru_v2/logging_{time}_fsd.log")
    start_time = time.time()
    logger.info("Calculating!!!")
    if args.real_dataset_values:
        logger.info(
            f"loading file for real dataset values from :{args.real_dataset_values}"
        )
        real_cond = np.load(args.real_dataset_values)
    else:
        real_cond = calculate_real_cond(
            args.dataset, args.save_real_dataset, sample_num=args.sample
        )
    logger.info(
        f"time taken to calculate statistics of real data: {time.time()-start_time}"
    )

    generated_cond = calculate_generated_cond(
        args.ckpt,
        args.sample,
        args.truncation,
        args.truncation_mean,
        args.batch,
        args.device,
    )
    np.save("./generated_cond.npy", generated_cond)
    logger.info(
        f"time taken to calculate statistics of fake data: {time.time()-start_time}"
    )
    fsd = calculate_fsd(real_cond, generated_cond)
    kl = calculate_kl(real_cond, generated_cond)
    fsd_count = calculate_fsd(real_cond * 100, generated_cond * 100)
    logger.info(f"KL divergence between 2 conditions is : {kl}")
    logger.info(
        f"FSD value by semantic pallete method is {fsd} and fsd_count {fsd_count} , Time Taken in total : {time.time()-start_time}"
    )
