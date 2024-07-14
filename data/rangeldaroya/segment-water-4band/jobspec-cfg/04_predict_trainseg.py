# Use pretrained model to segment

from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)
import argparse
from loguru import logger
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import mobilenetv3, ResNet50_Weights
import random
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import math
from models import Model4Band

import rasterio
from lang_sam import LangSAM    # See https://github.com/luca-medeiros/lang-segment-anything
from PIL import Image
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser(description='Lang SAM')
parser.add_argument('--i', default=0, type=int, help='Index to process [0-31]')
parser.add_argument('--ckpt_path', default="ckpts/deeplab_mobilenet/20240121-224805_e72.pth", type=str, help='Path to model checkpoint')
parser.add_argument('--with_transforms', default=0, type=int, help='Flag to standardize image (1 to standardize. 0 to not)')
parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold for predicting masks')
parser.add_argument('--crop_size', default=None, type=int, help='Num of splits in one dimension')
opt = parser.parse_args()
logger.debug(f"opt: {opt}")

all_fps = ['data/Connecticut_UTM18/Connecticut_20230706_01.tif',
 'data/Connecticut_UTM18/Connecticut_20230706_06.tif',
 'data/Connecticut_UTM18/Connecticut_20230706_13.tif',
 'data/PNW_UTM10/PNW_20230612_1_1.tif',
 'data/PNW_UTM10/PNW_20230612_1_3.tif',
 'data/PNW_UTM10/PNW_20230612_3_4.tif',
 'data/Willamette_UTM10/Willamette_20230606_01.tif',
 'data/Willamette_UTM10/Willamette_20230606_02.tif',
 'data/Willamette_UTM10/Willamette_20230606_03.tif',
 'data/Willamette_UTM10/Willamette_20230606_04.tif',
 'data/Willamette_UTM10/Willamette_20230606_06.tif',
 'data/Willamette_UTM10/Willamette_20230606_07.tif',
 'data/Willamette_UTM10/Willamette_20230606_08.tif',
 'data/Willamette_UTM10/Willamette_20230606_09.tif',
 'data/Willamette_UTM10/Willamette_20230612_01.tif',
 'data/Willamette_UTM10/Willamette_20230612_02.tif',
 'data/Willamette_UTM10/Willamette_20230612_03.tif',
 'data/Willamette_UTM10/Willamette_20230612_04.tif',
 'data/Willamette_UTM10/Willamette_20230612_05.tif',
 'data/Willamette_UTM10/Willamette_20230612_06.tif',
 'data/Willamette_UTM10/Willamette_20230612_07.tif',
 'data/Willamette_UTM10/Willamette_20230612_08.tif',
 'data/Willamette_UTM10/Willamette_20230612_09.tif',
 'data/Willamette_UTM10/Willamette_20230621_01.tif',
 'data/Willamette_UTM10/Willamette_20230621_02.tif',
 'data/Willamette_UTM10/Willamette_20230621_03.tif',
 'data/Willamette_UTM10/Willamette_20230621_04.tif',
 'data/Willamette_UTM10/Willamette_20230621_05.tif',
 'data/Willamette_UTM10/Willamette_20230621_06.tif',
 'data/Willamette_UTM10/Willamette_20230621_07.tif',
 'data/Willamette_UTM10/Willamette_20230621_08.tif',
 'data/Willamette_UTM10/Willamette_20230621_09.tif']


INPUT_DIR = "supervised_data"
input_label_mapping_train = {
    "PNW_20230612_3_4.tif": f"{INPUT_DIR}/PNW lakes/PNW_20230612_3_4_label.tif",    #i=5
    'Willamette_20230606_06.tif': f'{INPUT_DIR}/Willamette/Willamette_20230606_06_label.tif',   #i=10
    'Willamette_20230606_07.tif': f'{INPUT_DIR}/Willamette/Willamette_20230606_07_label.tif',   #i=11
}
input_label_mapping_val = {
    "PNW_20230612_1_1.tif": f"{INPUT_DIR}/PNW lakes/PNW_20230612_1_1_label.tif",    #i=3
}
input_label_mapping_test = {
    'Willamette_20230606_08.tif': f'{INPUT_DIR}/Willamette/Willamette_20230606_08_label.tif',   #i=12
}

EPS = 1e-7
def preprocess_input(input_chunk, trans):
    image = torch.from_numpy(input_chunk[:,:,:])
    image = image/255.   # normalize 0 to 1
    if trans is not None:
        image = trans(image)
    return image.type(torch.FloatTensor)

def compute_mask_metrics(pred, target, thresh=0.5):
    thresh_pred = np.where(pred > thresh, 1., 0.)
    rec = get_recall(thresh_pred, target)
    prec = get_precision(thresh_pred, target)
    f1 = get_f1(thresh_pred, target)

    return rec, prec, f1

def get_recall(y_pred, y_true):
    TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    TP_FN = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + EPS)
    return recall

def get_precision(y_pred, y_true):
    TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    TP_FP = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + EPS)
    return precision

def get_f1(y_pred, y_true):
    precision = get_recall(y_pred, y_true)
    recall = get_precision(y_pred, y_true)
    return 2 * ((precision * recall) / (precision + recall + EPS))

if __name__ == "__main__":
    opt = parser.parse_args()
    logger.debug(f"opt: {opt}")
    model_type = opt.ckpt_path.split("/")[-2]

    is_rgb = False
    model = Model4Band(model_type=model_type).cuda()
    if opt.crop_size is None:
        crop_size = model.crop_size
    else:
        crop_size = opt.crop_size
    if opt.with_transforms==1:
        trans = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406, 0.485), (0.229, 0.224, 0.225, 0.229)),
        ])
    else:
        trans = None
    logger.debug(f"Using model: {model_type} with crop_size: {crop_size}. trans: {trans}")
    ckpt = torch.load(opt.ckpt_path)
    tmp = model.load_state_dict(ckpt["state_dict"])
    logger.debug(f"Loaded model checkpoint: {tmp}")


    # fp = "Connecticut_20230706_01.tif"
    fp = all_fps[opt.i]


    # TODO: add output data and compute metrics
    out_data = None
    out_data_fp = None
    key = fp.split("/")[-1]
    if key in input_label_mapping_train.keys():
        metric_src = "train"
        out_data_fp = input_label_mapping_train[key]
    elif key in input_label_mapping_val.keys():
        metric_src = "val"
        out_data_fp = input_label_mapping_val[key]
    elif key in input_label_mapping_test.keys():
        metric_src = "test"
        out_data_fp = input_label_mapping_test[key]
    if out_data_fp is not None:
        output_dataset = rasterio.open(out_data_fp)
        out_data = output_dataset.read()
        out_data = (out_data > 0).astype(int).squeeze()


    out_name = fp.replace(".tif", f"_{model_type}_transform{opt.with_transforms}_thresh{opt.thresh}_crop{opt.crop_size}.png")
    out_name_tif = fp.replace(".tif", f"_{model_type}_transform{opt.with_transforms}_thresh{opt.thresh}_crop{opt.crop_size}.tif")
    # out_name_tif = fp.replace(".tif", f"_prediction.tif")
    logger.debug(f"Processing file: {fp}. out_name: {out_name}")
    
    input_dataset = rasterio.open(fp)
    input_data = input_dataset.read()   # (4, x, y) -- (C,H,W)
    _, s1,s2 = input_data.shape

    
    model.eval()
    pred_out_img = None
    ctr = 1
    for i in range(int(np.ceil(s1/crop_size))):
        vert_img = None
        for j in range(int(np.ceil(s2/crop_size))):
            ctr += 1
            input_chunk = input_data[:, i*crop_size : (i+1)*crop_size, j*crop_size : (j+1)*crop_size,]
            
            nonzero_mask = (np.sum(input_chunk, axis=0) > 0).astype(int)
            img = preprocess_input(input_chunk, trans)
            with torch.no_grad():
                pred_img = model(img.unsqueeze(0).cuda())["out"]
            pred_np = pred_img.detach().cpu().numpy().squeeze()
            pred_np = pred_np>opt.thresh
            pred_np = pred_np * nonzero_mask   # Multiply to nonzero mask (to make sure zero pixels are not mistaken as water)
            
            if vert_img is None:
                vert_img = np.zeros_like(pred_np)
                vert_img[:] = pred_np[:]
            else:
                # logger.debug(f"vert_img: {vert_img.shape}, pred_np: {pred_np.shape}")
                vert_img = np.concatenate((vert_img, pred_np), axis=1)
        if pred_out_img is None:
            pred_out_img = vert_img
        else:
            pred_out_img = np.concatenate((pred_out_img, vert_img), axis=0)
        
        # plt.imshow(pred_out_img, cmap="gray")
        # plt.savefig(f"pred_out_img_{ctr}.png")
        # plt.close()
    if out_data is not None:
        logger.debug(f"Computing metrics for pred_out_img: {pred_out_img.shape} vs out_data: {out_data.shape}")
        rec, prec, f1 = compute_mask_metrics(pred_out_img, out_data, thresh=opt.thresh)
        logger.debug(f"{metric_src} METRIC: f1={f1}, rec={rec}, prec={prec}")

    plt.imshow(pred_out_img, cmap="gray")
    plt.savefig(f"{out_name}.jpg")
    plt.close()

    out_3channel = np.stack((pred_out_img,pred_out_img,pred_out_img), axis=-1)
    cv2.imwrite(f"{out_name}", out_3channel*255)

    # TODO: write output as tif file with same format as input
    # Write to TIFF
    kwargs = input_dataset.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    with rasterio.open(out_name_tif, 'w', **kwargs) as dst:
        dst.write_band(1, pred_out_img.astype(rasterio.float32))