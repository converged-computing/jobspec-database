import argparse

import timeit
import numpy as np
import os
from tqdm import tqdm
from path import Path
import torch
from imageio.v2 import imread, imsave
import matplotlib.pyplot as plt
import cv2

from models.SFM.DispNetS import DispNetS
from models.DepthAnything.DepthAnything import DepthAnythingSFM
from train_only_depth.udepth_model.udepth import UDepth_SFM

from utils.eval import median_of_non_zero_values

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluate Depth")
parser.add_argument("--gt_depth", default='data/scaled_and_cropped_depth/2015', help="gt depth dir", type=str)
parser.add_argument("--img_dir", default='data/Eiffel-Tower_ready_Downscaled_colmap/2015',help="image directory for reading image", type=str)
parser.add_argument("--scale", action='store_true', help="Will correct the scale using Median")
parser.add_argument("--depth_model", default="dpts", help="model to select. options: dpts, udepth, dispnet", type=str)
parser.add_argument("--saved_model", default=None, type=str)
######################################################


def predict_fn(model, input_data):

    return model(input_data)

def compute_depth_errors(gt, pred, use_mask=True):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    if use_mask:
        mask = np.array(gt, dtype=bool).astype(int)
        pred = (pred*mask) #element wise multiply the mask

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    #flatten and get the non zeros to calculate rest to avoid zero division
    flat_gt = gt.flatten()
    gt = flat_gt[flat_gt != 0]
    
    flat_pred = pred.flatten()
    pred = flat_pred[flat_pred != 0]

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def load_tensor_image(filename):

    img = imread(filename).astype(np.float32)
    # print(f'img load shape {img.shape}')
    img = np.transpose(img, (2, 0, 1))
    
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.5)/0.5)
    
    return tensor_img

def load_gt_depth(filename):
    gt_depth  = cv2.imread(filename)
    gt_depth = cv2.cvtColor(gt_depth, cv2.COLOR_BGR2GRAY)

    return gt_depth

def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'found device {device}')

    #check if valid directories
    image_dir = Path(args.img_dir)
    depth_dir = Path(args.gt_depth)
    assert os.path.exists(image_dir) and os.path.exists(depth_dir), 'image or depth directory invalid'
    image_list = sorted(image_dir.glob('*.png'))
    depth_list = sorted(depth_dir.glob('*.png'))

    assert len(image_list)==len(depth_list)

    #check correct model name given:
    assert args.depth_model in ['dpts', 'dptl', 'dptb', 'dispnet','udepth'], "correct model name not given"

    #initialize the models
    if args.depth_model == 'dispnet':
        model = DispNetS()
        total_param = sum(param.numel() for param in model.parameters())
        print(f"total disp net parameter {total_param/1e6}M")
    
    elif args.depth_model == 'dpts':
        model = DepthAnythingSFM(encoder='vits')
    elif args.depth_model == 'dptl':
        model = DepthAnythingSFM(encoder='vitl')
    elif args.depth_model == 'dptb':
        model = DepthAnythingSFM(encoder='vitb')
    elif args.depth_model=='udepth':
        model = UDepth_SFM(True)

    model.to(device)
    #load saved model
    if args.saved_model is not None:
        saved_model_path = Path(args.saved_model)
        print(f"loading {saved_model_path}")
        weights = torch.load(saved_model_path, map_location=device)
        
        # if 'epoch' in weights:
        #     print(f'loading model from epoch{weights['epoch']}')

        model.load_state_dict(weights['state_dict'], strict=False)
        model.eval()

    else:
        model.eval()
        print('No saved model given. Running with initialized parameters')
        
    abs_rel_tot = 0
    sq_rel_tot = 0
    rmse_tot = 0
    rmse_log_tot = 0
    a1_tot = 0 
    a2_tot = 0
    a3_tot = 0
    for i,(image, depth) in tqdm(enumerate(zip(image_list, depth_list))):
        
        img = load_tensor_image(image).to(device)
        
        gt_depth = load_tensor_image(depth).to(device)
        gt_depth = load_gt_depth(depth)
        if args.depth_model == 'dispnet':
            pred_depth = model(img).cpu().squeeze().detach().numpy()
        else:
            pred_depth = model(img)[0].cpu().squeeze().detach().numpy()

        if args.depth_model == 'dispnet':
            pred_depth = 1/pred_depth
        else:
            pred_depth = pred_depth


        if i==0:
            print('calculating model avg inference time')
            number_of_iterations = 100  # Number of times to run the prediction

            # Time the prediction function
            time_taken = timeit.timeit(lambda: predict_fn(model, img), number=number_of_iterations)

            # Calculate average prediction time per iteration
            average_prediction_time = time_taken / number_of_iterations

            print(f"avg prediction time: {average_prediction_time}")

        if args.scale:
            gt_median = median_of_non_zero_values(gt_depth)
            predicted_median = median_of_non_zero_values(pred_depth)
            scale = gt_median/predicted_median
            # print(scale)
            
            pred_depth = pred_depth*scale


        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(gt_depth, pred_depth)
        
        abs_rel_tot += abs_rel
        sq_rel_tot += sq_rel
        rmse_tot += rmse
        rmse_log_tot += rmse_log
        a1_tot += a1
        a2_tot += a2
        a3_tot += a3
        
    return abs_rel_tot/len(image_list), sq_rel_tot/len(image_list), rmse_tot/len(image_list), rmse_log_tot/len(image_list), a1_tot/len(image_list), a2_tot/len(image_list), a3_tot/len(image_list)

if __name__=='__main__':
    abs_rel_tot, sq_rel_tot, rmse_tot, rmse_log_tot, a1_tot, a2_tot, a3_tot = main()
    
    print(f"result: abs_rel={abs_rel_tot}, sq_rel={sq_rel_tot}, rmse={rmse_tot}, rmse_log ={rmse_log_tot}, a1 = {a1_tot}, a2 = {a2_tot}, a3 = {a3_tot}")
