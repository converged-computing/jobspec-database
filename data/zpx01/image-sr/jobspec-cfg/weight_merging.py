import torch
import torch.nn
import numpy as np
import PIL.Image
from models.network_swinir import SwinIR as net
from utils import utils_calculate_psnr_ssim as util
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from collections import OrderedDict
import csv
import argparse

class SwinIR_Merger:
    def __init__(self, pretrained_model, ttt_train_checkpoints, ttt_test_checkpoints, train_images, test_images, gt_train_images, gt_test_images, alpha, device):
        self.pretrained_model = pretrained_model
        self.ttt_train_checkpoints = ttt_train_checkpoints
        self.ttt_test_checkpoints = ttt_test_checkpoints
        self.train_images = train_images
        self.test_images = test_images
        self.gt_train_images = gt_train_images
        self.gt_test_images = gt_test_images
        self.alpha = alpha
        self.device = device
    
    def prep_image(self, path):
        image = PIL.Image.open(path)
        image = np.array(image) / 255.
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis,...]
        return torch.from_numpy(image).float()

    def prep_gt(self, lr, gt):
        lr_image = self.prep_image(lr)
        gt_image = self.prep_image(gt)
        _, _, h_old, w_old = lr_image.size()
        gt_image = gt_image[..., :h_old * 4, :w_old * 4]
        return gt_image

    def merge_weights(self, model1, model2):
        merged_model = OrderedDict()
        for key in model1:
            if key.startswith('module.'):
                key_without_module = key[len('module.'):]
                if key_without_module in model2:
                    merged_model[key] = (1 - self.alpha) * model1[key] + self.alpha * model2[key_without_module]
                else:
                    merged_model[key] = model1[key]
            else:
                if key in model2:
                    merged_model[key] = (1 - self.alpha) * model1[key] + self.alpha * model2[key]
                else:
                    merged_model[key] = model1[key]
        return merged_model


    def infer(self, merged_model, image):
        input_image = self.prep_image(image).to(self.device)
        with torch.no_grad():
            output = merged_model(input_image).to(self.device)
        return output

    def merge_and_infer(self):
        csv_filename = "model_merge.csv"
        print("Merging Training Checkpoints...")
        ttt_model = net(upscale=4, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='pixelshuffle', resi_connection='1conv')
        for idx, (ttt_checkpoint, train_image) in enumerate(zip(self.ttt_train_checkpoints, self.train_images)):
            print(f'Starting Training Image {idx+1} @ alpha={self.alpha}')
            ttt_model_weights = torch.load(ttt_checkpoint, map_location=self.device)
            ttt_model.load_state_dict(ttt_model_weights)
            ttt_model.to(self.device)
            merged_weights = self.merge_weights(self.pretrained_model.state_dict(), ttt_model.state_dict())

            merged_model = self.pretrained_model
            merged_model.load_state_dict(merged_weights)
            merged_model.to(self.device)
            merged_model.eval()

        
            output_image = self.infer(merged_model, train_image)

            self.gt_train_images[idx] = self.prep_gt(train_image, self.gt_train_images[idx])
            self.gt_train_images[idx] = (self.gt_train_images[idx][0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            output_image = (output_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            psnr_y = util.calculate_psnr(output_image, self.gt_train_images[idx], crop_border=4, test_y_channel=True)
            ssim_y = util.calculate_ssim(output_image, self.gt_train_images[idx], crop_border=4, test_y_channel=True)
            with open(csv_filename, mode='a') as csv_file:
                csv_writer = csv.writer(csv_file)
                if csv_file.tell() == 0:
                    csv_writer.writerow(['Image Path', 'Alpha', 'PSNR', 'SSIM', 'Type'])
                csv_writer.writerow([train_image, self.alpha, psnr_y, ssim_y, 'train'])

        # Validation
        print("Merging Testing Checkpoints...")
        for idx, (ttt_checkpoint, test_image) in enumerate(zip(self.ttt_test_checkpoints, self.test_images)):
            print(f'Starting Testing Image {idx+1} @ alpha={self.alpha}')
            ttt_model_weights = torch.load(ttt_checkpoint, map_location=self.device)
            ttt_model.load_state_dict(ttt_model_weights)
            ttt_model.to(self.device)
            merged_weights = self.merge_weights(self.pretrained_model.state_dict(), ttt_model.state_dict())

            merged_model = self.pretrained_model
            merged_model.load_state_dict(merged_weights)
            merged_model.to(self.device)
            merged_model.eval()

            test_image.to(self.device)

            output_image = self.infer(merged_model, test_image)

            self.gt_test_images[idx] = self.prep_gt(test_image, self.gt_test_images[idx])
            psnr_y = util.calculate_psnr(output_image, self.gt_test_images[idx], crop_border=4, test_y_channel=True)
            ssim_y = util.calculate_ssim(output_image, self.gt_test_images[idx], crop_border=4, test_y_channel=True)
            with open(csv_filename, mode='a') as csv_file:
                csv_writer = csv.writer(csv_file)
                if csv_file.tell() == 0:
                    csv_writer.writerow(['Image Path', 'Alpha', 'PSNR', 'SSIM', 'Type'])
                csv_writer.writerow([test_image, self.alpha, psnr_y, ssim_y, 'test'])


def get_averages(data_list):
    data_dict = {}
    for alpha, value in data_list:
        if alpha not in data_dict:
            data_dict[alpha] = []
        data_dict[alpha].append(value)

    averages = [(alpha, np.mean(values)) for alpha, values in data_dict.items()]
    averages.sort(key=lambda x: x[0])
    return averages

def write_to_file(filename, train_psnr, train_ssim, test_psnr, test_ssim):
    with open(filename, 'w') as f:
        f.write("Train PSNR:\n")
        for alpha, psnr in train_psnr:
            f.write(f"Alpha: {alpha}, PSNR: {psnr}\n")

        f.write("\nTrain SSIM:\n")
        for alpha, ssim in train_ssim:
            f.write(f"Alpha: {alpha}, SSIM: {ssim}\n")

        f.write("\nTest PSNR:\n")
        for alpha, psnr in test_psnr:
            f.write(f"Alpha: {alpha}, PSNR: {psnr}\n")

        f.write("\nTest SSIM:\n")
        for alpha, ssim in test_ssim:
            f.write(f"Alpha: {alpha}, SSIM: {ssim}\n")


def main_worker(local_rank, world_size, args):
    # DDP set up
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    # Set relevant directories
    TRAIN_HR_DIR='/home/zeeshan/image-sr/trainsets/trainH/DIV2K_train_HR_sub/'
    TRAIN_LR_DIR='/home/zeeshan/image-sr/trainsets/trainL/DIV2K_train_LR_bicubic/X4_sub/'
    TRAIN_TTT_DIR='/checkpoints/zeeshan/test_time_training/ttt_div2k_trainset/models/'
    TEST_HR_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/original/'
    TEST_LR_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/LRbicx4/'
    TEST_TTT_DIR='/checkpoints/zeeshan/test_time_training/set14_ttt/sgd_models/'
    PRETRAINED_MODEL_PATH = '/home/zeeshan/image-sr/superresolution/swinir_sr_classical_patch48_x4/models/classicalSR_SwinIR_x4.pth'

    # Get all file paths
    ttt_train_checkpoints = sorted(glob.glob(os.path.join(TRAIN_TTT_DIR, '*.pth')))[local_rank::world_size]
    ttt_test_checkpoints = sorted(glob.glob(os.path.join(TEST_TTT_DIR, '*.pth')))[local_rank::world_size]
    train_images = sorted(glob.glob(os.path.join(TRAIN_LR_DIR, '*.png')))[local_rank::world_size]
    test_images = sorted(glob.glob(os.path.join(TEST_LR_DIR, '*.png')))[local_rank::world_size]
    gt_train_images = sorted(glob.glob(os.path.join(TRAIN_HR_DIR, '*.png')))[local_rank::world_size]
    gt_test_images = sorted(glob.glob(os.path.join(TEST_HR_DIR, '*.png')))[local_rank::world_size]
    print("Loaded file paths!")

    # Instantiate pretrained model 
    device = torch.device('cuda', local_rank)
    model = net(upscale=4, in_chans=3, img_size=48, window_size=8,
                           img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                           num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                           upsampler='pixelshuffle', resi_connection='1conv')
    pretrained_model = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    param_key_g = 'params'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    print("Loaded pretrained SwinIR model!")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Instantiate merger instance
    alpha = args.alpha
    merger = SwinIR_Merger(model, ttt_train_checkpoints, ttt_test_checkpoints, train_images, test_images, gt_train_images, gt_test_images, alpha, device)

    # Run merger
    print('Starting Merge...')
    merger.merge_and_infer()
    print('\nCompleted merge!')

def get_args_parser():
    parser = argparse.ArgumentParser('Model weighting', add_help=False)
    parser.add_argument('--alpha', default=0.1, type=float)
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()