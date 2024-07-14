import os
from pathlib import Path
import argparse
import sys
import glob
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from models.network_swinir import SwinIR as net
import os.path
import time
import random
from data_loader import DataLoaderRegression
from utils import utils_calculate_psnr_ssim as util
import cv2
import PIL.Image
import tqdm
import glob
import math
from sklearn.metrics import r2_score
import pytorch_warmup as warmup 
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_args_parser():
    parser = argparse.ArgumentParser('Training the binary classifier', add_help=False)
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--train_hr_dir', type=str, help='hq location')
    parser.add_argument('--train_lr_dir', type=str, help='lr location')
    parser.add_argument('--train_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--train_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--test_hr_dir', type=str, help='hq location')
    parser.add_argument('--test_lr_dir', type=str, help='lr location')
    parser.add_argument('--test_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--test_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--output_dir', type=str, help="location for new model checkpoints")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=0.000001, type=float, help='Min learning rate')
    parser.add_argument('--seed', default=2023, type=int, help='training seed')
    parser.add_argument('--img_size', default=48, type=int, help='training seed')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--window_size', type=int, default=8, help='SwinIR model window size')
    parser.add_argument('--opt_type', type=str, default='Adam', help='Optimizer for SwinIR model (SGD/Adam)')
    parser.set_defaults(pin_mem=True)
    return parser

def prep_image(path):
    image = PIL.Image.open(path)
    image = np.array(image) / 255.
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis,...]
    return torch.from_numpy(image).float()

@torch.no_grad()
def reg_and_psnr(model, ttt_path, orig_path, lr_path, gt_path, device, switch_after=1, scale=4):
    model.eval()
    reg_psnrs = []
    reg_ssims = []
    ttt_psnrs = []
    ttt_ssims = []
    orig_psnrs = []
    orig_ssims = []
    for idx, gt_image_path in enumerate(tqdm.tqdm(glob.glob(os.path.join(gt_path, '*.png')))):
        # if idx == switch_after:
        #     device = torch.device("cuda:1")
        #     model.to(device)
        image_name = os.path.split(gt_image_path)[-1]
        gt_image = prep_image(gt_image_path)
        lr_image = prep_image(os.path.join(lr_path, image_name))
        _, _, h_old, w_old = lr_image.size()
        gt_image = gt_image[..., :h_old * 4, :w_old * 4]  # crop gt
        gt_image.to(device)
        lr_image.to(device) 
        ttt_image = prep_image(os.path.join(ttt_path, image_name)).to(device)
        orig_image = prep_image(os.path.join(orig_path, image_name)).to(device)
        model_output = model(torch.cat((orig_image, ttt_image), dim=1))
        gt_image = gt_image[:, :, :model_output.shape[2], :model_output.shape[3]]
        assert model_output.shape == gt_image.shape
        gt_image = (gt_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        model_output = (model_output[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        ttt_image = (ttt_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        orig_image = (orig_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        # Merged:
        psnr_y = util.calculate_psnr(model_output, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(model_output, gt_image, crop_border=scale, test_y_channel=True)
        reg_psnrs.append(psnr_y)
        reg_ssims.append(ssim_y)
        # TTT
        psnr_y = util.calculate_psnr(ttt_image, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(ttt_image, gt_image, crop_border=scale, test_y_channel=True)
        ttt_psnrs.append(psnr_y)
        ttt_ssims.append(ssim_y)
        # ORIGINAL
        psnr_y = util.calculate_psnr(orig_image, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(orig_image, gt_image, crop_border=scale, test_y_channel=True)
        orig_psnrs.append(psnr_y)
        orig_ssims.append(ssim_y)

    print('Mean orig psnr:', np.mean(orig_psnrs))
    print('Mean orig ssim:', np.mean(orig_ssims))
    print('Mean ttt psnr:', np.mean(ttt_psnrs))
    print('Mean ttt ssim:', np.mean(ttt_ssims))
    print('Mean regression psnr:', np.mean(reg_psnrs))
    print('Mean regression ssim:', np.mean(reg_ssims))

def train_worker(args):
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate model
    model = net(upscale=1, in_chans=6, out_chans=3, 
                img_size=args.img_size, window_size=args.window_size,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], 
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', 
                resi_connection='1conv', rescale_back=False).to(device)

    # Set weights of last layer to 0
    last_layer_name = 'conv_last'
    getattr(model, last_layer_name).weight.data.zero_()

    # last_layer_weights = getattr(model, last_layer_name).weight
    # print(f"Weights of the last layer '{last_layer_name}':\n{last_layer_weights}")

    # Wrap model with DataParallel
    model = torch.nn.DataParallel(model)

    # Instantiate loss function and optimizer
    if args.opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer_to(optimizer, device)
    criterion = torch.nn.MSELoss().to(device)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min')
    warmup_scheduler = warmup.UntunedLinearWarmup(lr_scheduler)
    

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_dir = args.output_dir
    if not os.path.exists(args.output_dir):
         os.mkdir(args.output_dir)
    print(f'Saving to {save_dir}')

    train_dataset = DataLoaderRegression(args.train_hr_dir, args.train_lr_dir, 
                                         args.train_pretrain_dir, args.train_ttt_dir,
                                         split='train', img_size=args.img_size)
    
    test_dataset = DataLoaderRegression(args.test_hr_dir, args.test_lr_dir, 
                                         args.test_pretrain_dir, args.test_ttt_dir,
                                         split='test', img_size=args.img_size)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Training loop
    print("Training Batches:", len(train_data_loader))
    best_prec1 = 0
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}...")
        prec1 = train(args, train_data_loader, model, optimizer, criterion, lr_scheduler, warmup_scheduler, epoch, device)
        print(f'Done training. Epoch [{epoch:05}]: Loss={prec1}')
        test_prec1 = test(args, test_data_loader, model, criterion, epoch, device)
        best_prec1 = max(prec1, best_prec1)
        state_dict = {
            'epoch': epoch + 1,
            'arch': 'swinir',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'test_prec1': test_prec1
        }
        print(f'Done testing. Epoch [{epoch:05}]: Loss={test_prec1}')
        if (epoch + 1) % 2 == 0:
            torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_s{args.img_size}_ep{epoch+1}_win{args.window_size}_{prec1:.2f}_{test_prec1:.2f}.pth')

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def optimizer_to(optim, device):
    """
    Move pretrained optimizer to device used by PyTorch
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)



def train(args, data_loader, model, optimizer, criterion, lr_scheduler, warmup_scheduler, epoch, device):
    loss_vals, mae_vals, r2_scores = [], [], []
    model.train()

    for idx, (inputs, targets) in enumerate(tqdm.tqdm(data_loader)):
        # adjust_learning_rate(optimizer, idx / len(data_loader) + epoch, args)
        image_orig, image_ttt = inputs
        image_orig = image_orig.to(device)
        image_ttt = image_ttt.to(device)
        targets = targets.to(device)

        inputs = torch.cat((image_orig, image_ttt), dim=1)

        outputs = model(inputs) + image_orig

        loss = criterion(outputs, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with warmup_scheduler.dampening():
            lr_scheduler.step()

        with torch.no_grad():
            step_loss = loss.item()
            step_mae = mean_absolute_error(targets, outputs)
            step_r2 = r2_score(targets.reshape(-1).cpu().numpy(), outputs.reshape(-1).cpu().numpy())

        loss_vals.append(step_loss)
        mae_vals.append(step_mae)
        r2_scores.append(step_r2)

    avg_loss = np.mean(loss_vals)
    avg_mae = np.mean(mae_vals)
    avg_r2 = np.mean(r2_scores)

    print(f"Epoch [{epoch + 1}], Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}")

    return avg_loss

@torch.no_grad()
def test(args, data_loader, model, criterion, epoch, device):
    model.eval()
    loss_vals, mae_vals, r2_scores = [], [], []
    for idx, (inputs, targets) in enumerate(tqdm.tqdm(data_loader)):
        image_orig, image_ttt = inputs
        image_orig = image_orig.to(device, non_blocking=True)
        image_ttt = image_ttt.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(torch.cat((image_orig, image_ttt), dim=1))
        loss = criterion(outputs, targets)
        with torch.no_grad():
            test_loss = loss.item()
            test_mae = mean_absolute_error(targets, outputs)
            test_r2 = r2_score(targets.reshape(-1).cpu().numpy(), outputs.reshape(-1).cpu().numpy())

        loss_vals.append(test_loss)
        mae_vals.append(test_mae)
        r2_scores.append(test_r2)

    avg_loss = np.mean(loss_vals)
    avg_mae = np.mean(mae_vals)
    avg_r2 = np.mean(r2_scores)
    
    print(f"Epoch [{epoch + 1}], Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}")
    
    return avg_loss

def main():
    # Argument parser
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_worker(args)

if __name__ == '__main__':
    main()