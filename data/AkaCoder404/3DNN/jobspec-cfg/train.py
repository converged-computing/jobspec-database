"""
Title: Training Neural Networks for 3D Object classification, semantic segmentation and partial segmentation


# TODO
- [ ] Add time logging per epoch or per batch
"""

import os
import sys
import argparse
import torch
import time

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import datetime
from tqdm import tqdm
import numpy as np

from Models.pointnet import PointNetClassification, PointNetClassificationLoss, \
    PointNetSegmentation, PointNetSegmentationLoss
    
from Models.pointcnn_v1 import PointCNNClassification_V1
    
from custom_datasets import ModelNetDataLoader, ModelNetPlyHDF52048DataLoader, TUBerlinDataLoader
import utils

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode, gpu default')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--pretrained', type=str, default='', help="pretrained")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--momentum', type=float, default=0.9)

    # Dataset specific
    parser.add_argument('--dataset', type=str, default='ModelNet40', help='dataset for training')
    parser.add_argument('--num_category', default=40, type=int,  help='Number of categories')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    
    return parser.parse_args()

def create_training_directory(root_dir):
    # Ensure the root directory exists
    os.makedirs(root_dir, exist_ok=True)

    # Count the number of existing training directories
    existing_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('train')]
    num_existing_dirs = len(existing_dirs)

    # Create a new directory with the next number
    new_dir_number = num_existing_dirs + 1
    new_dir = os.path.join(root_dir, f'train{new_dir_number}')
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

#################################################################################################### 
def training_step(model: nn.Module, train_data_loader: DataLoader, optimizer: nn.Module, criterion: nn.Module, device: torch.device):
    """"""
    model.train()
    mean_correct = []
    
    total_acc, total_loss, total_batch_time  = 0.0, 0.0, 0.0
    for batch_idx, (points, target) in tqdm(enumerate(train_data_loader, 0), total=len(train_data_loader), smoothing=0.9):
        batch_start_time = time.time()
    
        # TODO Perform data augmentation
        points = points.data.numpy()
        points = utils.random_point_dropout(points)
        points[:, :, 0:3] = utils.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = utils.shift_point_cloud(points[:, :, 0:3])
        
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if device == torch.device("cuda"):
            points, target = points.to(device), target.to(device)
            
        optimizer.zero_grad()
        pred, trans_feat = model(points)
        loss = criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        # mean_correct.append(correct.item() / float(points.size()[0]))
        
        total_loss += loss.item()
        total_acc += correct.item() / float(points.size()[0])
        
        loss.backward()
        optimizer.step()
        
        batch_end_time = time.time()
        total_batch_time += batch_end_time - batch_start_time
        
    return total_acc / len(train_data_loader), total_loss / len(train_data_loader), total_batch_time / len(train_data_loader)
     
def testing_step(model: nn.Module, test_data_loader: DataLoader, criterion: nn.Module, device: torch.device, args):
    """"""
    model.eval()
    mean_correct = []
    
    # TODO Hardcode class counts
    class_acc = np.zeros((args.num_category, 3))
    
    total_acc, total_loss, total_batch_time  = 0.0, 0.0, 0.0
    with torch.inference_mode():
       for batch_idx, (points, target) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            batch_start_time = time.time()
            if device == torch.device("cuda"):
                points, target = points.to(device), target.to(device)
                
            points = points.transpose(2, 1)
            pred, _ = model(points)
            pred_choice = pred.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            batch_end_time = time.time()
            total_batch_time += batch_end_time - batch_start_time
            
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, total_batch_time / len(test_data_loader)
####################################################################################################


####################################################################################################
def pointcnn_v1_cls_training_step(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for batch_idx, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):        
        if device.type == "cuda":
            data, label = data.to(device), label.to(device)
        
        # TODO Augment batched point clouds by rotation and jittering
        # P_sampled -> (B, N, 3) -> (64, 1024, 3)
     
        optimizer.zero_grad()
        output = model((data, data))        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += torch.sum(torch.argmax(output, dim=1) == label).item() / len(label)
        # print(f"batch: {batch_idx}, loss: {loss.item()}")
    
    return total_acc / len(dataloader), total_loss / len(dataloader)

def pointcnn_v1_cls_testing_step(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for batch_idx, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
        if device.type == "cuda":
            data, label = data.to(device), label.to(device)
    
        output = model((data, data))
        predicted = torch.argmax(output, dim=1)
        total_acc += (predicted == label).sum().item() / label.size(0)

    # TODO Return something other than total loss or calculate total loss
    return total_acc / len(dataloader), total_loss 
####################################################################################################


def main(args):
    """"""   
    
    # Select model
    if args.model == "pointnet_cls":
        model = PointNetClassification(k=args.num_category, normal_channel=args.use_normals)
        criterion = PointNetClassificationLoss()
    elif args.model == "pointnet_seg":
        model = PointNetSegmentation()
        criterion = PointNetSegmentationLoss()
    elif args.model == "pointnet_part_seg":
        pass
    elif args.model == "pointnet2_cls":
        pass
    elif args.model == "pointcnn_v1_cls":
        model = PointCNNClassification_V1()
        criterion = nn.CrossEntropyLoss()
    elif args.model == "pointcnn_v2_cls":
        pass
    
    # Select dataset
    if args.dataset == "ModelNet40":
        data_path = "data/modelnet40_normal_resampled/"
        train_dataset   = ModelNetDataLoader(root=data_path, 
                                             num_point=args.num_point,
                                             use_uniform_sample=args.use_uniform_sample,
                                             use_normals=args.use_normals,
                                             num_category=args.num_category,
                                             split='train', process_data=args.process_data)
        test_dataset    = ModelNetDataLoader(root=data_path, 
                                             num_point=args.num_point,
                                             use_uniform_sample=args.use_uniform_sample,
                                             use_normals=args.use_normals,
                                             num_category=args.num_category,
                                             split='test', process_data=args.process_data)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
        test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
        
    elif args.dataset == "ModelNet40_hdf5":
        data_path = "data/modelnet40_ply_hdf5_2048/"
        train_dataset = ModelNetPlyHDF52048DataLoader(root="data/modelnet40_ply_hdf5_2048", num_point=1024, split="train")
        test_dataset = ModelNetPlyHDF52048DataLoader(root="data/modelnet40_ply_hdf5_2048", num_point=1024, split="test")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "ModelNet10":
        pass
    elif args.dataset == "ShapeNet":
        pass
    elif args.dataset == "TUBerlin":
        data_path = "data/tu_berlin/"
        train_dataset = TUBerlinDataLoader(root=data_path, num_point=args.num_point, split="train")
        test_dataset = TUBerlinDataLoader(root=data_path, num_point=args.num_point, split="test")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    print(f"Dataset Lengths: train({len(train_dataset)}), test({len(test_dataset)})")
        
    # Select Device
    device = torch.device("cpu")
    if not args.use_cpu:                        # If not use cpu, default to gpu
        if not torch.cuda.is_available():
            print("Trying to use GPU, but no GPU found, exiting...")
            sys.exit(0)
            
        device = torch.device("cuda")
        model       = model.to(device)
        criterion   = criterion.to(device)
        
    # Logging
    new_train_dir = create_training_directory("exps")
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    writer = SummaryWriter(new_train_dir + '/logs/')
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(new_train_dir + '/train.log'))
    logger.info(f"Training {args.model} on {args.dataset} on {device}")
    logger.info(f"Epochs: {args.epoch}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}, Optimizer: {args.optimizer}")
    logger.info(f"Dataset Lengths: train({len(train_dataset)}), test({len(test_dataset)})")
    logger.info(f"Start time {timestr}")
    logger.info("-" * 100)
    

    # Pretrained
    if args.pretrained != '':
        print("Using pretrained model")
        checkpoint = torch.load()
    else:
        print("Training from scratch")
        
    # Choose optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Training
    print("Start training...")
    start_epoch = 0
    
    for epoch in range(start_epoch, args.epoch):
        print(f"Epoch {epoch} / {args.epoch}")
        test_acc, test_loss, train_batch_time, test_batch_time = 0.0, 0.0, 0.0, 0.0

        ###############################
        # Perform training and testing
        if args.model == "pointnet_cls":
            train_acc, train_loss, train_batch_time = training_step(model, train_dataloader, optimizer, criterion, device)
            test_acc, test_loss, test_batch_time = testing_step(model, test_dataloader, criterion, device, args)
        elif args.model == "pointcnn_v1_cls":
            train_acc, train_loss = pointcnn_v1_cls_training_step(model, train_dataloader, optimizer, criterion, device)
            test_acc, test_loss = pointcnn_v1_cls_testing_step(model, test_dataloader, criterion, device)
       
        scheduler.step()
        ###############################
        
    
        # Log
        print()
        # print(f"Train_acc {train_acc}, Train_loss {train_loss}, test_acc {test_acc}, test_loss {test_loss}")
        writer.add_scalar('training loss', train_acc, epoch)
        logger.info(f"Epoch {epoch} / {args.epoch}")
        logger.info(f"Train_acc {train_acc}, Train_loss {train_loss}, test_acc {test_acc}, test_loss {test_loss}")
        logger.info(f"Average train/test batch time: {train_batch_time} / {test_batch_time}")
        
        # Get best instance and safe
        if (test_acc >= best_acc):
            best_acc = test_acc
            best_epoch = epoch + 1
            
            # TODO Save model at specific run
            print("Saving model...")
            save_path = new_train_dir + "/best_model.pth"
            print(f"Saving at {save_path}")
            
            state = {
                "epoch": epoch,
                "accuracy": test_acc,
                "class_accuracy": test_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            } 
            torch.save(state, save_path)
            
        # Best class 
        
        
    print("Training finished")
    logger.info("-" * 100)
    logger.info("Training finished")
    # logger.info(f"End time {}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    