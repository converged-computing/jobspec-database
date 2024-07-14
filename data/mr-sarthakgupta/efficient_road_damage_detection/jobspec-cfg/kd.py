from sched import scheduler
from statistics import mean
from turtle import shape
import numpy as np
import torch
import os
from fast_seg import get_colorized_seg_map, get_fastseg_model
from get_segformer import get_segformer, get_segformer_feature_extractor
# from BiSeNet.tools.get_bisenet import get_bisenet
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import time
from tqdm import tqdm
from fastseg.image import colorize, blend
from get_dataloader import get_only_nsv_360_images_dataloaders
import torchvision
import torchmetrics
from PIL import Image 

writer = SummaryWriter('tensorboard')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
segformer = get_segformer().to(device)
segformerextractor = get_segformer_feature_extractor()
segformer.eval()

def teacher(inputs):
    im_list = [inputs[x] for x in range(inputs.shape[0])]
    inputs = segformerextractor(images = im_list, return_tensors="pt")['pixel_values'].to(device) 
    return segformer(inputs)

def get_two_classes(vector):
    total = torch.cat((torch.unsqueeze(vector[:, 0, :, :], dim = 1), torch.sum(vector, dim = 1, keepdim = True) - torch.unsqueeze(vector[:, 0, :, :], dim = 1)), dim = 1)
    arg = torch.argmax(vector, dim = 1, keepdim = True)
    arg[arg != 0] = 1
    arg[arg == 0] = 0
    return total, arg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_fn(student, ds_train, i, optimizer, scheduler):
    resizer_1 = torchvision.transforms.Resize(size = (1024, 1024))
    resizer_2 = torchvision.transforms.Resize(size = (128, 128))
    student.to(device)
    student.train() 
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean').to(device)
    jaccard = torchmetrics.JaccardIndex(19).to(device)
    k = 0
    for imgs in ds_train:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        imgs = resizer_1(imgs)
        waste, teacher_iou = get_two_classes(teacher(imgs)['logits'].to(device))
        teacher_label = torch.permute(torch.squeeze(torch.nn.functional.one_hot(teacher_iou, num_classes=2)), (0, 3, 1, 2))
        imgs = imgs.to(device)
        student_label, student_iou = get_two_classes(student(imgs).to(device))
        # print(torch.sum(torch.argmax(student_label) - student_iou))
        del imgs
        student_label = resizer_2(student_label)
        student_iou = resizer_2(student_iou)
        loss = loss_fn(student_label, teacher_label.float())
        del teacher_label
        # show_label = teacher_iou
        # show_label_2 = student_iou
        iou = jaccard(teacher_iou, student_iou)
        del student_iou
        del teacher_iou
        print(f'epoch: {i} | train_loss : {loss.item()} | train_IOU : {iou.item()} | lr: {get_lr(optimizer)}')
        loss.backward()
        optimizer.step()
        del student_label
        # show_label = torch.squeeze(show_label[0], dim = 0).detach().cpu().numpy()
        # colorized = colorize(show_label)
        # og_img = (255*imgs[0]).permute((1, 2, 0)).type(torch.uint8).detach().cpu().numpy()
        # og_img = Image.fromarray(og_img)
        # og_img.save('og_img.png')
        # colorized.save('tester.png')
        # show_label_2 = torch.squeeze(show_label_2[0], dim = 0).detach().cpu().numpy()
        # colorized = colorize(show_label_2)
        # og_img_2 = (255*imgs[0]).permute((1, 2, 0)).type(torch.uint8).detach().cpu().numpy()
        # og_img_2 = Image.fromarray(og_img_2)
        # # og_img_2.save('og_img.png')
        # colorized.save('tester_2.png')
        # print('dshjb')
        # exit()
        writer.add_scalar('train_iou', iou.item(), i*len(ds_train) + k)
        writer.add_scalar('train_loss', loss.item(), i*len(ds_train) + k)
        torch.cuda.empty_cache()
        # if k % 000 == 0 and k != 0:
        k = k + 1
    scheduler.step()    
    return student

def test_fn(student, ds_test, i):
    resizer_1 = torchvision.transforms.Resize(size = (1024, 1024))
    resizer_2 = torchvision.transforms.Resize(size = (128, 128))
    student.eval() 
    student.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean').to(device)
    jaccard = torchmetrics.JaccardIndex(19).to(device)
    total_iou = 0
    k = 0
    for imgs in ds_test:
        imgs = resizer_1(imgs)
        waste, teacher_iou = get_two_classes(teacher(imgs)['logits'].to(device))
        teacher_label = torch.permute(torch.squeeze(torch.nn.functional.one_hot(teacher_iou, num_classes=2)), (0, 3, 1, 2))
        imgs = imgs.to(device)
        student_label, student_iou = get_two_classes(student(imgs).to(device))
        del imgs
        student_label = resizer_2(student_label)
        student_iou = resizer_2(student_iou)
        loss = loss_fn(student_label, teacher_label.float())
        del teacher_label
        iou = jaccard(teacher_iou, student_iou)
        del student_iou
        del teacher_iou
        del student_label
        total_iou += iou
        print(f'test_loss : {loss.item()} | test_IOU : {iou.item()}')
        writer.add_scalar('test_loss', loss.item(), i*len(ds_test) + k)
        writer.add_scalar('test_iou', iou.item(), i*len(ds_test) + k)
        k = k + 1
        torch.cuda.empty_cache()
    mean_iou = total_iou/len(ds_test)
    # writer.add_scalar('mean_test_iou', mean_iou, i)
    print(f'*************************** | mean_iou : {mean_iou} | ***************************')
    torch.save(student, f'students/{mean_iou}.pt')
    
    return student

def kd(student, ds_train, ds_test, num_epochs):
    t = 0
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 6, gamma=0.4, last_epoch=- 1, verbose=True)
    while t < num_epochs:
        student = train_fn(student, ds_train, t, optimizer, scheduler)
        student = test_fn(student, ds_test, t)
        t = t + 1


if __name__ == '__main__':
    stud_name = 'fast_seg' # bisenet, fast_seg
    if stud_name == 'fast_seg':
        student = get_fastseg_model()
    # if stud_name == 'bisenet':
    #     student = get_bisenet()
    # teacher = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", force_download=False)
    
    ds_train, ds_test = get_only_nsv_360_images_dataloaders(3, True)
    kd(student, ds_train, ds_test, 50)
    writer.close()