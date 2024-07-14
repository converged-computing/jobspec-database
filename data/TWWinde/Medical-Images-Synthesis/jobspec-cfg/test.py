import cv2
from PIL import Image
from numpy import generic
from torchvision.transforms import functional
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from utils.fid_scores import fid_pytorch
from utils.drn_segment import drn_105_d_miou as drn_105_d_miou
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import os
from utils.Metrics import metrics
import nibabel as nib


generate_images = True
compare_miou = False
compute_miou_generation = False
compute_fid_generation = False
compute_miou_segmentation_network = False
generate_niffti = False
compute_metrics = True

#from models.generator import WaveletUpsample, InverseHaarTransform, HaarTransform, WaveletUpsample2

#wavelet_upsample = WaveletUpsample()

# from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
# xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
# ifm = DWTInverse(mode='zero', wave='db3')

from utils.utils import tens_to_im
import numpy as np
#from torch.autograd import Variable


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    if np.sum(union) == 0:
        return 0.0
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_miou(folder1, folder2):
    image_name_list1 = [f for f in sorted(os.listdir(folder1)) if f.endswith(".png")]
    image_name_list2 = [f for f in sorted(os.listdir(folder2)) if f.endswith(".png")]
    num_classes = 37
    class_ious = np.zeros(num_classes)
    image_name_list = list(set(image_name_list1) & set(image_name_list2))
    for class_idx in range(num_classes):
        ious = []
        for image_name in image_name_list:
            # mask_img = np.array(Image.open(os.path.join(gt_folder, gt_file)).convert('L')).astype(np.uint8)
            # print(mask_img)
            mask1 = np.array(Image.open(os.path.join(folder1, image_name))).astype(np.uint8) == class_idx
            # print('shape1',pred_mask.shape)
            mask2 = np.array(Image.open(os.path.join(folder2, image_name))).astype(np.uint8) == class_idx
            # print('shape2',gt_mask.shape)
            iou = compute_iou(mask1, mask2)
            ious.append(iou)
        nonzero_ious = ious[ious != 0]
        class_ious[class_idx] = np.mean(nonzero_ious)
        print('Class idx', class_idx, 'ious', np.mean(ious))
    mIoU = np.mean(class_ious)
    print('mIoU ', mIoU)
    return mIoU


from collections import namedtuple

# --- read options ---#
opt = config.read_arguments(train=False)
print(opt.phase)

# --- create dataloader ---#
_, _, dataloader_val = dataloaders.get_dataloaders(opt)

# --- create utils ---#
image_saver_combine = utils.results_saver(opt)
image_saver = utils.results_saver_for_test(opt)
metrics_computer = metrics(opt, dataloader_val)
fid_computer = fid_pytorch(opt, dataloader_val)
# --- create models ---#
model = models.Unpaired_model(opt)
# model = models.Unpaired_model_cycle(opt)
model = models.put_on_multi_gpus(model, opt)
utils.load_networks(opt, model)
model.eval()

mae = []
mse = []

if compute_metrics:
    metrics_computer.metrics_test(model)
    fid_computer.fid_test(model)



if generate_niffti:
    j=0
    k=0
    # --- iterate over validation set ---#
    niffti = []
    label_niffti = []
    for i, data_i in tqdm(enumerate(dataloader_val)):

        label_save = data_i['label'].long()
        label_save = np.array(label_save).astype(np.uint8).squeeze(1)
        groundtruth, label = models.preprocess_input(opt, data_i)
        generated = model(None, label, "generate", None).cpu().detach()
        name_ = data_i["name"]

        for b in range(1, len(generated)):
            j += 1
            name = name_[b]
            # print(name)
            name_label = name_[b].split('/')[-1]
            # print(name_label)
            name_label_ = name_[b-1].split('/')[-1]
            per = int(name_label.split('_')[1])
            per_ = int(name_label_.split('_')[1])
            num = int(name_label_.split('_')[-1].split('.')[0])
            # print(generated[b].shape) [3, 256, 256]
            if per_ == per:
                one_channel = np.mean(generated[b].numpy(), axis=0)  # rgb to grey
                #one_channel = one_channel * 1000
                label_niffti.append(label_save[b])
                niffti.append(one_channel)
            else:
                image_array = np.array(niffti)
                label_array = np.array(label_niffti)
                print(image_array.shape)
                print(label_array.shape)
                k += 1
                nifti_image = nib.Nifti1Image(image_array, affine=np.eye(4))
                os.makedirs(f'/no_backups/s1449/Medical-Images-Synthesis/results/medicals/MOOSEv2_data/S{k}/', exist_ok=True)
                nib.save(nifti_image, f'/no_backups/s1449/Medical-Images-Synthesis/results/medicals/MOOSEv2_data/S{k}/CT_S{k}.nii.gz')
                nifti_label = nib.Nifti1Image(label_array, affine=np.eye(4))
                os.makedirs(f'/no_backups/s1449/Medical-Images-Synthesis/results/medicals/3d_label/', exist_ok=True)
                nib.save(nifti_label, f'/no_backups/s1449/Medical-Images-Synthesis/results/medicals/3d_label/label_S{k}.nii.gz')
                niffti = []
                label_niffti = []
                j = 0
        if k == 10:
            break





if generate_images:
    j=0
    k=0
    # --- iterate over validation set ---#
    for i, data_i in tqdm(enumerate(dataloader_val)):
        j+=1
        k+=1
        label_save = data_i['label'].long()
        label_save = np.array(label_save).astype(np.uint8).squeeze(1)
        groundtruth, label = models.preprocess_input(opt, data_i)
        #generated = model(None, label, "generate", None).cpu().detach()
        generated1 = model(None, label, "generate", None).cpu().detach()
        generated2 = model(None, label, "generate", None).cpu().detach()
        generated3 = model(None, label, "generate", None).cpu().detach()
        generated4 = model(None, label, "generate", None).cpu().detach()
        arr = generated1.numpy()

        image_saver(label, generated1, groundtruth, data_i["name"])

        #image_saver_combine(label, generated1, generated2, generated3, generated4, groundtruth, data_i["name"])
        if k == 303:
            pass

        if j == 200:
            break
        # plt.imshow(tens_to_im(generated[0]))
        # downsampled = torch.nn.functional.interpolate(generated,scale_factor = 0.5)
        # plt.figure()
        # downsampled_wavlet = HaarTransform(3,four_channels=False,levels=1)(downsampled)
        # downsampled_wavlet = HaarTransform(in_channels=3,four_channels=False)(downsampled)

        # upsampled_wavelet = WaveletUpsample2()(downsampled_wavlet)
        # plt.imshow(tens_to_im(downsampled_wavlet[0]))
        # upsampled = InverseHaarTransform(3,four_channels=False,levels=2)(upsampled_wavelet)
        # error = tens_to_im(upsampled[0]-generated[0])-0.5
        # error = tens_to_im(torch.nn.functional.interpolate(torch.nn.functional.interpolate(generated[0],scale_factor = 0.5),scale_factor = 2)-generated[0])-0.5
        # mae.append(np.absolute(error).mean())
        # mse.append(np.sqrt(np.multiply(error,error).mean()))
        # plt.figure()
        # plt.imshow(tens_to_im(upsampled[0]-generated[0]))
        # plt.show()


# print(np.array(mae).mean())
# print(np.array(mse).mean())

if compare_miou:
    pred_folder_generated = os.path.join('/no_backups/s1449/OASIS/results', 'medicals', 'test', 'segmentation')
    pred_folder_real = os.path.join('/no_backups/s1449/OASIS/results', 'medicals', 'test',
                                         'segmentation_real')
    gt_folder = os.path.join('/no_backups/s1449/OASIS/results', 'medicals', 'test', 'label')
    answer = compute_miou(pred_folder_generated, pred_folder_real)
    print('miou of fake images segmentation and real images segmentation ', answer)
    answer = compute_miou(pred_folder_generated, gt_folder)
    print('miou of fake images segmentation and gt label ', answer)
    answer = compute_miou(pred_folder_real, gt_folder)
    print('miou of real images segmentation and gt label ', answer)
    print('#####################################new model#########################################')
    pred_folder_generated = os.path.join('/no_backups/s1449/Medical-Images-Synthesis/results', 'medicals', 'test', 'segmentation')
    pred_folder_real = os.path.join('/no_backups/s1449/Medical-Images-Synthesis/results', 'medicals', 'test',
                                    'segmentation_real')
    gt_folder = os.path.join('/no_backups/s1449/Medical-Images-Synthesis/results', 'medicals', 'test', 'label')
    answer = compute_miou(pred_folder_generated, pred_folder_real)
    print('miou of fake images segmentation and real images segmentation ', answer)
    answer = compute_miou(pred_folder_generated, gt_folder)
    print('miou of fake images segmentation and gt label ', answer)
    answer = compute_miou(pred_folder_real, gt_folder)
    print('miou of real images segmentation and gt label ', answer)

'''print(drn_105_d_miou(opt.results_dir,opt.name,'latest'))
print(drn_105_d_miou(opt.results_dir,opt.name,'20000'))
print(drn_105_d_miou(opt.results_dir,opt.name,'40000'))
print(drn_105_d_miou(opt.results_dir,opt.name,'60000'))'''

if compute_miou_generation:
    print(drn_105_d_miou(opt.results_dir, opt.name, opt.ckpt_iter))
else:
    np_file = np.load(os.path.join(opt.checkpoints_dir, opt.name, 'MIOU', "miou_log.npy"))
    first = list(np_file[0, :])
    sercon_miou = list(np_file[1, :])
    # first.append(epoch)
    # sercon.append(cur_fid)
    np_file = [first, sercon_miou]
    print('max miou score is :')
    if opt.ckpt_iter == 'latest':
        print(sercon_miou[-1])
    elif opt.ckpt_iter == 'best':
        print(np.max(sercon_miou))
    else:
        print(sercon_miou[first.index(float(opt.ckpt_iter))])

if compute_fid_generation:
    fid_computer = fid_pytorch(opt, dataloader_val)
    fid_computer.fid_test(model)
else:
    np_file = np.load(os.path.join(opt.checkpoints_dir, opt.name, 'FID', "fid_log.npy"))
    first = list(np_file[0, :])
    sercon = list(np_file[1, :])
    # first.append(epoch)
    # sercon.append(cur_fid)
    np_file = [first, sercon]
    # print('fid score is :')
    if opt.ckpt_iter == 'latest':
        print(sercon[-1])
    elif opt.ckpt_iter == 'best':
        print('min fid : ', np.min(sercon))
        index = sercon.index(np.min(sercon))
        # print(len(sercon))
        # print(len(sercon_miou))
        # print(index)
        print('miou : ', sercon_miou[index])

    else:
        print(sercon[first.index(float(opt.ckpt_iter))])

