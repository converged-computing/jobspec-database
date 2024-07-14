import models
import datas
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import re
import numpy as np
from utils.config import Config
from collections import OrderedDict
import sys
import cv2
from utils.vis_flow import flow_to_color
from piqa import PSNR, SSIM
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAccuracy, \
                                        BinaryF1Score, BinaryRecall, BinaryPrecision, \
                                        BinaryAUROC
from PIL import Image
import matplotlib.pyplot as plt
from models.segmentation_models_pytorch.unet import Unet
from models.segmentation_models_pytorch.unet_rrdb import Unet_RRDB
from models.discriminator_model.discriminator import Discriminator_square, \
                                                    Discriminator_non_square, \
                                                    Discriminator_square_dumb
from models.discriminator_model.patch_discriminators import Discriminator_1x1_output, \
                                                        Discriminator_256x256_output, \
                                                        Discriminator_30x30_output, \
                                                        Discriminator_252x252_output, \
                                                        Discriminator_patch, \
                                                        dcgan_weights_init
import preprocess
from pytorch3d.loss import chamfer_distance
from lpips_pytorch import lpips

import pdb


def save_flow_to_img(flow, des, epoch):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '_epoch_'+str(epoch)+'.jpg', cf)


def save_mask_to_img(mask, name):
    #note mask should be np array
    if mask.shape[0] == 1:
        mask = np.transpose(mask, (1, 2, 0))
    cv2.imwrite(name, (1 - mask)*255)


def plot_loss(config, epoch_arr, loss_arr, loss_1_arr, loss_2_arr, \
                test_epoch_arr, test_loss_arr, test_loss_1_arr, \
                test_loss_2_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    
    fig, axs = plt.subplots(2, 2)
    axs[0,0].set_title('generator training loss')
    axs[0,0].plot(epoch_arr, loss_arr, 'blue')
    axs[0,1].set_title('streams 1/2 training loss (red/purple)')
    axs[0,1].plot(epoch_arr, loss_1_arr, 'red')
    axs[0.1].plot(epoch_arr, loss_2_arr, 'purple')
    axs[1,0].set_title('generator test loss')
    axs[1,0].plot(test_epoch_arr, test_loss_arr, 'blue')
    axs[1,1].set_title('streams 1/2 test loss (red/purple)')
    axs[1,1].plot(epoch_arr, test_loss_1_arr, 'red')
    axs[1.1].plot(epoch_arr, test_loss_2_arr, 'purple')

    filename = os.path.join(config.metrics_dir, config.loss_img_path)
    plt.savefig(filename)
    plt.close('all')


def save_training_metrics_to_arr(config, epoch_arr, loss_arr, loss_1_arr, loss_2_arr, \
                        test_epoch_arr, test_loss_arr, test_loss_1_arr, test_loss_2_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    epoch_arr_path = os.path.join(config.metrics_dir, "epoch_arr.npy")
    np.save(epoch_arr_path, epoch_arr)
    loss_arr_path = os.path.join(config.metrics_dir, "loss_arr.npy")
    np.save(loss_arr_path, loss_arr)
    loss_1_path = os.path.join(config.metrics_dir, "loss_1_arr.npy")
    np.save(loss_arr_path, loss_1_arr)
    loss_2_path = os.path.join(config.metrics_dir, "loss_2_arr.npy")
    np.save(loss_arr_path, loss_2_arr)
    
    test_epoch_arr_path = os.path.join(config.metrics_dir, "test_epoch_arr.npy")
    np.save(test_epoch_arr_path, test_epoch_arr)
    test_loss_arr_path = os.path.join(config.metrics_dir, "test_loss_arr.npy")
    np.save(test_loss_arr_path, test_loss_arr)
    test_loss_1_arr_path = os.path.join(config.metrics_dir, "test_loss_1_arr.npy")
    np.save(test_loss_1_arr_path, test_loss_1_arr)
    test_loss_2_arr_path = os.path.join(config.metrics_dir, "test_loss_2_arr.npy")
    np.save(test_loss_2_arr_path, test_loss_2_arr)


def save_all_metrics_to_txt_and_npy(config, cur_psnr, cur_ssim, cur_cham, cur_lpips, \
                                cur_auroc, cur_accs, cur_precs, cur_recalls, cur_f1s, epoch):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "psnr_ssim_epoch_"+str(epoch)+".txt")
    with open(textpath, 'w') as f:
        f.write("~~~~~~~~~~ PERCEPTUAL ~~~~~~~~~~~~~~\n")
        f.write("psnr "+str(cur_psnr)+"\n")
        f.write("ssim "+str(cur_ssim)+"\n")
        f.write("cham "+str(cur_cham)+"\n")
        f.write("lpips "+str(cur_lpips)+"\n")
        f.write("~~~~~~~~~~~ CLASSIFICATION ~~~~~~~~~~~~~~\n")
        f.write("auroc "+str(cur_auroc)+"\n")
        f.write("accuracy "+str(cur_accs)+"\n")
        f.write("precision "+str(cur_precs)+"\n")
        f.write("recall "+str(cur_recalls)+"\n")
        f.write("f1 "+str(cur_f1s)+"\n")
    np.save(os.path.join(config.metrics_dir, "acc_arr_epoch_{}.npy".format(epoch)), cur_accs)
    np.save(os.path.join(config.metrics_dir, "prec_arr_epoch_{}.npy".format(epoch)), cur_precs)
    np.save(os.path.join(config.metrics_dir, "recall_arr_epoch_{}.npy".format(epoch)), cur_recalls)
    np.save(os.path.join(config.metrics_dir, "f1_arr_epoch_{}.npy".format(epoch)), cur_f1s)


def plot_roc_curve(config, cur_precs, cur_recalls, epoch):
    plt.plot(cur_recalls, cur_precs)
    plt.xlabel("recall")
    plt.ylabel("precision")
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    figpath = os.path.join(config.metrics_dir, "roc_curve_epoch"+str(epoch)+".png")
    plt.savefig(figpath)
    plt.clf()


def training_loop(epoch, model, model_1, model_2, trainloader, \
                    trainset, optimizer, optimizer_1, optimizer_2, \
                    scheduler, scheduler_1, scheduler_2, \
                    config, loss_arr, loss_1_arr, loss_2_arr ,\
                    to_img):
    #  start training...
    model.train()
    running_loss = 0.0
    running_1_loss = 0.0
    running_2_loss = 0.0

    threshold = 0.5

    for trainIndex, trainData in enumerate(trainloader):
        if trainIndex % 10 == 0:
            print('Training {}/{}-th group...'.format(trainIndex, len(trainloader)))
        sys.stdout.flush()

        rgb_sample, masks, flow, folder, index = trainData

        #get flow
        flow_15, flow_51, flow_13, flow_31, flow_35, flow_53 = flow[0]

        F15 = flow_15.float().cuda().to(memory_format=torch.channels_last)
        F51 = flow_51.float().cuda().to(memory_format=torch.channels_last)
        F13 = flow_13.float().cuda().to(memory_format=torch.channels_last)
        F31 = flow_31.float().cuda().to(memory_format=torch.channels_last)
        F35 = flow_35.float().cuda().to(memory_format=torch.channels_last)
        F53 = flow_53.float().cuda().to(memory_format=torch.channels_last)

        srcmask = masks[0].cuda().float().to(memory_format=torch.channels_last)
        ibmask = masks[1].cuda().float().to(memory_format=torch.channels_last)
        trgmask = masks[2].cuda().float().to(memory_format=torch.channels_last)

        #weight bce loss by inverse distance from black pixel
        num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
        num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
        p_bg = num_bg/(num_bg+num_fg)
        loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)
        loss_weights = loss_weights.to(memory_format=torch.channels_last)
        
        #2 streams
        input_cat_1 = torch.cat([srcmask, F15], 1).to(memory_format=torch.channels_last)
        input_cat_2 = torch.cat([trgmask, F51], 1).to(memory_format=torch.channels_last)

        F13_output, decoder_output_1 = model_1(input_cat_1)
        F53_output, decoder_output_2 = model_2(input_cat_2)

        decoder_output_1 = decoder_output_1.to(memory_format=torch.channels_last)
        decoder_output_2 = decoder_output_2.to(memory_format=torch.channels_last)

        loss_13 = F.mse_loss(F13_output, F13)
        loss_53 = F.mse_loss(F53_output, F53)
        loss_streams = loss_13 + loss_53

        outpath = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        if epoch % 10 == 0:
            save_flow_to_img(F15.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F15"), epoch)
            save_flow_to_img(F51.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F51"), epoch)
            save_flow_to_img(F13.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13"), epoch)
            save_flow_to_img(F53.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53"), epoch)
            save_flow_to_img(F13_output.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13_est"), epoch)
            save_flow_to_img(F53_output.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53_est"), epoch)
            srcmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
        
        input_cat = torch.cat([decoder_output_1, decoder_output_2], 1)
        input_cat = input_cat.to(memory_format=torch.channels_last)
        outputs, _ = model(input_cat)
        outputs = outputs.to(memory_format=torch.channels_last)
        output_answer = torch.where(outputs > threshold, 1., 0.)

        model.zero_grad()
        optimizer.zero_grad()

        # calculate loss(es)
        loss = torch.mean(torch.mul(F.binary_cross_entropy_with_logits(outputs, ibmask), loss_weights))
        loss += loss_streams
        loss.backward()
        optimizer.step()
        optimizer_1.step()
        if optimizer_2 != None:
            optimizer_2.step()
        running_loss += loss.detach().item()

        running_1_loss += loss_13.item()
        running_2_loss += loss_53.item()

        input_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
        output_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
        output_grayscale_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
        output_np = output_answer.cpu().detach().numpy()
        target_np = ibmask.cpu().detach().numpy()
        output_not_binarized_np = outputs.cpu().detach().numpy()
        
        if epoch % 10 == 0:
            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

    scheduler.step()
    scheduler_1.step()
    if scheduler_2 != None:
        scheduler_2.step()
    cur_loss = running_loss / (len(trainloader))
    cur_1_loss = running_1_loss / (len(trainloader))
    cur_2_loss = running_2_loss / (len(trainloader))

    print("epoch", epoch, "loss", cur_loss, "discrim loss", cur_d_loss)
    loss_arr.append(cur_loss)
    loss_1_arr.append(cur_1_loss)
    loss_2_arr.append(cur_2_loss)

    return loss_arr, loss_1_arr, loss_2_arr


def testing_loop(epoch, model, model_1, model_2, testloader, testset, config, \
                    to_img, test_loss_arr, test_loss_1_arr, test_loss_2_arr):    
    default_threshold = 0.5
    thresholds_1 = np.arange(10)*0.1
    thresholds_2 = (np.arange(10)*0.01)+0.9
    thresholds = np.concatenate((thresholds_1, thresholds_2))
    
    running_loss = 0.0
    running_1_loss = 0.0
    running_2_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_cham = 0.0
    running_lpips = 0.0
    running_auroc = 0.0
    running_accuracies = np.zeros(len(thresholds))
    running_precisions = np.zeros(len(thresholds))
    running_recalls = np.zeros(len(thresholds))
    running_f1s = np.zeros(len(thresholds))

    psnr = PSNR().cuda()
    ssim = SSIM(n_channels=1).cuda()
    auroc = BinaryAUROC(thresholds=torch.Tensor(thresholds))

    #  start testing...
    with torch.no_grad():
        model.eval()

        for validationIndex, validationData in enumerate(testloader):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testloader)))
            sys.stdout.flush()

            rgb_sample, masks, flow, folder, index = validationData

            flow_15, flow_51, flow_13, flow_31, flow_35, flow_53 = flow[0]

            F15 = flow_15.float().cuda().to(memory_format=torch.channels_last)
            F51 = flow_51.float().cuda().to(memory_format=torch.channels_last)
            F13 = flow_13.float().cuda().to(memory_format=torch.channels_last)
            F31 = flow_31.float().cuda().to(memory_format=torch.channels_last)
            F35 = flow_35.float().cuda().to(memory_format=torch.channels_last)
            F53 = flow_53.float().cuda().to(memory_format=torch.channels_last)

            srcmask = masks[0].cuda().float().to(memory_format=torch.channels_last)
            ibmask = masks[1].cuda().float().to(memory_format=torch.channels_last)
            trgmask = masks[2].cuda().float().to(memory_format=torch.channels_last)

            #weight bce loss by inverse distance from black pixel
            num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
            num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
            p_bg = num_bg/(num_bg+num_fg)
            loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)
            loss_weights = loss_weights.to(memory_format=torch.channels_last)

            #2 streams
            input_cat_1 = torch.cat([srcmask, F15], 1)
            input_cat_2 = torch.cat([trgmask, F51], 1)
            input_cat_1 = input_cat_1.to(memory_format=torch.channels_last)
            input_cat_2 = input_cat_2.to(memory_format=torch.channels_last)

            F13_output, decoder_output_1 = model_1(input_cat_1)
            F53_output, decoder_output_2 = model_2(input_cat_2)
            decoder_output_1 = decoder_output_1.to(memory_format=torch.channels_last)
            decoder_output_2 = decoder_output_2.to(memory_format=torch.channels_last)

            loss_13 = F.mse_loss(F13_output, F13)
            loss_53 = F.mse_loss(F53_output, F53)

            outpath = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            srcmask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            save_flow_to_img(F15.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F15"), epoch)
            save_flow_to_img(F51.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F51"), epoch)
            save_flow_to_img(F13.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13"), epoch)
            save_flow_to_img(F53.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53"), epoch)
            save_flow_to_img(F13_output.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13_est"), epoch)
            save_flow_to_img(F53_output.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53_est"), epoch)
            
            input_cat = torch.cat([decoder_output_1, decoder_output_2], 1)
            outputs, _ = model(input_cat)
            outputs = outputs.to(memory_format=torch.channels_last)
            # outputs = torch.sigmoid(outputs)
            output_answer = torch.where(outputs > default_threshold, 1., 0.)

            loss = torch.mean(torch.mul(F.binary_cross_entropy_with_logits(outputs, ibmask), loss_weights))
            running_loss += loss.detach().item()
            running_1_loss += loss_13.item()
            running_2_loss += loss_53.item()
            running_psnr += psnr(output_answer.detach(), ibmask.detach())
            running_ssim += ssim(output_answer.detach(), ibmask.detach())
            running_cham += chamfer_distance(output_answer.detach()[0], ibmask.detach()[0])[0].detach().cpu().item()
            running_lpips += lpips(output_answer, ibmask, net_type='vgg', version='0.1').detach().cpu().squeeze().item()
            running_auroc += auroc(outputs.cpu().detach(), ibmask.cpu().detach().int()).detach().item()

            for i, threshold_in in enumerate(thresholds):
                acc_metric = BinaryAccuracy(threshold=threshold_in)
                prec_metric = BinaryPrecision(threshold=threshold_in)
                recall_metric = BinaryRecall(threshold=threshold_in)
                f1_metric = BinaryF1Score(threshold=threshold_in)

                running_accuracies[i] += acc_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()
                running_precisions[i] += prec_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()
                running_recalls[i] += recall_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()
                running_f1s[i] += f1_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()

            input_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
            output_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
            output_grayscale_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
            output_np = output_answer.data.cpu().detach().numpy()
            target_np = ibmask.cpu().detach().numpy()
            output_not_binarized_np = outputs.cpu().detach().numpy()

            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

        cur_loss = running_loss / len(testloader)
        cur_1_loss = running_1_loss / len(testloader)
        cur_2_loss = running_2_loss / len(testloader)
        cur_psnr = running_psnr / len(testloader)
        cur_ssim = running_ssim / len(testloader)
        cur_cham = running_cham / len(testloader)
        cur_lpips = running_lpips / len(testloader)
        cur_auroc = running_auroc / len(testloader)

        cur_accs = running_accuracies / len(testloader)
        cur_precs = running_precisions / len(testloader)
        cur_recalls = running_recalls / len(testloader)
        cur_f1s = running_f1s / len(testloader)

        print("epoch", epoch, "loss", cur_loss)
        test_loss_arr.append(cur_loss)
        test_loss_1_arr.append(cur_1_loss)
        test_loss_2_arr.append(cur_2_loss)

    return test_loss_arr, test_loss_1_arr, test_loss_2_arr, cur_psnr, \
            cur_ssim, cur_cham, cur_lpips, cur_auroc, \
            cur_accs, cur_precs, cur_recalls, cur_f1s



def main(config, args):
    print("tae is goin for it on ig")

    trainset = datas.AnimTripletWFlow(config.dataset_root, config.flow_root, \
                                    train=True, img_size=config.img_size, \
                                    random_reverse=config.random_reverse)
    testset = datas.AnimTripletWFlow(config.dataset_root, config.flow_root, \
                                    train=False, img_size=config.img_size, \
                                    random_reverse=config.random_reverse)

    sampler = torch.utils.data.RandomSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, \
                                            batch_size=config.batch_size, \
                                            num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, \
                                            batch_size=config.test_batch_size, \
                                            shuffle=False, num_workers=config.num_workers)
    to_img = transforms.ToPILImage()
 
    sys.stdout.flush()

    # create the generator
    model_1 = Unet(
        encoder_name=config.stream_encoder_name,
        encoder_weights=config.stream_encoder_weights,
        encoder_depth=config.stream_encoder_depth,
        decoder_channels=config.stream_decoder_channels,
        in_channels=3,
        classes=2
    )
    model_2 = Unet(
        encoder_name=config.stream_encoder_name,
        encoder_weights=config.stream_encoder_weights,
        encoder_depth=config.stream_encoder_depth,
        decoder_channels=config.stream_decoder_channels,
        in_channels=3,
        classes=2
    )
    model = Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        encoder_depth=config.encoder_depth,
        decoder_channels=config.decoder_channels,
        in_channels=config.in_channels,
        classes=1
    )
    if config.checkpoint_in != None:
        dict1 = torch.load(config.checkpoint_in)
        newdict = OrderedDict()
        for key, value in dict1.items():
            name = key[7:]
            newdict[name] = value
        model.load_state_dict(newdict, strict=True)

        dict2 = torch.load(config.checkpoint_in_1)
        newdict2 = OrderedDict()
        for key, value in dict2.items():
            name = key[7:]
            newdict2[name] = value
        model_1.load_state_dict(newdict2, strict=True)

        dict3 = torch.load(config.checkpoint_in_2)
        newdict3 = OrderedDict()
        for key, value in dict3.items():
            name = key[7:]
            newdict3[name] = value
        model_2.load_state_dict(newdict3, strict=True)
    
    model = nn.DataParallel(model)
    model = model.cuda().to(memory_format=torch.channels_last)
    model_1 = nn.DataParallel(model_1)
    model_1 = model_1.cuda().to(memory_format=torch.channels_last)
    model_2 = nn.DataParallel(model_2)
    model_2 = model_2.cuda().to(memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.SequentialLR(
                                            optimizer,
                                            [
                                                optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=200),
                                                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                                            ],
                                            milestones=[200],
                                        )
    optimizer_1 = optim.Adam(model_1.parameters(), lr=config.lr_stream, betas=(config.beta1, 0.999))
    scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=config.num_epochs-1)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=config.lr_stream, betas=(config.beta1, 0.999))
    scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=config.num_epochs-1)

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    if config.checkpoint_in != None:
        epoch_arr = np.load(os.path.join(config.metrics_dir, "epoch_arr.npy"))
        epoch_arr = list(epoch_arr)
        loss_arr = np.load(os.path.join(config.metrics_dir, "loss_arr.npy"))
        loss_arr = list(loss_arr)
        test_epoch_arr = np.load(os.path.join(config.metrics_dir, "test_epoch_arr.npy"))
        test_epoch_arr = list(test_epoch_arr)
        test_loss_arr = np.load(os.path.join(config.metrics_dir, "test_loss_arr.npy"))
        test_loss_arr = list(test_loss_arr)
    else:
        epoch_arr = []
        loss_arr = []
        loss_1_arr = []
        loss_2_arr = []

        test_epoch_arr = []
        test_loss_arr = []
        test_loss_1_arr = []
        test_loss_2_arr = []

    for epoch in range(config.cur_epoch, config.num_epochs):
        print("######### EPOCH", epoch, "##########")
        epoch_arr.append(epoch)
        loss_arr, \
            loss_1_arr, \
            loss_2_arr = training_loop(epoch, model, model_1, \
                                    model_2, trainloader, trainset,\
                                    optimizer, optimizer_1, optimizer_2,\
                                    scheduler, scheduler_1, scheduler_2, \
                                    config, loss_arr, loss_1_arr, \
                                    loss_2_arr, to_img)
        if epoch % 10 == 0:
            test_epoch_arr.append(epoch)
            test_loss_arr, \
                test_loss_1_arr, \
                test_loss_2_arr, \
                cur_psnr, \
                cur_ssim, \
                cur_cham, \
                cur_lpips, \
                cur_auroc, \
                cur_accs, \
                cur_precs, \
                cur_recalls, \
                cur_f1s = testing_loop(epoch, model, model_1, \
                                        model_2, testloader, testset, config, \
                                        to_img, test_loss_arr, test_loss_1_arr, \
                                        test_loss_2_arr)
            save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, cur_cham, cur_lpips, \
                                cur_auroc, cur_accs, cur_precs, cur_recalls, cur_f1s, epoch)
            plot_roc_curve(config, cur_precs, cur_recalls, epoch)

            if not os.path.exists(config.checkpoint_latest_dir):
                os.makedirs(config.checkpoint_latest_dir)
            checkpoint_path = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+".pth")
            checkpoint_path_1 = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+"_1.pth")
            checkpoint_path_2 = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+"_2.pth")
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model_1.state_dict(), checkpoint_path_1)
            torch.save(model_2.state_dict(), checkpoint_path_2)
        
        plot_loss(config, epoch_arr, loss_arr, loss_1_arr, loss_2_arr, test_epoch_arr, test_loss_arr, \
                    test_loss_1_arr, test_loss_2_arr)
        save_training_metrics_to_arr(config, epoch_arr, loss_arr, loss_1_arr, loss_2_arr, \
                                    test_epoch_arr, test_loss_arr, test_loss_1_arr, test_loss_2_arr)
        
    return loss_arr


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    loss_arr, loss_d_arr = main(config, args)

    print("")
    print("######### RESULTS ###########")
    print("LOSS:", loss_arr)
    print("#############################")
    print("\a"*2)

