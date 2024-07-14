r""" Cross-Domain Few-Shot Semantic Segmentation testing code """
import argparse

import torch.nn as nn
import torch

# from model.patnet import PATNetwork
from model.patnet_vdb import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from datetime import datetime
import torch.optim as optim



def test(model, dataloader, nshot):
    r""" Test PATNet """

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    # model.requires_grad_(True)
    mean_time = 0
    size = 0
    LR = 0.001

    for idx, batch in enumerate(dataloader):
        # 1. PATNetworks forward pass
        batch = utils.to_cuda(batch)

        ###
        start_time = datetime.now()
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        end_time = datetime.now()

        # print('Duration: {}'.format(end_time - start_time))
        mean_time += (end_time - start_time).total_seconds()
        size += 1 
        ###
        assert pred_mask.size() == batch['query_mask'].size()
        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)
        # break 

    print('Average Duration: {}'.format(mean_time/size))
    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou

def finetuning(model, dataloader, optimizer_ft, nshot, epoch):
    # utils.fix_randseed(0)
    utils.fix_randseed(None) 
    average_meter = AverageMeter(dataloader.dataset)
    model.module.train_mode()
    torch.set_grad_enabled(True)  

    k = 0 
    for idx, batch in enumerate(dataloader):
        k += 1 
        # 1. PATNetworks forward pass
        batch = utils.to_cuda(batch)
        # query_img = batch['query_img']
        # support_imgs = batch['support_imgs']
        # support_masks = batch['support_masks']

        # query_img.requires_grad = True
        # support_imgs.requires_grad = True
        # support_masks.requires_grad = True

        # logit_mask = model(query_img, support_imgs.squeeze(1), support_masks.squeeze(1))

        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # loss = model.module.finetune_reference(batch, batch['query_mask'], nshot=nshot)
        loss = model.module.finetune_reference(batch, pred_mask, nshot=nshot)
        # print(loss)
        # loss = model.module.finetune_reference(batch, pred_mask, nshot=nshot)
        # loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        # loss = model.module.compute_objective(logit_mask, pred_mask)

        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
        # if k > 20:
        #     break 

    # Write evaluation results
    average_meter.write_result('Finetuning', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./Datasets_PATNET')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['fss', 'deepglobe', 'isic', 'lung'])
    parser.add_argument('--logpath', type=str, default='./')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='path_to_your_trained_model')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    # parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--backbone', type=str, default='resnet50_vdb', choices=['vgg16', 'resnet50', 'resnet50_vdb'])
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = PATNetwork(args.backbone)
    # model.module.train_mode()
    # model.train()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)


    model.module.train_mode()
    # Test PATNet
    LR = 0.0001
    params_to_update = []
    for name,param in model.named_parameters():
        # if param.requires_grad == True and ('reference_layer' in name or 'hpn_learner' in name):
        if param.requires_grad == True and 'reference_layer' in name:
            # print(name)
            params_to_update.append(param)

    # for name,param in model.named_parameters():
    #     params_to_update.append(param)

    # optimizer_ft = optim.SGD(params_to_update, lr=LR, momentum=0.9)
    optimizer_ft = optim.Adam([{"params":params_to_update, 'lr':LR}])
    # optimizer_ft = optim.Adam([{"params": model.parameters(), 'lr':LR}])
    for epoch in range(1):
        trn_loss, trn_miou, trn_fb_iou = finetuning(model, dataloader_test, optimizer_ft, args.nshot, epoch)
    
    model.module.eval()
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    
    # torch.set_grad_enabled(True)  # Context-manager 
    # test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info("Finetuning with the batch['query_mask']")

    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
