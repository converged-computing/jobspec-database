from torch.utils.data import Dataset
import cv2
from utils.util import *
from Model.HRNet_3D import *
import config

import argparse
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HRNet')
    parser.add_argument('-r', '--root', type=str, default='../ISBI2015/', help='data_root')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-s', '--seed', type=str, default=20, help="seed string")  # 随机种子不要一样
    parser.add_argument('--log_interval', type=str, default=1, help="log_interval")
    parser.add_argument('--resume', type=bool, default=False, help="if load model")
    parser.add_argument('--save_dir', type=str, default="./results_HRNet_ISBI_march", help="save directory")
    opt = parser.parse_args()
    print(opt)
    if not os.path.isdir(opt.save_dir):
        os.mkdir(opt.save_dir)
    setup_seed(opt.seed)
    k_flod=5
    for k in range(k_flod):
    # for k in [4,0]:
        print("The "+str(k+1)+" flod")
        train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list=load_data_aug(opt.root,k)
        print("successfully loaded data")
        train_dataset = CustomDataset(train_img_masks, transforms=transforms.Compose([ToTensor()]))
        val_dataset = CustomDataset(val_img_masks, transforms=transforms.Compose([ToTensor()]))
        print("train data number: ",len(train_dataset))
        print("test data number: ",len(val_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        model_save_path = os.path.join(opt.save_dir,str(k+1)+"_flod")  # directory to same the model after each epoch.

        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        # net=  get_pose_net(config.cfg, 1)
        net = torch.nn.DataParallel(get_pose_net(config.cfg,1, 1), device_ids=[0, 1])
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
        if opt.resume:
            save_name = os.path.join(model_save_path, 'model_best.pth')
            load_checkpoint(save_name, net, optimizer)
        criterion = nn.BCELoss()
        N_train = len(train_img_masks)
        max_dice = 0
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters in network: ', n_params)

        # val_dice = eval_net_batch(net, val_loader, real_test_mask_list, device)
        # print('Validation Dice Coeff: {}'.format(val_dice))

        for epoch in range(opt.epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, opt.epochs))
            net.train()
            epoch_loss = 0
            #training
            for i, b in enumerate(train_loader):

                imgs = b['img'].to(device)
                true_masks = b['label'].to(device)

                masks_pred = net(imgs).squeeze(0)
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()
                if i%20==0:
                    dice_score = (2 * torch.dot(masks_probs_flat, true_masks_flat) + 1) / (
                                torch.sum(masks_probs_flat) + torch.sum(true_masks_flat) + 1)
                    print('{0:.4f} --- loss: {1:.6f}, dice_score:{2:.6f}'.
                          format(i * opt.batch_size / N_train, loss.item(),dice_score))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch finished ! Loss: {}'.format(epoch_loss / (len(train_loader)*opt.batch_size)))

            if epoch>=0:
                val_dice = eval_net_batch(net, val_loader, real_test_mask_list, device)
                print('Validation Dice Coeff: {}'.format(val_dice))
                if np.mean(val_dice) > max_dice:
                    max_dice = np.mean(val_dice)
                    print('New best performance! saving')
                    save_name = os.path.join(model_save_path, 'model_best.pth')
                    save_checkpoint(save_name, net, optimizer)
                    print('model saved to {}'.format(save_name))
                if (epoch + 1) % opt.log_interval == 0:
                    save_name = os.path.join(model_save_path, 'model_routine.pth')
                    save_checkpoint(save_name, net, optimizer)
                    print('model saved to {}'.format(save_name))
