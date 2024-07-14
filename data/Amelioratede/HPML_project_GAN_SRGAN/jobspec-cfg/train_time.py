import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils import DIV2K_train_set, DIV2K_valid_set, FeatureExtractor, TV_Loss
import torchvision.transforms as transforms
import time

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_dir', type=str, default='./data/DIV2K_train_HR/', help='training dataset path')
parser.add_argument('--validset_dir', type=str, default='./data/DIV2K_valid_HR/', help='validation dataset path')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2,4,8], help='super resolution upscale factor')
parser.add_argument('--epochs', type=int, default=10, help='training epoch number')
parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
parser.add_argument('--mode', type=str, default='adversarial', choices=['adversarial', 'generator'], help='apply adversarial training')
parser.add_argument('--pretrain', type=str, default=None, help='load pretrained generator model')
parser.add_argument('--cuda', action='store_true', help='Using GPU to train')
parser.add_argument('--out_dir', type=str, default='./', help='The path for checkpoints and outputs')
parser.add_argument('--lr', type=float, default=0.0001, help='generator learning rate')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--dp', action='store_true', default=False, help='Using DataParallel')


sr_transform = transforms.Compose([
	transforms.Normalize((-1,-1,-1),(2,2,2)),
	transforms.ToPILImage()
	])


lr_transform = transforms.Compose([
	transforms.ToPILImage()
	])


opt = parser.parse_args()
upscale_factor = opt.upscale_factor
generator_lr = opt.lr
discriminator_lr = 0.0001

check_points_dir = opt.out_dir + 'check_points/'
weights_dir = opt.out_dir + 'weights/'
imgout_dir = opt.out_dir + 'output/'
os.makedirs(check_points_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(imgout_dir, exist_ok=True)
train_set = DIV2K_train_set(opt.trainset_dir, upscale_factor=4, crop_size = 128)
valid_set = DIV2K_valid_set(opt.validset_dir, upscale_factor=4)
trainloader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.bs, shuffle=True)
validloader = DataLoader(dataset=valid_set, num_workers=4, batch_size=1, shuffle=False)
generator_net = Generator(upscale_factor = upscale_factor, num_blocks=16)
discriminator_net = Discriminator()

adversarial_criterion = nn.BCELoss()
content_criterion = nn.MSELoss()
tv_reg = TV_Loss()
test_criterion = nn.CrossEntropyLoss()

generator_optimizer = optim.Adam(generator_net.parameters(), lr=generator_lr)
discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=discriminator_lr)
feature_extractor = FeatureExtractor()





if __name__ == '__main__':
	
	if torch.cuda.is_available() and opt.cuda:
		generator_net.cuda()
		discriminator_net.cuda()
		adversarial_criterion.cuda()
		content_criterion.cuda()
		feature_extractor.cuda()
		
	
	
	generator_running_loss = 0.0
	generator_losses = []
	discriminator_losses = []
	PSNR_valid = []
    	
	if opt.resume != 0:
		check_point = torch.load(check_points_dir + "check_point_epoch_" + str(opt.resume)+'.pth')
		generator_net.load_state_dict(check_point['generator'])
		generator_optimizer.load_state_dict(check_point['generator_optimizer'])
		generator_losses = check_point['generator_losses']
		PSNR_valid = check_point['PSNR_valid']
		if opt.mode == 'adversarial':
			discriminator_net.load_state_dict(check_point['discriminator'])
			discriminator_optimizer.load_state_dict(check_point['discriminator_optimizer'])
			discriminator_losses = check_point['discriminator_losses']

	if opt.pretrain != None:
		print('loading pre-trained SR ResNet')
		saved_G_state = torch.load(str(opt.pretrain))
		# generator_net.load_state_dict(saved_G_state['generator'])
		generator_net.load_state_dict(saved_G_state)

	if opt.dp:
		print('using DataParallel')
		generator_net = torch.nn.DataParallel(generator_net)
		discriminator_net = torch.nn.DataParallel(discriminator_net)
		torch.backends.cudnn.benchmark = True

	## Pre-train the generator
	if opt.mode == 'generator':
		print('pre-training the generator')
		for epoch in range(1+opt.resume, opt.epochs+1):
			print("epoch: %i/%i" % (int(epoch), int(opt.epochs)))
			generator_net.train()
			training_bar = tqdm(trainloader)
			training_bar.set_description('Running Loss: %f' % (generator_running_loss/len(train_set)))
			generator_running_loss = 0.0

			for hr_img, lr_img in training_bar:
				if torch.cuda.is_available() and opt.cuda:
					hr_img = hr_img.cuda()
					lr_img = lr_img.cuda()

				sr_img = generator_net(lr_img)

				content_loss = content_criterion(sr_img, hr_img)
				perceptual_loss = content_criterion(feature_extractor(sr_img), feature_extractor(hr_img))

				generator_loss = content_loss + 2e-8*tv_reg(sr_img) #  + 0.006*perceptual_loss
				
				generator_loss.backward()
				generator_optimizer.step()

				generator_running_loss += generator_loss.item() * hr_img.size(0)
				generator_net.zero_grad()

			torch.save(generator_net.state_dict(), weights_dir+ 'G_epoch_%d.pth' % (epoch))
			generator_losses.append((epoch,generator_running_loss/len(train_set)))
	
			if epoch  ==opt.epochs:
				with torch.no_grad():
					cur_epoch_dir = imgout_dir+str(epoch)+'/'
					os.makedirs(cur_epoch_dir, exist_ok=True)
					generator_net.eval()
					valid_bar = tqdm(validloader)
					img_count = 0
					psnr_avg = 0.0
					psnr = 0.0
					for hr_img, lr_img in valid_bar:
						valid_bar.set_description('Img: %i   PSNR: %f' % (img_count ,psnr))
						if torch.cuda.is_available():
							lr_img = lr_img.cuda()
							hr_img = hr_img.cuda()
						sr_tensor = generator_net(lr_img)
						mse = torch.mean((hr_img-sr_tensor)**2)
						psnr = 10* (torch.log10(1/mse) + np.log10(4))
						psnr_avg += psnr
						img_count +=1
						sr_img = sr_transform(sr_tensor[0].data.cpu())
						lr_img = lr_transform(lr_img[0].cpu())
						sr_img.save(cur_epoch_dir+'sr_' + str(img_count)+'.png')
						lr_img.save(cur_epoch_dir+'lr_'+str(img_count)+'.png')

					psnr_avg /= img_count
					PSNR_valid.append((epoch, psnr_avg.cpu()))

				check_point = {'generator': generator_net.state_dict(), 'generator_optimizer': generator_optimizer.state_dict(),'generator_losses': generator_losses ,'PSNR_valid': PSNR_valid}
				torch.save(check_point, check_points_dir + 'check_point_epoch_%d.pth' % (epoch))	
				np.savetxt(opt.out_dir + "generator_losses", generator_losses, fmt='%i,%f')
				np.savetxt(opt.out_dir + "PSNR", PSNR_valid, fmt='%i, %f')


	## Adversarial training
	if opt.mode == 'adversarial':
		total_time_list=[]
		train_time_list=[]
		compute_time_list=[]
		generator_time_list=[]
		discriminator_time_list=[]
    
		print('adversarial training')
		discriminator_running_loss = 0.0
        
		total_time_start=time.perf_counter()
        
		for epoch in range(1+opt.resume, opt.epochs+1):
			print("epoch: %i/%i" % (int(epoch), int(opt.epochs)))
			generator_net.train()
			discriminator_net.train()

			training_bar = tqdm(trainloader)
			training_bar.set_description('G: %f    D: %f' % (generator_running_loss/len(train_set), discriminator_running_loss/len(train_set)))
			generator_running_loss = 0.0
			discriminator_running_loss = 0.0
            
			total_time=0.0
			train_time=0.0
			compute_time=0.0
			generator_time=0.0
			discriminator_time=0.0
			total_time_start=time.perf_counter()
			for hr_img, lr_img in training_bar:
				hr_labels = torch.from_numpy(np.random.random((hr_img.size(0),1)) * 0.1 + 0.95).float()
				# sr_labels = torch.from_numpy(np.random.random((hr_img.size(0),1)) * 0.05).float()
				# ones = torch.from_numpy(np.ones((hr_img.size(0),1))).float()
				ones = torch.ones(hr_img.size(0), 1).float()
				# hr_labels = torch.ones(hr_img.size(0), 1).float()
				sr_labels = torch.zeros(hr_img.size(0), 1).float()
                
				train_start_time = time.perf_counter()
                
				if torch.cuda.is_available() and opt.cuda:
					hr_img = hr_img.cuda()
					lr_img = lr_img.cuda()
					hr_labels = hr_labels.cuda()
					sr_labels = sr_labels.cuda()
					ones = ones.cuda()
				sr_img = generator_net(lr_img)

				generator_net.zero_grad()
				discriminator_net.zero_grad()
				discriminator_output=discriminator_net(sr_img)
                
				generator_start_time = time.perf_counter()
                

				#===================== train generator =====================
				adversarial_loss = adversarial_criterion(discriminator_output, ones)
				perceptual_loss = content_criterion(feature_extractor(sr_img), feature_extractor(hr_img))
				content_loss = content_criterion(sr_img, hr_img)


				generator_loss =  0.006*perceptual_loss + 1e-3*adversarial_loss  + content_loss 
				

				generator_loss.backward()
				generator_optimizer.step()
                
				discriminator_start_time = time.perf_counter()

				#===================== train discriminator =====================
				discriminator_loss = (adversarial_criterion(discriminator_net(hr_img), hr_labels) + adversarial_criterion(discriminator_net(sr_img.detach()), sr_labels))/2
				
				discriminator_loss.backward()
				discriminator_optimizer.step()
                
				train_end_time = time.perf_counter()

				generator_running_loss += generator_loss.item() * hr_img.size(0)
				discriminator_running_loss += discriminator_loss.item() * hr_img.size(0)
                
                
				train_time+=train_end_time-train_start_time
				compute_time+=generator_start_time-train_start_time
				generator_time+=discriminator_start_time-generator_start_time
				discriminator_time=train_end_time-discriminator_start_time

			total_time=time.perf_counter()-total_time_start
			total_time_list.append(total_time)
			train_time_list.append(train_time)
			compute_time_list.append(compute_time)
			generator_time_list.append(generator_time)
			discriminator_time_list.append(discriminator_time)
            
			torch.save(generator_net.state_dict(), weights_dir+ 'G_epoch_%d.pth' % (epoch))
			generator_losses.append((epoch,generator_running_loss/len(train_set)))
			discriminator_losses.append((epoch,discriminator_running_loss/len(train_set)))
			
 
			if epoch  ==opt.epochs:
				
				with torch.no_grad():
					cur_epoch_dir = imgout_dir+str(epoch)+'/'
					os.makedirs(cur_epoch_dir, exist_ok=True)
					generator_net.eval()
					discriminator_net.eval()
					valid_bar = tqdm(validloader)
					img_count = 0
					psnr_avg = 0.0
					psnr = 0.0
					for hr_img, lr_img in valid_bar:
						valid_bar.set_description('Img: %i   PSNR: %f' % (img_count ,psnr))
						if torch.cuda.is_available():
							lr_img = lr_img.cuda()
							hr_img = hr_img.cuda()
						sr_tensor = generator_net(lr_img)
						mse = torch.mean((hr_img-sr_tensor)**2)
						psnr = 10* (torch.log10(1/mse) + np.log10(4))
						psnr_avg += psnr
						img_count +=1
						sr_img = sr_transform(sr_tensor[0].data.cpu())
						lr_img = lr_transform(lr_img[0].cpu())
						sr_img.save(cur_epoch_dir+'sr_' + str(img_count)+'.png')
						lr_img.save(cur_epoch_dir+'lr_'+str(img_count)+'.png')


					psnr_avg /= img_count
					PSNR_valid.append((epoch, psnr_avg.cpu()))

				check_point = {'generator': generator_net.state_dict(), 'generator_optimizer': generator_optimizer.state_dict(),
				'discriminator': discriminator_net.state_dict(), 'discriminator_optimizer': discriminator_optimizer.state_dict(),
				'discriminator_losses': discriminator_losses, 'generator_losses': generator_losses ,'PSNR_valid': PSNR_valid}
				torch.save(check_point, check_points_dir + 'check_point_epoch_%d.pth' % (epoch))	
				np.savetxt(opt.out_dir + "generator_losses", generator_losses, fmt='%i,%f')
				np.savetxt(opt.out_dir + "discriminator_losses", discriminator_losses, fmt='%i, %f')
				np.savetxt(opt.out_dir + "PSNR", PSNR_valid, fmt='%i, %f')

		gpu_num = torch.cuda.device_count()
		print(f'using {gpu_num} GPUs:')
		print(f'batch size = {opt.bs}')
		print(f'learning rate = {opt.lr}')
		print(f'total_time:{total_time_list}')
		print(f'train_time:{train_time_list}')
		print(f'compute_time:{compute_time_list}')
		print(f'generator_time:{generator_time_list}')
		print(f'discriminator_time:{discriminator_time_list}')
