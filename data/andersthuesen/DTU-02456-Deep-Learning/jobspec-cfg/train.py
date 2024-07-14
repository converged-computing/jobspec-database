#!/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/subseg")

import matplotlib.pyplot as plt

from models import SubtitleSegmentation, BaselineModel
from datasets import SubtitleSegmentationDataset
from transforms import Rescale, ToTensor

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = SubtitleSegmentationDataset("data", transform=transforms.Compose([Rescale((360, 640)), ToTensor()]))
	test_dataset = SubtitleSegmentationDataset("test-dataset", transform=transforms.Compose([Rescale((360, 640)), ToTensor()]))


	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size

	train_dataset, val_dataset  = random_split(dataset, [train_size, val_size])
	
	batch_size = 8
	num_workers = 8

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers)
	test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers)

	#model = BaselineModel(in_channels=3, height=360, width=640).to(device)
	#model.load_state_dict(torch.load("baseline-model-batch_size-8.num_epochs-3-with-has-subtitles.torch", map_location=device), strict=False)

	model = SubtitleSegmentation(in_channels=3, height=360, width=640).to(device)
	#model.load_state_dict(torch.load("model-600_epochs-has_subtitle.torch", map_location=device), strict=False)

	freeze_weights = False
	if freeze_weights:
		for param in model.parameters():
			param.requires_grad = False

		model.has_subtitle_conv.weight.requires_grad = True
		model.has_subtitle_conv.requires_grad = True
		#model.fc[0].weight.requires_grad = True
		#model.fc[0].bias.requires_grad = True


	model = nn.DataParallel(model)

	criterion = nn.BCEWithLogitsLoss()
	# For SubSeg model
	optimizer = optim.SGD(model.parameters(), lr=1e-2) #, momentum=0.5) #lr=1e-2, momentum=0.5)

	# For baseline model
	#optimizer = optim.SGD(model.parameters(), lr=1e-3) #, momentum=0.5) #lr=1e-2, momentum=0.5)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

	num_epochs = 600

	step = 1
	for epoch in range(num_epochs):
		# Training
		train_dataloader_tqdm = tqdm(train_dataloader)
		for batch in train_dataloader_tqdm:
			step += 1
			model.train()
			optimizer.zero_grad()
			image = batch["image"].float().to(device)
			mask = batch["mask"].float().to(device)
			has_subtitle = batch["has_subtitle"].float().to(device)

			pred_mask, pred_has_subtitle = model(image)
			
			loss = criterion(pred_mask, mask)
			# For 2nd part of training
			#loss = criterion(pred_has_subtitle, has_subtitle)

			train_dataloader_tqdm.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
			writer.add_scalar("Loss/train", loss.item(), step)
			
			loss.backward()
			optimizer.step()

		# Validation
		model.eval()
		val_losses = []
		for batch in val_dataloader:
			image = batch["image"].float().to(device)
			mask = batch["mask"].float().to(device)
			has_subtitle = batch["has_subtitle"].float().to(device)
			pred_mask, pred_has_subtitle = model(image)
			sub_loss = criterion(pred_has_subtitle, has_subtitle)
			val_loss = loss.item()
			val_losses.append(val_loss)

		val_loss = np.mean(val_losses)
		writer.add_scalar("Loss/val", val_loss, step)

	# Testing
	print("Testing model")
	model.eval()
	test_loss = []
	has_subtitle_accuracy = []
	mask_accuracy = []
	mask_f1 = []
	mask_precision = []
	mask_recall = []
	
	test_dataloader_tqdm = tqdm(test_dataloader)
	for batch in test_dataloader_tqdm:
		image = batch["image"].float().to(device)
		mask = batch["mask"].float().to(device)
		has_subtitle = batch["has_subtitle"].float().to(device)
		pred_mask, pred_has_subtitle = model(image)
		has_subtitle_accuracy.append(1 if (pred_has_subtitle.item() > 0.5) == has_subtitle else 0)

		if has_subtitle.item():
			true_positive = torch.sum(pred_mask[mask == True] >= 0.5).float()
			false_positive = torch.sum(pred_mask[mask == False] >= 0.5).float()
			false_negative = torch.sum(pred_mask[mask == True] < 0.5).float()
			
			precision = true_positive / (true_positive + false_positive)
			recall = true_positive / (true_positive + false_negative)
			f1 = 1 / (0.5 * (1 / recall + 1 / precision))

			mask_precision.append(precision.item())
			mask_recall.append(recall.item())
			mask_f1.append(f1.item())

	print(f"Has subtitle accuracy: {np.mean(has_subtitle_accuracy)}")
	print(f"Mask precision: {np.mean(mask_precision)}")
	print(f"Mask recall: {np.mean(mask_recall)}")
	print(f"Mask f1: {np.mean(mask_f1)}")

	# Saving
	name = f"subseg-batch_size-{batch_size}.num_epochs-{num_epochs}.torch"
	print(f"Saving: {name}")
	torch.save(model.module.state_dict(), name)

