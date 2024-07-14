# -*- coding: utf-8 -*-
"""# Import dependencies"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import ToLAB, ReshapeChannelFirst, ToTensor
from data import ImageDataset
from model import models, Discriminator

from torchvision import transforms
from torch.utils.data import DataLoader


def train(
        images_path,
        device,
        model,
        load,
        batch_size,
        shuffle,
        log_output,
        loss,
        num_workers,
        num_epochs,
        learning_rate,
        save_best,
        save,
        save_frequency  # Save after 10 epochs
):

    phases = ["validation", "training"]

    images_dataset = {
        x: ImageDataset(
            os.path.join(images_path, x),
            transform=transforms.Compose(
                [ToLAB(), ReshapeChannelFirst(),
                 ToTensor()]))
        for x in phases
    }

    images = {
        x: DataLoader(
            images_dataset[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        for x in phases
    }

    # Make instance of model
    colorizer = models[model]()

    if load != None:
        print("Loading model from: {}".format(load))
        try:
            colorizer.load_state_dict(torch.load(load))
        except:
            print("Could not load model.")
            pass

    num_cuda_devices = torch.cuda.device_count()
    print("Training on {} cuda devices.".format(num_cuda_devices))
    if num_cuda_devices > 1:
        colorizer = nn.DataParallel(colorizer)

    colorizer = colorizer.to(device)

    discriminator = None
    criterion = None
    optimizer = None
    d_optimizer = None

    if loss == "gan":
        discriminator = Discriminator()
        if load != None:
            try:
                discriminator.load_state_dict(torch.load("{}.discriminator".format(load)))
            except:
                print("Could not load discriminator model.")
                pass
        discriminator = discriminator.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(colorizer.parameters(), lr=learning_rate)
        d_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate)
    else:
        criterion = nn.MSELoss()
        optimizer = optim.SGD(colorizer.parameters(), lr=learning_rate)

    best_validation_loss = math.inf

    if log_output != None:
        with open(log_output, "w+") as log_file:
            log_file.write("train,training_loss,validation_loss\n")
    
    try:
        # Train our model
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch + 1, num_epochs))

            running_train_loss = 0.0
            running_validation_loss = 0.0

            for phase in phases:
                if phase == "training":
                    colorizer.train()
                elif phase == "validation":
                    colorizer.eval()

                for L, AB in images[phase]:
                    L = L.to(device)
                    AB = AB.to(device)
                    AB_pred = colorizer(L)

                    if loss == "gan":
                        LAB = torch.cat((L, AB), dim=1)
                        LAB_gen = torch.cat((L, AB_pred), dim=1)
                        valid = torch.ones((L.shape[0], 1)).to(device)
                        invalid = torch.zeros((L.shape[0], 1)).to(device)

                        if phase == "training":

                            optimizer.zero_grad()
                            g_loss = criterion(discriminator(LAB_gen), valid)
                            g_loss.backward()
                            optimizer.step()

                            d_optimizer.zero_grad()
                            real_loss = criterion(discriminator(LAB), valid)
                            fake_loss = criterion(discriminator(LAB_gen.detach()), invalid)
                            d_loss = (real_loss + fake_loss) / 2
                            d_loss.backward()
                            d_optimizer.step()

                            running_train_loss += g_loss.item() * L.size(0)

                        elif phase == "validation":
                            g_loss = criterion(discriminator(LAB_gen), valid)
                            running_validation_loss += g_loss.item() * L.size(0)

                    else:
                        _loss = criterion(AB_pred, AB)
                        if phase == "training":
                            optimizer.zero_grad()
                            _loss.backward()
                            optimizer.step()
                            running_train_loss += _loss.item() * L.size(0)
                        elif phase == "validation":
                            running_validation_loss += _loss.item() * L.size(0)


            epoch_train_loss = running_train_loss / len(images_dataset["training"])
            epoch_validation_loss = running_validation_loss / len(images_dataset["validation"])
            print("Training loss: {}".format(epoch_train_loss))
            print("Validation loss: {}".format(epoch_validation_loss))

            if log_output != None:
                with open(log_output, "a+") as log_file:
                    log_file.write("{},{},{}\n".format(epoch + 1, epoch_train_loss, epoch_validation_loss))

            if save_best != None and epoch_validation_loss < best_validation_loss:
                print("Saving best model.")
                best_validation_loss = epoch_validation_loss
                torch.save(colorizer.state_dict(), save_best)
                
                if loss == "gan":
                    torch.save(discriminator.state_dict(), "{}.discriminator".format(save_best))

            if save != None and epoch % save_frequency == 0:
                print("Saving model.")
                torch.save(colorizer.state_dict(), save)

                if loss == "gan":
                    torch.save(discriminator.state_dict(), "{}.discriminator".format(save))

            print("-" * 30)
    except KeyboardInterrupt:
            print("Stopping training.")
            pass

    if save != None:
        print("Saving final model.")
        torch.save(colorizer.state_dict(), save)

        if loss == "gan":
            torch.save(discriminator.state_dict(), "{}.discriminator".format(save))
