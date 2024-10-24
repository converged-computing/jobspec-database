#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options
import itertools
# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init

args = Options().parse()
n_class = args.n_class

torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

data_path = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################
train = args.train
test = args.test
graphcam = args.graphcam
print("train:", train, "test:", test, "graphcam:", graphcam)

##### Load datasets
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

if train:
    ids_train = open(args.train_set).readlines()
    dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=1,
                                                   collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    total_train_num = len(dataloader_train) * batch_size

ids_val = open(args.val_set).readlines()
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=1,
                                             collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##### creating models #############
print("creating models......")

num_epochs = args.num_epochs
learning_rate = args.lr

model = Classifier(n_class)
model = nn.DataParallel(model)
if args.resume:
    print('load model {}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)  # best:5e-4, 4e-3
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100],
                                                 gamma=0.1)  # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

criterion = nn.CrossEntropyLoss()

if not test:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(n_class)
evaluator = Evaluator(n_class)

best_pred = 0.0
for epoch in range(num_epochs):
    # optimizer.zero_grad()
    model.train()
    train_loss = 0.
    total = 0.

    current_lr = optimizer.param_groups[0]['lr']
    print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch + 1, current_lr, best_pred))

    if train:
        for i_batch, sample_batched in enumerate(dataloader_train):
            # scheduler(optimizer, i_batch, epoch, best_pred)
            preds, labels, loss, out = trainer.train(sample_batched, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss
            total += len(labels)

            trainer.metrics.update(labels, preds)
            # trainer.plot_cm()

            if (i_batch + 1) % args.log_interval_local == 0:
                print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (
                total, total_train_num, train_loss / total, trainer.get_scores()))
                trainer.plot_cm()

    if not test:
        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (
        total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
        trainer.plot_cm()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            total = 0.
            batch_idx = 0
            slide_labels = []
            slide_preds = []
            slide_probs = []
            for i_batch, sample_batched in enumerate(dataloader_val):
                # pred, label, _ = evaluator.eval_test(sample_batched, model)

                preds, labels, _, out = evaluator.eval_test(sample_batched, model, graphcam)

                slide_labels.append(labels.cpu().detach().numpy().tolist())
                slide_preds.append(preds.cpu().detach().numpy().tolist())
                slide_probs.append(out.cpu().detach().numpy().tolist())

                total += len(labels)

                evaluator.metrics.update(labels, preds)

                if (i_batch + 1) % args.log_interval_local == 0:
                    print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                    evaluator.plot_cm()

            slide_probs = np.vstack((slide_probs))
            slide_labels = list(itertools.chain(*slide_labels))
            slide_probs = list(itertools.chain(*slide_probs))
            slide_preds = list(itertools.chain(*slide_preds))
            slide_probs = np.reshape(slide_probs, (len(slide_labels), 3))

            #auc = roc_auc_score(slide_labels, slide_probs[:, 1].reshape(-1, 1), average="macro")
            # fscore = f1_score(slide_labels, np.clip(slide_preds, 0, 1))
            # recall = recall_score(slide_labels, np.clip(slide_preds, 0, 1))
            # precision = precision_score(slide_labels, np.clip(slide_preds, 0, 1))

            auc = roc_auc_score(slide_labels, slide_probs, average="macro", multi_class='ovr')
            #
            fscore = f1_score(slide_labels, slide_preds , average="macro")
            print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
            print('[%d/%d] val AUC: %.3f' % (total_val_num, total_val_num, auc))
            print('[%d/%d] val fscore: %.3f' % (total_val_num, total_val_num, fscore))
            # print('[%d/%d] val recall: %.3f' % (total_val_num, total_val_num, recall))
            # print('[%d/%d] val precision: %.3f' % (total_val_num, total_val_num, precision))

            evaluator.plot_cm()

            # torch.cuda.empty_cache()

            val_acc = evaluator.get_scores()
            if val_acc > best_pred:
                best_pred = val_acc
                if not test:
                    print("saving model...")
                    torch.save(model.state_dict(), model_path + task_name + ".pth")

            log = ""
            log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}'.format(epoch + 1, num_epochs,
                                                                                        trainer.get_scores(),
                                                                                        evaluator.get_scores()) + "\n"

            log += "================================\n"
            print(log)
            if test: break

            f_log.write(log)
            f_log.flush()

            writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()},
                               epoch + 1)

    trainer.reset_metrics()
    evaluator.reset_metrics()

if not test: f_log.close()