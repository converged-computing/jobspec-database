import sys, os
import os.path as osp
import numpy as np
sys.path.append('/scratch/midway3/cdonnat/CCA')

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.nn import GAE, VGAE, APPNP
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import torch_geometric.transforms as T
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
     to_dense_adj
)
#### MISSING: RANDOMISATION WITH RESPECT TO TRAINING NODES
#### MISSING: AUC for validation set --- jhow should we choose the best params otherwise?

from models.basicVGNAE import *
from models.DeepVGAEX import *
from models.baseline_models import *
from aug import *
from models.cca import CCA_SSG
from models.ica_gnn import GraphICA, iVGAE, random_permutation
from train_utils import *

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--model', type=str, default='VGAEX')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--normalize', type=parse_boolean, default=True)
parser.add_argument('--non_linear', type=str, default='relu')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_train_per_class', type=float, default=20)
parser.add_argument('--max_epoch_eval', type=int, default=2000)
parser.add_argument('--result_file', type=str, default="/results/n_experiments_")
args = parser.parse_args()

MAX_EPOCH_EVAL = args.max_epoch_eval

file_path = os.getcwd() + str(args.result_file) + args.model + '_' + args.dataset +'_normalize' +\
 str(args.normalize) + '_nonlinear' + str(args.non_linear) + '_lr' + str(args.lr) + '.csv'

print(file_path)
path = '/scratch/midway3/cdonnat/CCA/data'
if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root=path  + '/Planetoid', name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
if args.dataset in ['CS', 'Physics']:
    dataset = Coauthor(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    transform_nodes = RandomNodeSplit(split = 'test_rest',
                                      num_train_per_class = args.num_train_per_class,
                                      num_val = 500)
    data = transform_nodes(data)


if args.dataset in ['Computers', 'Photo']:
    dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    transform_nodes = RandomNodeSplit(split = 'test_rest',
                                          num_train_per_class = args.num_train_per_class,
                                          num_val = 500)
    data = transform_nodes(data)

if args.non_linear == 'relu':
    activation  = torch.nn.ReLU()
else:
    activation = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'VGNAE':
    alphas = [0.]
elif args.model == 'VGAEX':
    alphas = [1, 10, 50, 100, 500]
else:
    alphas = [0]

n_layers = [1,2, 3, 4, 5]

results =[]
if args.model in ['VGNAE', 'VGAEX']:
    for training_rate in [0.85, 0.1, 0.2, 0.4, 0.6, 0.8]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        transform_nodes = RandomNodeSplit(split = 'test_rest',
                                          num_train_per_class = args.num_train_per_class,
                                          num_val = 500)
        train_data, val_data, test_data = transform(data)
        rand_data = transform_nodes(data)
        for alpha in alphas:
            for n_lay in n_layers:
                #for out_channels in [ 32]:
                for out_channels in [64, 32, 128, 256, 512]:
                    if args.model == 'VGNAE':
                        model = DeepVGAE(data.x.size()[1], out_channels, out_channels,
                                         n_layers=n_lay, normalize=args.normalize,
                                         activation=args.non_linear).to(device)
                        y_randoms = None
                    else:
                        model = DeepVGAEX(data.x.size()[1], out_channels, out_channels,
                                         n_layers=n_lay, normalize=args.normalize,
                                         h_dims_reconstructiony = [out_channels, out_channels],
                                         y_dim=alpha, dropout=0.5,
                                         lambda_y =0.5/alpha, activation=args.non_linear).to(device)
                        w = torch.randn(size= (data.num_features, alpha)).float()
                        y_randoms = torch.mm(data.x, w)
                    # move to GPU (if available)
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                    last_ac = 0
                    trigger_times = 0
                    best_epoch_model = 0
                    temp_res = []


                    for epoch in range(1, args.epochs):
                        model.train()
                        optimizer.zero_grad()
                        loss = model.loss(train_data.x, y=y_randoms,
                                          pos_edge_index=train_data.pos_edge_label_index,
                                          neg_edge_index=train_data.neg_edge_label_index,
                                          train_mask=train_data.train_mask)
                        loss.backward()
                        optimizer.step()
                        #if epoch == 50: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2)
                        #if epoch == 100: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/4)
                        #if epoch == 150: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/8)
                        if epoch % 5 == 0:
                            loss = float(loss)
                            train_auc, train_ap = model.single_test(data.x,
                                                train_data.pos_edge_label_index,
                                                train_data.pos_edge_label_index,
                                                train_data.neg_edge_label_index)
                            roc_auc, ap = model.single_test(data.x,
                                                train_data.pos_edge_label_index,
                                                test_data.pos_edge_label_index,
                                                test_data.neg_edge_label_index)
                            temp_res += [[epoch, train_auc, train_ap, roc_auc, ap]]
                            print('Epoch: {:03d}, LOSS: {:.4f}, AUC(train): {:.4f}, AP(train): {:.4f}  AUC(test): {:.4f}, AP(test): {:.4f}'.format(epoch, loss, train_auc, train_ap, roc_auc, ap))

                            #### Add early stopping to prevent overfitting
                            out  = model.single_test(data.x,
                                                train_data.pos_edge_label_index,
                                                val_data.pos_edge_label_index,
                                                val_data.neg_edge_label_index)
                            current_ac = np.mean(out)

                            if current_ac <= last_ac:
                                trigger_times += 1
                                #print('Trigger Times:', trigger_times)
                                #if triggertimes == 2: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2)
                                #if triggertimes == 6: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/4)
                                #if triggertimes == 10: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/8)
                                if trigger_times >= args.patience:
                                    #print('Early stopping!\nStart to test process.')
                                    break
                            else:
                                #print('trigger times: 0')
                                trigger_times = 0
                                last_ac = current_ac
                                best_epoch_model = epoch

                    print(temp_res)
                    print(temp_res[best_epoch_model//5-1])

                    train_auc, train_ap, roc_auc, ap = temp_res[best_epoch_model//5-1][1], temp_res[best_epoch_model//5-1][2], temp_res[best_epoch_model//5-1][3], temp_res[best_epoch_model//5-1][4]
                    embeds = model.encode(train_data.x, edge_index=train_data.pos_edge_label_index)
                    _, nodes_res, best_epoch = node_prediction(embeds.detach(),
                                                   dataset.num_classes, data.y,
                                                   rand_data.train_mask,
                                                   rand_data.test_mask,
                                                   rand_data.val_mask,
                                                   lr=0.001, wd=1e-4,
                                                   patience = 20,
                                                   max_epochs=MAX_EPOCH_EVAL)
                    acc_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3]

                    _, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                                                   dataset.num_classes, data.y,
                                                   data.train_mask,
                                                   data.test_mask,
                                                   data.val_mask,
                                                   lr=0.005, wd=1e-4,
                                                   patience = 20,
                                                   max_epochs=MAX_EPOCH_EVAL)
                    acc_train_default, acc_default = nodes_res_default[best_epoch][2], nodes_res_default[best_epoch][3]
                    results += [[args.model, args.dataset, str(args.non_linear),
                                 args.normalize, args.lr, out_channels,
                                 training_rate, val_ratio, test_ratio, n_lay, alpha, train_auc, train_ap,
                                 roc_auc, ap, acc_train, acc, acc_train_default, acc_default, epoch, 0, 0]]
                    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity',
                                                              'normalize',  'lr', 'channels',
                                                              'train_rate','val_ratio', 'test_ratio',
                                                              'n_layers', 'lambd',  'train_auc', 'train_ap',
                                                              'test_auc', 'test_ap', 'accuracy_train',
                                                              'accuracy_test', 'accuracy_train_default',
                                                              'accuracy_test_default', 'epoch',
                                                              'drop_edge_rate', 'drop_feat_rate'])
                    res1.to_csv(file_path, index=False)
elif args.model == 'CCA':
    #### Test the CCA approach
    print("CCA_SSG")
    N = data.num_nodes

    ##### Train the CCA model
    for training_rate in [0.85, 0.1, 0.2, 0.4, 0.6, 0.8]:
    #for training_rate in [0.1]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        transform_nodes = RandomNodeSplit(split = 'test_rest',
                                          num_train_per_class = args.num_train_per_class,
                                          num_val = 500)
        train_data, val_data, test_data = transform(data)
        rand_data = transform_nodes(data)
        for n in n_layers:
            for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
                for channels in [32, 64, 128, 256, 512]:
                    for drop_rate_edge in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
                        model = CCA_SSG(data.num_features, channels, channels, n,
                                        activation=args.non_linear, slope=.1,
                                        device=device,
                                        normalize=args.normalize, use_mlp=False)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                                     weight_decay=1e-4)
                        #for epoch in range(300):
                        for epoch in range(300):
                            model.train()
                            optimizer.zero_grad()
                            new_data1 = random_aug(data, drop_rate_edge, drop_rate_edge)
                            new_data2 = random_aug(data, drop_rate_edge, drop_rate_edge)

                            z1, z2 = model(new_data1, new_data2)

                            c = torch.mm(z1.T, z2)
                            c1 =torch.mm(z1.T, z1)
                            c2 = torch.mm(z2.T, z2)

                            c = c / N
                            c1 = c1 / N
                            c2 = c2 / N

                            loss_inv = -torch.diagonal(c).sum()
                            iden = torch.tensor(np.eye(c.shape[0]))
                            loss_dec1 = (iden - c1).pow(2).sum()
                            loss_dec2 = (iden - c2).pow(2).sum()

                            loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

                            loss.backward()
                            optimizer.step()

                            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

                        # In[ ]:


                        print("=== Evaluation ===")
                        embeds = model.get_embedding(data)
                        _, res, best_epoch = edge_prediction(embeds.detach(), embeds.shape[1],
                                                 train_data, test_data, val_data,
                                                 lr=0.001, wd=1e-4,
                                                 patience = 20,
                                                 max_epochs=MAX_EPOCH_EVAL)
                        val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[best_epoch][1], res[best_epoch][2], res[best_epoch][3], res[best_epoch][4], res[best_epoch][5], res[best_epoch][6]
                        _, nodes_res, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       rand_data.train_mask, rand_data.test_mask,
                                                       rand_data.val_mask,
                                                       lr=0.005, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)

                        acc_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3]

                        _, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       data.train_mask, data.test_mask,
                                                       data.val_mask,
                                                       lr=0.005, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)
                        print(nodes_res_default)
                        print("here")
                        print(nodes_res_default[best_epoch])
                        acc_train_default, acc_default = nodes_res_default[best_epoch][2], nodes_res_default[best_epoch][3]

                        results += [['CCA', args.dataset, str(args.non_linear),
                                     args.normalize, args.lr, channels,
                                     training_rate, val_ratio, test_ratio,
                                     n, lambd, train_roc, train_ap,
                                     test_roc, test_ap, acc_train, acc,
                                     acc_train_default, acc_default, epoch,
                                     drop_rate_edge, drop_rate_edge]]
                        print(['CCA', args.dataset, str(args.non_linear),
                               args.normalize, args.lr, channels,
                               training_rate, val_ratio, test_ratio,
                               n, lambd, train_roc, train_ap,
                               test_roc, test_ap, acc_train, acc,
                               acc_train_default, acc_default, epoch, drop_rate_edge, drop_rate_edge])

                        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity',
                                                              'normalize',  'lr', 'channels',
                                                              'train_rate','val_ratio', 'test_ratio',
                                                              'n_layers', 'lambd',  'train_auc', 'train_ap',
                                                              'test_auc', 'test_ap', 'accuracy_train',
                                                              'accuracy_test', 'accuracy_train_default',
                                                              'accuracy_test_default', 'epoch',
                                                              'drop_edge_rate', 'drop_feat_rate'])
                        res1.to_csv(file_path, index=False)

elif args.model == 'ICA':
    print("ICA 1")
    criterion = torch.nn.CrossEntropyLoss()
    ##### Train the ICA model
    for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        transform_nodes = RandomNodeSplit(split = 'test_rest',
                                          num_train_per_class = args.num_train_per_class,
                                          num_val = 500)
        train_data, val_data, test_data = transform(data)
        rand_data = transform_nodes(data)
        for n in n_layers:
            #for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
                #for channels in [32]:
                for channels in [32, 64, 128, 256, 512]:
                        model = GraphICA(data.num_features,
                                        channels, channels, use_mlp = False,
                                        use_graph=True,
                                        regularize=True)
                        z = model(data)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
                        for epoch in range(args.epochs):
                            model.train()
                            optimizer.zero_grad()
                            new_data1 = random_permutation(data)

                            z = model(data)
                            z_fake = model(new_data1)
                            loss_pos = criterion(z, torch.ones(z.shape[0]).long())
                            loss_neg = criterion(z_fake, torch.zeros(z_fake.shape[0]).long())
                            #neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

                            loss = loss_pos + loss_neg

                            loss.backward()
                            optimizer.step()
                            preds  = torch.argmax(z, 1)
                            preds0  = torch.argmax(z_fake, 1)
                            acc = 0.5*((preds==1).sum()/preds.shape[0]).item() +  0.5* ((preds0==0).sum()/preds0.shape[0]).item()
                            print('Epoch={:03d}, loss={:.4f}, acc={:.4f}'.format(epoch, loss.item(), acc))


                        print("=== Evaluation ===")
                        embeds = model.get_embedding(data)
                        _, res, best_epoch = edge_prediction(embeds.detach(),
                                                             embeds.shape[1],
                                                             train_data,
                                                             test_data,
                                                             val_data,
                                                             lr=0.001, wd=1e-4,
                                                             patience = 20,
                                                             max_epochs=MAX_EPOCH_EVAL)
                        val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[best_epoch][1], res[best_epoch][2], res[best_epoch][3], res[best_epoch][4], res[best_epoch][5], res[best_epoch][6]
                        _, nodes_res, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       rand_data.train_mask, rand_data.test_mask,
                                                       rand_data.val_mask,
                                                       lr=0.005, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)

                        acc_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3]

                        _, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       data.train_mask, data.test_mask,
                                                       data.val_mask,
                                                       lr=0.005, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)
                        acc_train_default, acc_default = nodes_res_default[best_epoch][2], nodes_res_default[best_epoch][3]

                        results += [['ICA', args.dataset, str(args.non_linear),
                                     args.normalize, args.lr, channels,
                                     training_rate, val_ratio, test_ratio,
                                     n, None, train_roc, train_ap,
                                     test_roc, test_ap, acc_train, acc,
                                     acc_train_default, acc_default, epoch, 0, 0]]
                        print(['ICA', args.dataset, str(args.non_linear),
                               args.normalize, args.lr, channels,
                               training_rate, val_ratio, test_ratio,
                               n, None, train_roc, train_ap,
                               test_roc, test_ap, acc_train, acc,
                               acc_train_default, acc_default, epoch, 0, 0])

                        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity',
                                                              'normalize',  'lr', 'channels',
                                                              'train_rate','val_ratio', 'test_ratio',
                                                              'n_layers', 'lambd',  'train_auc', 'train_ap',
                                                              'test_auc', 'test_ap', 'accuracy_train',
                                                              'accuracy_test', 'accuracy_train_default',
                                                              'accuracy_test_default', 'epoch',
                                                              'drop_edge_rate', 'drop_feat_rate'])
                        res1.to_csv(file_path, index=False)

else:
    ##### Train the non linear ICA model
    print("ICA non linear")
    for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
    #for training_rate in [0.1]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        transform_nodes = RandomNodeSplit(split = 'test_rest',
                                          num_train_per_class = args.num_train_per_class,
                                          num_val = 500)
        train_data, val_data, test_data = transform(data)
        rand_data = transform_nodes(data)
        for n in n_layers:
            #for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
                #for channels in [32]:
                for channels in [32, 64, 128, 256, 512]:
                        aux_dim = dataset.num_classes
                        model = iVGAE(latent_dim=channels, data_dim=data.num_features,
                                      aux_dim=dataset.num_classes, activation=args.non_linear,
                                      device=device, n_layers=2, hidden_dim = channels)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, verbose=True)

                        # training loop

                        print("Training..")
                        model.train()
                        trigger_times = 0
                        best_elbo_train  = 1e4
                        x = data.x.to(device)
                        u = torch.nn.functional.one_hot(data.y,
                                    num_classes=dataset.num_classes).float().to(device)

                        for epoch in range(args.epochs):
                            elbo_train = 0

                            optimizer.zero_grad()
                            elbo, z_est = model.elbo(x, u, data.edge_index)
                            elbo.mul(-1).backward()
                            optimizer.step()
                            elbo_train += -elbo.item()
                            #elbo_train /= len(train_loader)
                            #scheduler.step(elbo_train)
                            print('epoch {}/{} \tloss: {}'.format(epoch, args.epochs, elbo_train))
                            if elbo_train >= elbo.item():
                                trigger_times += 1
                                #print('Trigger Times:', trigger_times)
                                #if triggertimes == 2: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2)
                                #if triggertimes == 6: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/4)
                                #if triggertimes == 10: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/8)
                                if trigger_times >= args.patience:
                                    #print('Early stopping!\nStart to test process.')
                                    break
                            else:
                                #print('trigger times: 0')
                                trigger_times = 0
                                best_elbo_train = elbo.item()
                                best_epoch_model = epoch
                        # save model checkpoint after training
                        print("=== Evaluation ===")
                        data = dataset[0]
                        Xt, Ut = data.x, u
                        decoder_params, encoder_params, z, prior_params = model(Xt, Ut, data.edge_index)
                        params = {'decoder': decoder_params, 'encoder': encoder_params, 'prior': prior_params}
                        embeds = params['encoder'][0].detach()
                        _, res, best_epoch = edge_prediction(embeds.detach(),
                                                             embeds.shape[1],
                                                             train_data,
                                                             test_data,
                                                             val_data,
                                                             lr=0.001, wd=1e-4,
                                                             patience = 20,
                                                             max_epochs=MAX_EPOCH_EVAL)
                        val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[best_epoch][1], res[best_epoch][2], res[best_epoch][3], res[best_epoch][4], res[best_epoch][5], res[best_epoch][6]
                        _, nodes_res, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       rand_data.train_mask, rand_data.test_mask,
                                                       rand_data.val_mask,
                                                       lr=0.005, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)

                        acc_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3]

                        _, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       data.train_mask, data.test_mask,
                                                       data.val_mask,
                                                       lr=0.005, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)
                        acc_train_default, acc_default = nodes_res_default[best_epoch][2], nodes_res_default[best_epoch][3]

                        results += [['ICA non linear', args.dataset, str(args.non_linear),
                                     args.normalize, args.lr, channels,
                                     training_rate, val_ratio, test_ratio,
                                     n, None, train_roc, train_ap,
                                     test_roc, test_ap, acc_train, acc,
                                     acc_train_default, acc_default, epoch, 0, 0]]
                        print(['ICA non linear', args.dataset, str(args.non_linear),
                               args.normalize, args.lr, channels,
                               training_rate, val_ratio, test_ratio,
                               n, None, train_roc, train_ap,
                               test_roc, test_ap, acc_train, acc,
                               acc_train_default, acc_default, epoch, 0, 0])
                        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity',
                                                              'normalize',  'lr', 'channels',
                                                              'train_rate','val_ratio', 'test_ratio',
                                                              'n_layers', 'lambd',  'train_auc', 'train_ap',
                                                              'test_auc', 'test_ap', 'accuracy_train',
                                                              'accuracy_test', 'accuracy_train_default',
                                                              'accuracy_test_default', 'epoch',
                                                              'drop_edge_rate', 'drop_feat_rate'])
                        res1.to_csv(file_path, index=False)
