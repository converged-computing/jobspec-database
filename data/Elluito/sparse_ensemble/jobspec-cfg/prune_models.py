import os
import time

import torch
import re
import argparse
import glob
import torchvision.transforms as transforms
import torchvision
from pathlib import Path
from alternate_models import *
import pandas as pd
from main import prune_function, remove_reparametrization, get_layer_dict, get_datasets, count_parameters
import omegaconf
from similarity_comparison_architecture import features_similarity_comparison_experiments
import numpy as np
from torch.nn.utils import parameters_to_vector
from sparse_ensemble_utils import disable_bn, mask_gradient, sparsity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Level 0
rf_level0_s1 = "trained_models/cifar10/resnet50_cifar10.pth"
name_rf_level0_s1 = "_seed_1_rf_level_0"

rf_level0_s2 = "trained_models/cifar10/resnet50_normal_seed_2_tst_acc_95.65.pth"
name_rf_level0_s2 = "_seed_2_rf_level_0"

# level 1
rf_level1_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_1_95.26.pth"
name_rf_level1_s1 = "_seed_1_rf_level_1"
rf_level1_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_1_94.93.pth"
name_rf_level1_s2 = "_seed_2_rf_level_1"

# level 2
rf_level2_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
name_rf_level2_s1 = "_seed_1_rf_level_2"

rf_level2_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
name_rf_level2_s2 = "_seed_2_rf_level_2"
# Level 3

rf_level3_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_3_92.38.pth"
name_rf_level3_s1 = "_seed_1_rf_level_3"

rf_level3_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_3_92.25.pth"
name_rf_level3_s2 = "_seed_2_rf_level_3"

# Level 4
rf_level4_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
name_rf_level4_s1 = "_seed_1_rf_level_4"
rf_level4_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
name_rf_level4_s2 = "_seed_2_rf_level_4"

files_names = [name_rf_level1_s1, name_rf_level1_s2, name_rf_level2_s1, name_rf_level2_s2, name_rf_level3_s1,
               name_rf_level3_s2, name_rf_level4_s1, name_rf_level4_s2]
files = [rf_level1_s1, rf_level1_s2, rf_level2_s1, rf_level2_s2, rf_level3_s1, rf_level3_s2, rf_level4_s1, rf_level4_s2]
level = [1, 1, 2, 2, 3, 3, 4, 4]
modelstypes = ["alternative"] * len(level)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_cuda = True if device == "cuda" else False


def record_features_cifar10_model_pruned(architecture="resnet18", seed=1, modeltype="alternative", solution="",
                                         seed_name="_seed_1", rf_level=0, model=None):
    from feature_maps_utils import save_layer_feature_maps_for_batch

    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype,
         "solution": solution,
         # "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": 1,
         "num_workers": 0,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    resnet18_normal = model
    ################################# dataset cifar10 ###########################################################################

    from alternate_models.resnet import ResNet50_rf
    from torchvision.models import resnet18, resnet50
    if cfg.dataset == "cifar10":
        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)

        # cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        # val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
        #                                          num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
    if cfg.dataset == "cifar100":
        if cfg.model_type == "alternative":
            resnet18_normal = ResNet50_rf(num_classes=100, rf_level=rf_level)
            resnet18_normal.load_state_dict(torch.load(cfg.solution)["net"])
        if cfg.model_type == "hub":
            resnet18_normal = resnet50()
            in_features = resnet18_normal.fc.in_features
            resnet18_normal.fc = torch.nn.Linear(in_features, 100)
            temp_dict = torch.load(cfg.solution)["net"]
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            resnet18_normal.load_state_dict(real_dict)
        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\cifar100"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.compose([
            transforms.randomcrop(32, padding=4),
            transforms.randomhorizontalflip(),
            transforms.totensor(),
            transforms.normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])
        transform_test = transforms.compose([
            transforms.totensor(),
            transforms.normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])

        trainset = torchvision.datasets.cifar100(root=data_path, train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.dataloader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        testset = torchvision.datasets.cifar100(root=data_path, train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.dataloader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)

    current_directory = Path().cwd()
    add_nobackup = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        add_nobackup = "/nobackup/sclaam/"

    prefix_custom_test = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = 2000
    resnet18_normal.cuda()
    o = 0
    for x, y in testloader:
        x = x.cuda()
        save_layer_feature_maps_for_batch(resnet18_normal, x, prefix_custom_test, seed_name=seed_name)
        # Path(file_prefix / "layer{}_features{}.npy".format(i, seed_name))

        print("{} batch out of {}".format(o, len(testloader)))
        if o == maximun_samples:
            break
        o += 1


def test_ffcv(net, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0, number_batches=0):
    criterion = nn.CrossEntropyLoss()
    net.cuda()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if count_flops:
        assert batch_flops != 0, "If count_flops is True,batch_flops must be non-zero"

    sparse_flops = 0
    first_time = 1
    sparse_flops_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if count_flops:
                sparse_flops += batch_flops
            test_loss += loss.data.item()
            if torch.all(outputs > 0):
                _, predicted = torch.max(outputs.data, 1)
            else:
                soft_max_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(soft_max_outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # print(correct/total)

            if batch_idx % 100 == 0:
                if verbose == 2:
                    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
            if one_batch:
                if count_flops:
                    return 100. * correct.item() / total, sparse_flops
                else:
                    return 100. * correct.item() / total

            if number_batches > 0:
                if number_batches < batch_idx:
                    return 100. * correct.item() / total

    if verbose == 1 or verbose == 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss / len(testloader), correct, total,
            100. * correct.item() / total))
    # net.cpu()
    if count_flops:
        return 100. * correct.item() / total, sparse_flops
    else:
        return 100. * correct.item() / total


def save_pruned_representations():
    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    train, val, testloader = get_datasets(cfg)

    dense_accuracy_list = []
    pruned_accuracy_list = []

    for i in range(len(files)):
        state_dict_raw = torch.load(files[i])
        dense_accuracy_list.append(state_dict_raw["acc"])
        net = ResNet50_rf(num_classes=10, rf_level=level[i])
        net.load_state_dict(state_dict_raw["net"])
        prune_function(net, cfg)
        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
        record_features_cifar10_model_pruned("resnet50", seed_name="pruned_{}".format(files_names[i]),
                                             rf_level=level[i], model=net)


def similarity_comparisons():
    for i in range(len(files)):
        for j in range(i, len(files)):
            p_name1 = "pruned_{}".format(files_names[i])
            p_name2 = "pruned_{}".format(files_names[j])
            features_similarity_comparison_experiments(architecture="resnet50",
                                                       modeltype1="alternative",
                                                       modeltype2="alternative",
                                                       name1=p_name1,
                                                       name2=p_name2, filetype1="npy",
                                                       filetype2="npy")
            # filename = "similarity_experiments/{}_{}_V_{}_.txt".format("resnet50", p_name1, p_name2)
            #
            # np.savetxt(filename, similarity_for_networks, delimiter=",")


def gradient_flow_calculation(args):
    from flowandprune.imp_estimator import cal_grad
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, rf_level=args.RF_level)

    gradient_flow_at_init_list = []
    pruned_accuracy_list = []
    files_names = []

    for i, name in enumerate(
            glob.glob("{}/{}_normal_{}_*_level_{}_initial_weights.pth".format(args.folder, args.model, args.dataset,
                                                                              args.RF_level))):
        state_dict_raw = torch.load(name)
        net.load_state_dict(state_dict_raw["net"])
        net.cuda()
        gradient_flow = cal_grad(net, trainloader=train)
        grad_vect = parameters_to_vector(gradient_flow)
        # hg_vect = parameters_to_vector(hg)
        norm_grad = torch.norm(grad_vect)
        gradient_flow_at_init_list.append(norm_grad.detach().cpu().numpy())
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)
    df = pd.DataFrame({"Name": files_names,
                       "Gradient Flow at init": gradient_flow_at_init_list,
                       })
    df.to_csv("GF_init_{}_{}_{}_summary.csv".format(args.model, args.RF_level, args.dataset), index=False)


def logistic_probes_for_model(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": args.solution,
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, rf_level=args.RF_level)

    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    state_dict_raw = torch.load(args.solution, map_location=device)
    net.load_state_dict(state_dict_raw["net"])


def test(net, testloader=None, verbose=0, name="ckpt", save_folder="./checkpoint", args=None):
    # global best_acc, testloader, device, criterion
    net.eval()
    net = net.to(device)
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if batch_idx == 10:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print("Predicted:{}".format(predicted.cpu().numpy()))
            print("Targets:{}".format(targets.cpu().numpy()))

    # Save checkpoint.
    acc = 100. * correct / total
    return acc


def fine_tune_pruned_model_with_mask(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                     testLoader: torch.utils.data.DataLoader,
                                     epochs=1,
                                     initial_flops=0, exclude_layers=[],
                                     fine_tune_exclude_layers=False, fine_tune_non_zero_weights=True,
                                     cfg=None, save_folder="", name=""):
    from main import get_mask
    from train_CIFAR10 import progress_bar

    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.99, weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 60
    # lr = 0.0005   if 60 <= epoch < 90
    # ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    pruned_accuracy = test(pruned_model, use_cuda=True, testloader=testLoader, verbose=0)
    print("Pruned accuracy inside fine tuning:{}".format(pruned_accuracy))
    # grad_clip = 0
    # if cfg.gradient_cliping:
    grad_clip = 0.1
    names, weights = zip(*get_layer_dict(pruned_model))

    mask_dict = get_mask(model=pruned_model)
    for n in exclude_layers:
        if n in list(mask_dict.keys()):
            mask_dict.pop(n)

    total_sparse_FLOPS = initial_flops
    # first_time = 1

    # forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)

    pruned_model.cuda()
    pruned_model.train()
    disable_bn(pruned_model)
    # if not fine_tune_exclude_layers:
    #     disable_exclude_layers(pruned_model, exclude_layers)
    # if not fine_tune_non_zero_weights:
    #     disable_all_except(pruned_model, exclude_layers)
    #
    criterion = nn.CrossEntropyLoss()

    pruned_model.train()
    train_loss = 0
    correct = 0
    total = 0
    best_acc = 0
    pruned_accuracy = test(pruned_model, use_cuda=True, testloader=testLoader, verbose=0)
    print("Pruned after disable_bn :{}".format(pruned_accuracy))
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        #################################
        # Train
        #################################
        for batch_idx, (inputs, targets) in enumerate(dataLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = pruned_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Mask the gradient
            mask_gradient(pruned_model, mask_dict=mask_dict)
            # if grad_clip:
            #     nn.utils.clip_grad_value_(pruned_model.parameters(), grad_clip)

            optimizer.step()
            # lr_scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print("Accuracy: {:2.3f}%, Correct: {} , Total: {}".format(100. * correct / total, correct, total))
            progress_bar(batch_idx, len(dataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        #################################
        #    TEST
        #################################

        print("\nTest Set\n")
        test_loss = 0
        correct = 0
        total = 0
        pruned_model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Save checkpoint.
        acc: float = 100. * correct / total
        print("Total test accuracy: {}".format(acc))
        pruned_accuracy = test(pruned_model, use_cuda=True, testloader=testLoader, verbose=0)
        print("Pruned accuracy with \"test\" function :{}".format(pruned_accuracy))
        if acc > best_acc:
            # print('Saving..')
            state = {
                'net': pruned_model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
                os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
            torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, name, acc))
            best_acc = acc
        scheduler.step()

    return best_acc


def pruning_fine_tuning_experiment(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, rf_level=args.RF_level)

    dense_accuracy_list = []
    fine_tuned_accuracy = []
    folder_name = "{}/pruned/{}".format(args.folder, args.pruning_rate)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    # new_folder = "{}/pruned/{}".format(args.folder, args.pruning_rate)

    #
    # for i, name in enumerate(
    #         glob.glob("{}/{}_normal_{}_*_level_{}_test_acc_*.pth".format(args.folder, args.model, args.dataset,
    #                                                                      f"{args.RF_level}{args.name}"))):

    state_dict_raw = torch.load("{}/{}".format(args.folder, args.solution))
    dense_accuracy_list.append(state_dict_raw["acc"])
    print(state_dict_raw["acc"])
    net.load_state_dict(state_dict_raw["net"])
    prune_function(net, cfg)
    remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
    pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
    print("Pruned accuracy:{}".format(pruned_accuracy))
    file_name = args.solution
    print(file_name)
    if "test_acc" in file_name:
        index_until_test = file_name.index("test_acc")
        base_name = file_name[:index_until_test]
    else:
        base_name = file_name

    # Strings in between _

    final_accuracy = fine_tune_pruned_model_with_mask(net, dataLoader=val, testLoader=testloader, epochs=args.epochs,
                                                      exclude_layers=cfg.exclude_layers, cfg=cfg,
                                                      save_folder=folder_name,
                                                      name=base_name)
    print("Final accuracy:{}".format(final_accuracy))
    print("Sparsity: {}".format(sparsity(net)))

    # if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
    #     os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
    #
    # strings_in_between = re.findall("(?<=\_)(.*?)(?=\_)", file_name)


def main(args):

    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": args.model,
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers,
         "data_path": args.data_folder,
         "input_resolution": args.input_resolution

         })

    if args.ffcv:
        from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
        train, val, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                      128, args.num_workers)
    else:
        print("Normal data loaders loaded!!!!")
        cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats
        # Data
        print('==> Preparing data..')
        current_directory = Path().cwd()
        data_path = "."
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
            data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "/home/luisaam/Documents/PhD/data/"
        print(data_path)
        batch_size = args.batch_size
        if "32" in args.name:
            batch_size = 32
        if "64" in args.name:
            batch_size = 64

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats_to_use),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if args.dataset == "cifar10":
            trainset = torchvision.datasets.CIFAR10(
                root=data_path, train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

            testset = torchvision.datasets.CIFAR10(
                root=data_path, train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        if args.dataset == "cifar100":
            trainset = torchvision.datasets.CIFAR100(
                root=data_path, train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

            testset = torchvision.datasets.CIFAR100(
                root=data_path, train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        if args.dataset == "tiny_imagenet":
            from test_imagenet import load_tiny_imagenet
            trainloader, valloader, testloader = load_tiny_imagenet(
                {"traindir": data_path + "/tiny_imagenet_200/train", "valdir": data_path + "/tiny_imagenet_200/val",
                 "num_workers": args.num_workers, "batch_size": batch_size})
        if args.dataset == "small_imagenet":
            if args.ffcv:
                from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
                trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train,
                                                                                          args.ffcv_val,
                                                                                          batch_size, args.num_workers)
            else:
                from test_imagenet import load_small_imagenet
                trainloader, valloader, testloader = load_small_imagenet(
                    {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                     "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution})

    from torchvision.models import resnet18, resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet24":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type))
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    dense_accuracy_list = []
    pruned_accuracy_list = []
    files_names = []
    search_string = "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset,
                                                                          args.RF_level, args.name)
    things = list(glob.glob(search_string))

    # if len(things) < 2:
    #     search_string = "{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset, args.RF_level)

    print("Glob text:{}".format(
        "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset, args.RF_level,
                                                              args.name)))
    print(things)

    for i, name in enumerate(
            glob.glob(search_string)):

        print(name)

        print("Device: {}".format(device))

        state_dict_raw = torch.load(name, map_location=device)

        dense_accuracy_list.append(state_dict_raw["acc"])
        net.load_state_dict(state_dict_raw["net"])

        print("Dense accuracy:{}".format(state_dict_raw["acc"]))
        print("Calculated Dense accuracy:{}".format(test(net, testloader=testloader)))

        prune_function(net, cfg)
        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)

        t0 = time.time()
        if args.ffcv:
            pruned_accuracy = test_ffcv(net, testloader=testloader, verbose=0)
        else:
            pruned_accuracy = test(net, testloader=testloader)

        t1 = time.time()

        print("Pruned accuracy:{}".format(pruned_accuracy))

        print("Time for inference: {}".format(t1 - t0))

        pruned_accuracy_list.append(pruned_accuracy)
        weight_names, weights = zip(*get_layer_dict(net))

        zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()

        pruning_rates_per_layer = list(map(zero_number, weights))

        seed_from_file1 = re.findall("_[0-9]_", name)

        print(seed_from_file1)

        seed_from_file2 = re.findall("_[0-9]_[0-9]_", name)

        print(seed_from_file2)

        seed_from_file3 = re.findall("\.[0-9]_", name)

        print(seed_from_file3)

        if seed_from_file3:

            seed_from_file = seed_from_file3[0].replace(".", "_")

        elif seed_from_file2:

            seed_from_file = seed_from_file2[0].split("_")[2]

        elif seed_from_file1:

            seed_from_file = seed_from_file1[0].replace("_", "")
        else:
            seed_from_file = i

        df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
        print("Seed from file {}".format(seed_from_file1))
        df2.to_csv(
            "{}_level_{}_seed_{}_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
                                                                              args.dataset, args.name,
                                                                              args.pruning_rate),
            index=False)

        print("Done")
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)

    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list,
                       })
    df.to_csv(
        "RF_{}_{}_{}_{}_{}_one_shot_summary.csv".format(args.model, args.RF_level, args.dataset, args.pruning_rate,
                                                        args.name),
        index=False)


def adjust_pruning_rate(list_of_excluded_weight, list_of_not_excluded_weight, global_pruning_rate):
    count_fn = lambda w: w.nelement()
    total_excluded = sum(list(map(count_fn, list_of_excluded_weight)))
    total_not_excluded = sum(list(map(count_fn, list_of_not_excluded_weight)))
    print("Total excluded: {}".format(total_excluded))
    print("Total not excluded: {}".format(total_not_excluded))
    if total_not_excluded != 0:
        print("Total excluded/Total not excluded: {}".format(total_excluded / total_not_excluded))
    if total_not_excluded == 0:
        return -1
    new_pruning_rate = ((total_excluded / total_not_excluded) + 1) * global_pruning_rate
    if new_pruning_rate > 0.98:
        return 0.98
    return new_pruning_rate


def n_shallow_layer_experiment(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)

    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    dense_accuracy_list = []
    pruned_accuracy_list = []
    files_names = []
    search_string = "{}/{}_normal_{}_*_level_{}*test_acc*.pth".format(args.folder, args.model, args.dataset,
                                                                      args.RF_level)
    things = list(glob.glob(search_string))
    if len(things) < 1:
        search_string = "{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset, args.RF_level)
    print("Glob text:{}".format(
        "{}/{}_normal_{}_*_level_{}*test_acc*.pth".format(args.folder, args.model, args.dataset, args.RF_level)))
    print(things)

    weight_names_begining, weights = zip(*get_layer_dict(net))

    name2index = dict(zip(weight_names_begining, range(len(weight_names_begining))))
    help_dict = dict(zip(weight_names_begining, weights))
    names_to_use = [i for i in weight_names_begining if i not in cfg.exclude_layers]
    n_shallow_layer_index_list = []
    n_shallow_layer_name_list = []
    adjusted_prunig_rate = []
    reset_exclude_layers = exclude_layers
    # names_to_use = set(weight_names_begining).difference(cfg.exclude_layers)

    for i, name in enumerate(
            glob.glob(search_string)):
        if "width" in name and "width" not in args.name:
            continue
        print("************************ {} ******************************".format(name))
        temp_list = []
        temp_list.extend(reset_exclude_layers)
        file_name = os.path.basename(name)
        print(file_name)
        print(names_to_use)
        print(cfg.exclude_layers)

        best_layer_name = None
        best_pruned_acc = 0
        best_new_pr = 0.9
        best_dense_accuracy = 0
        for layer_name in names_to_use:
            temp_list.append(layer_name)
            cfg.exclude_layers = temp_list

            state_dict_raw = torch.load(name, map_location=device)
            dense_accuracy = state_dict_raw["acc"]
            print("Dense accuracy:{}".format(state_dict_raw["acc"]))
            net.load_state_dict(state_dict_raw["net"])

            excluded_weights = [w for k, w in help_dict.items() if k in cfg.exclude_layers]

            not_excluded_weights = [w for k, w in help_dict.items() if k not in cfg.exclude_layers]

            new_pr = adjust_pruning_rate(excluded_weights, not_excluded_weights, args.pruning_rate)
            if new_pr == -1 or new_pr > 0.95:
                # best_layer_name = layer_name
                # best_pruned_acc = pruned_accuracy
                # best_dense_accuracy = dense_accuracy
                # best_new_pr = new_pr
                break
            cfg.amount = new_pr
            print("Excluded layers: {}".format(cfg.exclude_layers))

            prune_function(net, cfg)
            remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
            pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
            print("Pruned accuracy at layer {} with index {} and adjusted pruning rate {}:{}".format(layer_name,
                                                                                                     name2index[
                                                                                                         layer_name],
                                                                                                     new_pr,
                                                                                                     pruned_accuracy))

            if abs(dense_accuracy - pruned_accuracy) < 3:
                best_layer_name = layer_name
                best_pruned_acc = pruned_accuracy
                best_dense_accuracy = dense_accuracy
                best_new_pr = new_pr
                break
            if pruned_accuracy > best_pruned_acc:
                best_layer_name = layer_name
                best_pruned_acc = pruned_accuracy
                best_dense_accuracy = dense_accuracy
                best_new_pr = new_pr

            # pruning_rates_per_layer = list(map(zero_number, weights))
            #
            # seed_from_file1 = re.findall("_[0-9]_", name)[0].replace("_", "")
            # seed_from_file2 = re.findall("_[0-9]_[0-9]_", name)[0].split("_")[2]
            # if seed_from_file2:
            #     seed_from_file = seed_from_file2
            # else:
            #     seed_from_file = seed_from_file1

            # df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
            # df2.to_csv(
            #     "{}_level_{}_seed_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
            #                                                                    args.dataset, args.pruning_rate),
            #     index=False)

        n_shallow_layer_index_list.append(name2index[best_layer_name])
        n_shallow_layer_name_list.append(best_layer_name)
        dense_accuracy_list.append(best_dense_accuracy)
        pruned_accuracy_list.append(best_pruned_acc)
        files_names.append(file_name)
        adjusted_prunig_rate.append(best_new_pr)
        print("Done")
    #
    df = pd.DataFrame({"Name": files_names,
                       "N Shallow layer index": n_shallow_layer_index_list,
                       "N Shallow layer name": n_shallow_layer_name_list,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list
                       })
    df.to_csv("RF_{}_{}_{}_{}_N_shallow_summary.csv".format(args.model, args.RF_level, args.dataset, args.pruning_rate),
              index=False)


def fine_tune_summary(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)

    dense_accuracy_list = []
    pruned_accuracy_list = []
    fine_tuned_accuracy = []
    files_names = []
    fine_tuning_best_epoch = []

    for i, name in enumerate(
            glob.glob("{}/{}_normal_{}_*_level_{}*".format(args.folder, args.model, args.dataset,
                                                           args.RF_level))):
        print(name)
        if "initial_weights" in name or "32_bs" in name:
            continue
        state_dict_raw = torch.load(name)

        dense_accuracy_list.append(state_dict_raw["acc"])

        print("Dense accuracy:{}".format(state_dict_raw["acc"]))

        net.load_state_dict(state_dict_raw["net"])

        prune_function(net, cfg)

        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)

        pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)

        print("Pruned accuracy:{}".format(pruned_accuracy))

        pruned_accuracy_list.append(pruned_accuracy)

        weight_names, weights = zip(*get_layer_dict(net))

        zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()

        pruning_rates_per_layer = list(map(zero_number, weights))

        seed_from_file = re.findall("_[0-9]_", name)[0].replace("_", "")

        df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})

        df2.to_csv(
            "{}_level_{}_seed_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
                                                                           args.dataset, args.pruning_rate),
            index=False)

        # print(file_name)
        file_name = os.path.basename(name)
        if "test_acc" in file_name:
            index_until_test = file_name.index("test_acc")
            base_name = file_name[:index_until_test]

        else:

            base_name = file_name

        for pruned_name in glob.glob("{}/pruned/{}/{}*.pth".format(args.folder, args.pruning_rate, base_name)):
            files_names.append(pruned_name)
            state_dict_raw = torch.load(pruned_name)
            fine_tuned_accuracy.append(state_dict_raw["acc"])
            fine_tuning_best_epoch.append(state_dict_raw["epoch"])
    # for i, name in enumerate(
    #         glob.glob("{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset,
    #                                                                      args.RF_level))):
    #     print(name)
    #     state_dict_raw = torch.load(name)
    #
    #     dense_accuracy_list.append(state_dict_raw["acc"])
    #
    #     print("Dense accuracy:{}".format(state_dict_raw["acc"]))
    #
    #     net.load_state_dict(state_dict_raw["net"])
    #
    #     prune_function(net, cfg)
    #
    #     remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
    #
    #     pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
    #
    #     print("Pruned accuracy:{}".format(pruned_accuracy))
    #
    #     pruned_accuracy_list.append(pruned_accuracy)
    #
    #     weight_names, weights = zip(*get_layer_dict(net))
    #
    #     zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()
    #
    #     pruning_rates_per_layer = list(map(zero_number, weights))
    #
    #     seed_from_file = re.findall("_[0-9]_", name)[0].replace("_", "")
    #
    #     df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
    #
    #     df2.to_csv(
    #         "{}_level_{}_seed_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
    #                                                                        args.dataset, args.pruning_rate),
    #         index=False)
    #
    #     # print(file_name)
    #     file_name = os.path.basename(name)
    #     if "test_acc" in file_name:
    #         index_until_test = file_name.index("test_acc")
    #         base_name = file_name[:index_until_test]
    #     else:
    #
    #         base_name = file_name
    #     for pruned_name in glob.glob("{}/pruned/{}/{}*.pth".format(args.folder, args.pruning_rate, base_name)):
    #         files_names.append(pruned_name)
    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Fine Tuned Accuracy": fine_tuned_accuracy,
                       "Fine-tuning Best Epoch": fine_tuning_best_epoch,
                       })
    df.to_csv(
        "RF_{}_{}_{}_{}_fine_tuned_summary_100_epochs.csv".format(args.model, args.RF_level, args.dataset,
                                                                  args.pruning_rate),
        index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='One shot pruning statistics')

    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default="4", type=str, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet18", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
                        help='Location to save the models', required=True)
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--epochs', default=200, type=int, help='Epochs to train')
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size for training')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for small ImageNet')
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')

    args = parser.parse_args()

    try:

        args.RF_level = int(args.RF_level)

    except Exception as e:

        pass

    if args.experiment == 1:
        print("Experiment 1")
        print(args)
        main(args)
        # pruning_fine_tuning_experiment(args)
    if args.experiment == 2:
        print("Experiment 2")
        print(args)
        # main(args)
        pruning_fine_tuning_experiment(args)
    if args.experiment == 3:
        # pruning_fine_tuning_experiment(args)
        fine_tune_summary(args)
    if args.experiment == 4:
        n_shallow_layer_experiment(args)
        # main(args)
    # gradient_flow_calculation(args)
    # save_pruned_representations()
    # similarity_comparisons()
