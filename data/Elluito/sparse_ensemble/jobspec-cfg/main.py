print("First line")
import os

print("After os")
# import accelerate
print("After accelerate")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle

print("After Pickle")
# import paretoset
print("After pareto set")
import glob

print("After glob")
# import pygmo
import typing

print("After typing")
import pandas as pd

print("Until: 16")
import datetime as date

print("after datetime")
# import umap
print("after umap")
import wandb
import optuna

print("After optuna")
# sys.path.append('csgmcmc')
from alternate_models import ResNet, VGG
from csgmcmc.models import *

print("Until 20")
import omegaconf
import copy
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# import hydra
import torchvision
import torchvision.transforms as transforms
import scipy
import argparse
import scipy.optimize as optimize
# from torch.autograd import Variable
import numpy as np
import random
import torch.nn.utils.prune as prune
import platform
from functools import partial

print("Unitl line 40")
import glob
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.metrics as igm
# from torchmetrics import Accuracy
# import array as pyarr
import matplotlib

# - interactive backends:
#           GTK3Agg, GTK3Cairo, MacOSX, nbAgg,
#           Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo,
#           TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo
# matplotlib.use("TkAgg")
print("matplotlib backend {}".format(matplotlib.get_backend()))
# matplotlib.use('Agg')
from matplotlib.patches import PathPatch
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.axes import SubplotBase
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
from sklearn.manifold import MDS, TSNE
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from torch.utils.data import DataLoader, random_split, Dataset
import logging
import torchvision as tv
from itertools import combinations
import seaborn as sns
import seaborn.objects as so
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from plot_utils import plot_ridge_plot, plot_double_barplot, plot_histograms_per_group, stacked_barplot, \
    stacked_barplot_with_third_subplot, plot_double_barplot
from sparse_ensemble_utils import erdos_renyi_per_layer_pruning_rate
from sparse_ensemble_utils import erdos_renyi_per_layer_pruning_rate, get_layer_dict, is_prunable_module, \
    count_parameters, sparsity, get_percentile_per_layer, get_sampler, test, restricted_fine_tune_measure_flops, \
    get_random_batch, efficient_population_evaluation, get_random_image_label, check_for_layers_collapse, get_mask, \
    apply_mask, restricted_IMAGENET_fine_tune_ACCELERATOR_measure_flops, test_with_accelerator
from feature_maps_utils import load_layer_features
import re
from itertools import cycle
# import pylustrator
from shrinkbench.metrics.flops import flops
from pathlib import Path
import argparse
from decimal import Decimal

print("safe All imports")
# For different epsilon
plt.rcParams["mathtext.fontset"] = "cm"

# enable cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fs = 25

fig_size = (5, 3)
sns.reset_orig()
sns.reset_defaults()
matplotlib.rc_file_defaults()
plt.rcParams.update({
    "axes.linewidth": 0.5,
    'axes.edgecolor': 'black',
    "grid.linewidth": 0.4,
    "lines.linewidth": 1,
    'xtick.bottom': True,
    'xtick.color': 'black',
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    'ytick.left': True,
    'ytick.color': 'black',
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "figure.figsize": [3.3, 2.5],
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath}",
})


# matplotlib.use('TkAgg')


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


# function taken from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# #sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, convert_to_int=True, percentage=False, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if convert_to_int:
                kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                text = im.axes.text(j, i, valfmt(int(data[i, j]), None), **kw)
                texts.append(text)
            else:
                if percentage:
                    kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                    text = im.axes.text(j, i, valfmt(format_percentages(data[i, j]), None), **kw)
                    texts.append(text)
                else:
                    kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                    texts.append(text)

    return texts


# def all_subsets(ss):
#     return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))


def strip_prefix(net_state_dict: dict):
    new_dict = {}
    for k, v in net_state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict


def load_model(net, path):
    state_dict = torch.load(path)
    if "net" in state_dict.keys():
        net.load_state_dict(strip_prefix(state_dict["net"]))
    else:
        net.load_state_dict(torch.load(path))


def save_onnx(cfg, name: str = ""):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    net = get_model(cfg)
    train, val, test = get_datasets(cfg)
    batch = next(iter(test))

    input_names = ['Image']

    output_names = ['y_hat']

    torch.onnx.export(net, batch[0],
                      f'onnx_models/onnx_models_{cfg.architecture}_{cfg.dataset}_{cfg.model_type}_{name}.onnx',
                      input_names=input_names,
                      output_names=output_names)


def models_inspection(population):
    total_params = count_parameters(population[0])
    layers_of_models = [get_layer_dict(mod) for mod in population]
    zero_number = lambda w: torch.count_nonzero(w == 0) / total_params
    tsne = []
    mds = []
    MDS_embeddings = MDS(n_components=2)
    tsne_embeddings = TSNE(n_components=2)
    plt.figure(figsize=(10, 15), dpi=80)
    points = defaultdict(list)

    for i, layer in enumerate(layers_of_models):
        names, weigths = zip(*layer)
        sparsities = list(map(zero_number, weigths))
        plt.bar(x=np.arange(0, len(sparsities), 1), height=sparsities, tick_label=names)
        plt.xticks(rotation=-45)
        plt.ylabel("Percentage of total parameters", fontsize=20)
        plt.title("Number of zero weights", fontsize=20)
        plt.savefig("data/figures/bar_plot_model_{}.png".format(i))
        plt.close()
        for element in layer:
            # MDS.append("Model {} {}".format(i,element[0]),MDS_embeddings.fit_transform(element[1]))
            # embedding = MDS_embeddings.fit_transform(element[1].reshape(1,-1))
            label = "M:{}L:{}".format(i, element[0])
            points[element[0]].append((label, element[1]))

    fig, ax = plt.subplots()
    for layer_name, plotting_stuf in points.items():
        point_names, tensors = zip(*plotting_stuf)
        tensors = torch.stack(tensors)
        embeddings = tsne_embeddings.fit_transform(tensors)
        ax.scatter(embeddings[0], embeddings[1], label=layer_name)
        # ax.annotate(point_names[k], (emb[0], emb[1]))
    plt.ylabel("D1", fontsize=20)
    plt.xlabel("D2", fontsize=20)
    plt.title("T-SNE", fontsize=20)
    ax.legend()
    # plt.show()
    plt.savefig("data/figures/layer_projection_tsne.png")
    plt.close()

    fig, ax = plt.subplots()
    list_of_names = list(points.keys())
    for layer_name, plotting_stuf in points.items():
        point_names, tensors = zip(*plotting_stuf)
        tensors = torch.stack(tensors)
        embeddings = MDS_embeddings.fit_transform(tensors)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], label=layer_name)
        # ax.annotate(point_names[k], (emb[0], emb[1]))
    plt.ylabel("D1", fontsize=20)
    plt.xlabel("D2", fontsize=20)
    plt.title("MDS", fontsize=20)
    ax.legend()
    # plt.show()
    plt.savefig("data/figures/layer_projection_MDS.png")
    plt.close()


def plot_pareto(df, title: str):
    # define aesthetics for plot
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig, ax = plt.subplots()
    ax.bar(df.index, df['count'], color=color1)

    # add cumulative percentage line to plot
    ax2 = ax.twinx()
    ax2.plot(df.index, df['cumperc'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # specify axis colors
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    ax.tick_params(axis="x", labelrotation=90)
    ax2.tick_params(axis="x", labelrotation=90)
    plt.title(title, fontsize=20)
    # display Pareto chart
    plt.show()


def weight_inspection(cfg, cutoff):
    """
    This function inspects statistics about the weights of the pruned models
    :return:
    """
    performances = np.load("data/population_data/performances_{}.npy".format(cfg.noise))
    with open("data/population_data/population_models_{}.pkl".format(cfg.noise), "rb") as f:
        pop = pickle.load(f)
    sorted_indx = np.argsort(performances)
    sorted_models = [pop[i] for i in sorted_indx]

    above_cutoff_index = np.where(performances[sorted_indx] >= cutoff)[0]
    underneath_cutoff_index = np.where(performances[sorted_indx] < cutoff)[0]
    total_params = count_parameters(pop[0])
    zero_number = lambda w: torch.count_nonzero(w == 0)
    number_param = lambda w: w.nelement()
    layer = get_layer_dict(pop[0])
    names, weights = zip(*layer)
    elements = list(map(number_param, weights))

    sparsities = list(map(zero_number, weights))
    weights = list(map(lambda w: w.numpy(), weights))
    sparsities = list(map(lambda w: w.numpy(), sparsities))
    # elements =list(map(lambda w: w.numpy(),elements))
    df = pd.DataFrame({'count': elements})
    df.index = names
    df = df.sort_values(by='count', ascending=False)
    df['cumperc'] = df['count'].cumsum() / df['count'].sum() * 100
    plot_pareto(df, "Number of 0 parameters")
    # Go trough all the above_cutoff_index


######################### Noise adding functions #################################################################
def add_geometric_gaussian_noise_to_weights(m, sigma=0.2):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_weights(m, sigma=0.01, adaptive=False):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if adaptive:
                sigma_adaptive = torch.quantile(torch.abs(m.weight), torch.tensor([0.50]))
                m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma_adaptive).to(m.weight.device))
            else:
                m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))


def add_geogaussian_noise_to_layers(model: torch.nn.Module, sigma_per_layer: dict, exclude_layers: list = []):
    named_modules = model.named_modules()

    with torch.no_grad():
        for name, m in named_modules:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                     nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layers:
                sigma = sigma_per_layer[name]
                m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_layers(model: torch.nn.Module, sigma_per_layer: dict, iterative: bool = False, exclude_layers:
list =
[]):
    named_modules = model.named_modules()
    with torch.no_grad():
        for name, m in named_modules:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                     nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layers:
                sigma = sigma_per_layer[name]
                if "weight_mask" in dict(m.named_buffers()).keys() and iterative:
                    weight_mask = dict(m.named_buffers())["weight_mask"]
                    noise = torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device)
                    noise.mul_(weight_mask)
                    m.weight.data.add_(noise)
                else:
                    m.weight.data.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))


def get_noisy_sample_sigma_per_layer(net: torch.nn.Module, cfg: omegaconf.DictConfig, sigma_per_layer, clone=True):
    current_model = None
    if clone:
        current_model = copy.deepcopy(net)
    else:
        current_model = net
    if cfg.noise == "gaussian":
        add_gaussian_noise_to_layers(current_model, sigma_per_layer=sigma_per_layer, exclude_layers=cfg.exclude_layers)
    elif cfg.noise == "geogaussian":
        add_geogaussian_noise_to_layers(current_model, sigma_per_layer=sigma_per_layer,
                                        exclude_layers=cfg.exclude_layers)
    return current_model


# def get_noisy_sample_pruned_net_work(net: torch.nn.Module, cfg: omegaconf.DictConfig, sigma_per_layer):
def get_noisy_sample(net: torch.nn.Module, cfg: omegaconf.DictConfig, noise_on_LAMP_scores: bool = False):
    current_model = copy.deepcopy(net)
    if cfg.noise == "gaussian":
        current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))

    elif cfg.noise == "geogaussian":
        current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
    return current_model


##################################################################################################################


def weights_to_prune(model: torch.nn.Module, exclude_layer_list=[]):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((m, "weight"))
            print(name)

    return modules


def get_weights_of_layer_name(model, layer_name):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if name == layer_name:
                return (m, "weight")


def remove_reparametrization(model, name_module="", exclude_layer_list: list = []):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


def count_zero_parameters(model):
    return sum((p == 0).sum() for p in model.parameters() if p.requires_grad)


def get_proba_function(C, N):
    """
    :param C: pruning rate
    :param N:  number of parameters
    :return: function
    """

    def loss(k):
        return ((1 - np.exp(-k * N)) / k - C) ** 2

    result = optimize.minimize(loss, np.array([1]))
    print("K:{}".format(result["x"]))
    K = result["x"]

    def function(x):
        return np.exp(-K * x)

    return function


############################### 27 of september experiments ###################################################
def heatmap1_exp(cfg):
    # data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
    #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()

    trainloader, valloader, testloader = get_datasets(cfg)
    # for filename in glob.glob("trained_models/cifar10/*.pt"):

    load_model(net, cfg.solution)
    # model_index= filename[filename.index(".")-1]
    N = cfg.population
    number_populations = 3
    original_performance = test(net, use_cuda, testloader)
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pruning_percentages.reverse()
    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_mean_decrease = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_ranking = np.zeros((len(sigmas), len(pruning_percentages)))
    deterministic_pruning_acc = []
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        pruned_original = copy.deepcopy(net)
        weights_to_prunes = weights_to_prune(pruned_original)
        prune.global_unstructured(
            weights_to_prunes,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage
        )
        remove_reparametrization(pruned_original)
        print("performance of pruned original")
        pruned_original_performance = test(pruned_original, use_cuda, testloader)

        for j, sigma in enumerate(sigmas):
            performance_diff = []
            performances = [pruned_original_performance]
            ranks_pruned_model = []
            for _ in range(number_populations):

                for n in range(N):
                    current_model = copy.deepcopy(net)
                    if cfg.noise == "gaussian":

                        current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                    elif cfg.noise == "geogaussian":

                        current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                    prune.global_unstructured(weights_to_prune(current_model),
                                              pruning_method=prune.L1Unstructured,
                                              amount=pruning_percentage)

                    remove_reparametrization(current_model)
                    print("Performance for model {}".format(n))
                    current_model_acc = test(current_model, use_cuda, testloader)
                    performances.append(current_model_acc)
                    performance_diff.append((original_performance - current_model_acc) / original_performance)

                sorted_performances = sorted(performances)
                sorted_performances.reverse()
                rank_original_model = sorted_performances.index(pruned_original_performance)
                ranks_pruned_model.append(rank_original_model)

            matrix_mean_decrease[j, i] = np.mean(performance_diff)
            matrix_ranking[j, i] = np.mean(ranks_pruned_model)
            print("Matrix Mean Decrease")
            print(np.matrix(matrix_mean_decrease))
            print("Matrix Ranking")
            print(np.matrix(matrix_ranking))
            # matrix_mean_decrease[j, i] = i+j
        # matrix_ranking[j, i] = i-j

    df_ranking = pd.DataFrame(matrix_ranking, columns=pruning_percentages)
    df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_ranking.index = sigmas
    df_mean_decrease.index = sigmas
    df_ranking.to_csv(f"data/ranking_matrix_{cfg.noise}_traditional_train.csv")
    df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}_traditional_train.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    im = ax.imshow(matrix_mean_decrease.transpose(), aspect=0.5)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    # plt.yticks(ticks=pruning_percentages)
    # plt.xticks(ticks=sigmas)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Mean decrease in performance', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.title(f"Decrease in performance model", fontsize=15)
    plt.tight_layout()
    result = time.localtime(time.time())
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.savefig(
        f"data/figures/meanDecrease_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.pdf")
    plt.savefig(
        f"data/figures/meanDecrease_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.png")
    plt.close()
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_ranking.transpose(), aspect=0.5)
    ax.set_xticklabels(sigmas, minor=False)
    ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Rank', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title(f"Rank of det model")
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(
        f"data/figures/ranking_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.pdf")
    plt.savefig(
        f"data/figures/ranking_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.png")
    plt.close()


def copy_buffers(from_net: nn.Module, to_net: nn.Module):
    """

    @param from_net: Network that contains the mask to be transfered to "to_net"
    @param to_net: Network to wich weights are going to be pruned by the mask in "from_net"
    @return: None
    """
    iter_1 = to_net.named_modules()
    with torch.no_grad():
        for name, m in iter_1:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                temp_dict = dict(dict(from_net.named_modules())[name].named_buffers())
                if len(list(temp_dict.items())) == 1:
                    weight_mask = dict(temp_dict)["weight_mask"]
                    m.weight.data.mul_(weight_mask)


def heatmap2_mean_decrease_maskTransfer_exp(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    N = cfg.population
    pop = []
    performance = []
    number_of_populations = 3
    original_performance = test(net, use_cuda, testloader)
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pruning_percentages.reverse()
    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_mean_decrease = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_ranking = np.zeros((len(sigmas), len(pruning_percentages)))
    deterministic_pruning_acc = []
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        pruned_original = copy.deepcopy(net)
        weights_to_prune = weights_to_prune(pruned_original)
        prune.global_unstructured(
            weights_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage
        )
        remove_reparametrization(pruned_original)
        print("performance of pruned original")
        pruned_original_performance = test(pruned_original, use_cuda, testloader)

        for j, sigma in enumerate(sigmas):
            # mean_index = []
            # for k in range(number_of_populations):
            pruned_original = copy.deepcopy(net)

            performance_diff = []
            # performances = [pruned_original_performance]
            # Loop over the  population
            for n in range(N):
                current_model = copy.deepcopy(net)
                if cfg.noise == "gaussian":

                    current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                elif cfg.noise == "geogaussian":

                    current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                prune.global_unstructured(
                    weights_to_prune(current_model),
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage)
                copy_buffers(from_net=current_model, to_net=pruned_original)

                # remove_reparametrization(current_model)

                print("Performance for model {}".format(n))

                current_model_acc = test(pruned_original, use_cuda, testloader)
                performance_diff.append((current_model_acc - pruned_original_performance) / pruned_original_performance)
                # current_model.cpu()
                # performances.append(current_model_acc)
            #

            # sorted_performances = sorted(performances)
            # sorted_performances.reverse()
            # rank_original_model = sorted_performances.index(pruned_original_performance)
            # mean_index.append(rank_original_model)
            matrix_mean_decrease[j, i] = np.mean(performance_diff)
            # matrix_ranking[j, i] = np.mean(mean_index)
            print("Matrix Mean Decrease")
            print(np.matrix(matrix_mean_decrease))
            # print("Matrix Ranking")
            # print(np.matrix(matrix_ranking))
        # matrix_mean_decrease[j, i] = i+j
        # matrix_ranking[j, i] = i-j

    df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/buffer_transfer_meanDecrease_matrix_{cfg.noise}_pop_{cfg.population}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_mean_decrease.transpose(), aspect=0.5)
    # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Mean decrease', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title("Mask transfer: {}".format(cfg.noise))
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.png")
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.pdf")
    plt.close()


def heatmap2_mean_decrease_maskTransfer_exp_to_stochastic(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    N = cfg.population
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # pruning_percentages.reverse()

    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_diff = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_stochastic_acc = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_transfer_mask_acc = np.zeros((len(sigmas), len(pruning_percentages)))
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        # pruned_original = copy.deepcopy(net)
        # weights_to_prune = weigths_to_prune(pruned_original)
        # prune.global_unstructured(
        #     weights_to_prune,
        #     pruning_method=prune.L1Unstructured,
        #     amount=pruning_percentage
        # )
        # remove_reparametrization(pruned_original)
        # print("performance of pruned original")
        # pruned_original_performance = test(pruned_original, use_cuda, testloader)

        for j, sigma in enumerate(sigmas):
            # mean_index = []
            # for k in range(number_of_populations):
            pruned_original = copy.deepcopy(net)

            performance_diff = []
            stochastic_performance = []
            transfer_mask_performance = []

            # performances = [pruned_original_performance]
            # Loop over the  population
            for n in range(N):
                current_model = copy.deepcopy(net)
                if cfg.noise == "gaussian":

                    current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                elif cfg.noise == "geogaussian":

                    current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                prune.global_unstructured(
                    weights_to_prune(current_model),
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage)
                copy_buffers(from_net=current_model, to_net=pruned_original)

                # remove_reparametrization(current_model)

                print("Performance for model {}".format(n))

                original_transfer_mask_model_acc = test(pruned_original, use_cuda, testloader)
                stochastic_model_performance = test(current_model, use_cuda, testloader)
                performance_diff.append(original_transfer_mask_model_acc - stochastic_model_performance)
                stochastic_performance.append(stochastic_model_performance)
                transfer_mask_performance.append(original_transfer_mask_model_acc)
                # current_model.cpu()
                # performances.append(current_model_acc)
            #

            # sorted_performances = sorted(performances)
            # sorted_performances.reverse()
            # rank_original_model = sorted_performances.index(pruned_original_performance)
            # mean_index.append(rank_original_model)
            matrix_diff[j, i] = np.mean(performance_diff)
            matrix_transfer_mask_acc[j, i] = np.mean(transfer_mask_performance)
            matrix_stochastic_acc[j, i] = np.mean(stochastic_performance)
            # matrix_ranking[j, i] = np.mean(mean_index)
            print("Matrix Diff")
            print(np.matrix(matrix_diff))
            print("Matrix stochastic performance")
            print(np.matrix(matrix_stochastic_acc))
            print("Matrix mask transfer")
            print(np.matrix(matrix_transfer_mask_acc))
            # print("Matrix Ranking")
            # print(np.matrix(matrix_ranking))
        # matrix_diff[j, i] = i+j
        # matrix_ranking[j, i] = i-j
    # This is the difference between the stochastic pruned model and the mask transfer on to the original one
    df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/bufferTransfer_V_stochasticPruned_diff_matrix_{cfg.noise}_pop"
                            f"_{cfg.population}.csv")
    #  This is the performance of the stochastic pruned model
    df_mean_decrease = pd.DataFrame(matrix_stochastic_acc, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/stochastic_comparison_stochastic_performance_matrix_{cfg.noise}_pop"
                            f"_{cfg.population}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    #  This is the performance of the mask transfer model
    df_mean_decrease = pd.DataFrame(matrix_transfer_mask_acc, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/stochastic_comparison_maskTransfer_performance_matrix_{cfg.noise}_pop"
                            f"_{cfg.population}.csv")
    ####################################  Mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_diff.transpose(), aspect=0.5)
    # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Mean decrease', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title("Mask Transfer V Stochastic pruned: {}".format(cfg.noise))
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.png")
    plt.close()


def heatmap2_mean_decrease_maskTransfer_exp_to_random(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    N = cfg.population
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pruning_percentages.reverse()
    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_diff = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_stochastic = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_deterministic = np.zeros((len(sigmas), len(pruning_percentages)))
    random_net_1 = ResNet18()
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        pruned_original = copy.deepcopy(net)
        weights_to_prune = weights_to_prune(pruned_original)
        prune.global_unstructured(
            weights_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage
        )
        # remove_reparametrization(pruned_original)
        deterministic_net = copy.deepcopy(random_net_1)
        copy_buffers(from_net=pruned_original, to_net=deterministic_net)
        for j, sigma in enumerate(sigmas):
            # mean_index = []
            # for k in range(number_of_populations):
            performance_diff = []
            stochastic_performance = []
            deterministic_net_performance = []
            # # performances = [pruned_original_performance]
            # # Loop over the  population
            for n in range(N):
                stochastic_random_net = copy.deepcopy(random_net_1)

                current_model = copy.deepcopy(net)
                if cfg.noise == "gaussian":

                    current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                elif cfg.noise == "geogaussian":

                    current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                prune.global_unstructured(
                    weights_to_prune(current_model),
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage)
                copy_buffers(from_net=current_model, to_net=stochastic_random_net)

                # remove_reparametrization(current_model)

                print("Performance for model {}".format(n))

                stochastic_random_net_acc = test(stochastic_random_net, use_cuda, testloader)
                deterministic_net_acc = test(deterministic_net, use_cuda, testloader)

                performance_diff.append(deterministic_net_acc - stochastic_random_net_acc)
                stochastic_performance.append(stochastic_random_net_acc)
                deterministic_net_performance.append(deterministic_net_acc)

            # sorted_performances = sorted(performances)
            # sorted_performances.reverse()
            # rank_original_model = sorted_performances.index(pruned_original_performance)
            # mean_index.append(rank_original_model)
            matrix_diff[j, i] = np.mean(performance_diff)
            matrix_stochastic[j, i] = np.mean(stochastic_performance)
            matrix_deterministic[j, i] = np.mean(deterministic_net_performance)

            # matrix_ranking[j, i] = np.mean(mean_index)
            print("Matrix Diff deterministic vs stochastic (det-stho)")
            print(np.matrix(matrix_diff))
            print("Matrix mean stochastic performance")
            print(np.matrix(matrix_stochastic))
            print("Matrix mean deterministic performance")
            print(np.matrix(matrix_deterministic))

            # print("Matrix Ranking")
            # print(np.matrix(matrix_ranking))
        # matrix_mean_decrease[j, i] = i+j
        # matrix_ranking[j, i] = i-j

    df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/random_stochastic_V_deterministic_diff_matrix_{cfg.noise}_pop_{cfg.population}.csv")

    df_mean_decrease = pd.DataFrame(matrix_stochastic, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/random_stochastic_performance_matrix_{cfg.noise}_pop_{cfg.population}.csv")

    df_mean_decrease = pd.DataFrame(matrix_deterministic, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/random_deterministic_performance_matrix_{cfg.noise}_pop_{cfg.population}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_diff.transpose(), aspect=0.5)
    # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Difference in performance', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title("Stochastic V deterministic Mask: {}".format(cfg.noise))
    texts = annotate_heatmap(im, valfmt="{x:1.2f}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(f"data/figures/diff_bufferTransfer_det_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/diff_bufferTransfer_det_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.png")
    plt.close()
    ######################################
    # ax = plt.subplot()
    # im = ax.imshow(matrix_diff.transpose(), aspect=0.5)
    # # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # # ax.set_yticklabels(pruning_percentages, minor=False)
    # plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    # plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    # plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    # plt.ylabel("Pruning Rate", fontsize=12)
    # cbar = plt.colorbar(im)
    # cbar.ax.set_ylabel('Difference in performance', rotation=270, labelpad=8.5)
    # # plt.gcf().set_size_inches(5.1, 5.1)
    # plt.title("Stochastic V deterministic Mask: {}".format(cfg.noise))
    # texts = annotate_heatmap(im, valfmt="{x:1.2f}")
    # plt.tight_layout()
    # # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # result = time.localtime(time.time())
    # plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
    #             f"-{result.tm_min}.pdf")
    # plt.close()


def format_sigmas(sigma):
    string = "{:1.3f}".format(sigma)
    return string


def format_percentages(value):
    string = "{:10.1f}%".format(value).replace(" ", "")
    return string


def plot_heatmaps(cfg, plot_index=1):
    sigmas = None
    if cfg.noise == "gaussian":
        matrix_mean_decrease = np.loadtxt("data/mean_decrease_matrix_gaussian_traditional_average.txt") * 100
        matrix_ranking = np.loadtxt("data/ranking_matrix_gaussian_traditional_average.txt")
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        matrix_mean_decrease = np.loadtxt("data/mean_decrease_mask_transfer_geogaussian.txt") * 100
        matrix_ranking = np.loadtxt("data/ranking_matrix_geogaussian.txt")
        sigmas = np.linspace(0.1, 0.7, 10)
    df_mean_decrease = pd.read_csv(f"data/ranking_matrix.csv", header=0, index_col=0)
    # sigmas = list(df_mean_decrease.index)
    sigmas = list(map(format_sigmas, sigmas))
    pruning_percentages = df_mean_decrease.columns

    # df_ranking = pd.DataFrame(matrix_ranking, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_ranking, columns=pruning_percentages)
    # df_ranking.index = sigmas
    # df_mean_decrease.index = sigmas
    # df_ranking.to_csv(f"data/ranking_matrix_{cfg.noise}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    if plot_index >= 0:
        ####################################  mean decrease heatmap #################################
        ax = plt.subplot()
        im = ax.imshow(matrix_mean_decrease.transpose(), aspect=0.5)
        # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
        plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
        plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
        # ax.set_yticklabels(pruning_percentages, minor=False)
        # plt.yticks(ticks=pruning_percentages)
        # plt.xticks(ticks=sigmas)
        plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
        plt.ylabel("Pruning Rate", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Mean decrease in performance', rotation=270, labelpad=8.5)
        # plt.gcf().set_size_inches(5.1, 5.1)
        texts = annotate_heatmap(im, valfmt="{x:0.1f}", convert_to_int=False)
        plt.title("Mask transfer performance decrease", fontsize=15)
        plt.tight_layout()
        result = time.localtime(time.time())
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        plt.savefig(f"data/figures/meanDecrease_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                    f"-{result.tm_min}.png")
        plt.savefig(f"data/figures/meanDecrease_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                    f"-{result.tm_min}.pdf")
        plt.close()
    ####################################  ranking of original pruned model heatmap #################################
    if plot_index > 0:
        ax = plt.subplot()
        im = ax.imshow(matrix_ranking.transpose(), aspect=0.5)
        # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
        # ax.set_yticklabels(pruning_percentages, minor=False)
        plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
        plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
        plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
        plt.ylabel("Pruning Rate", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Rank', rotation=270, labelpad=8.5)
        # plt.gcf().set_size_inches(5.1, 5.1)
        plt.title("Rank of deterministic")
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        result = time.localtime(time.time())
        plt.savefig(
            f"data/figures/ranking_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}-{result.tm_min}.png")
        plt.savefig(f"data/figures/ranking_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}-"
                    f"{result.tm_min}.png")
        plt.close()


################################# Noise calibration with optuna ##################################
def objective_function(sto_performance, deter_performance, pruning_rate):
    if sto_performance > deter_performance:
        return ((sto_performance - deter_performance)) * pruning_rate
    if sto_performance <= deter_performance:
        return ((sto_performance - deter_performance))


def conver_fitness_to_difference(fitness, pruning_rate):
    if fitness < 0:
        return fitness
    else:
        fitnes / pruning_rate


def find_pr_sigma_MOO_for_dataset_architecture_one_shot_GMP(trial: optuna.trial.Trial, cfg, one_batch=True,
                                                            use_population=True, use_log_sigma=False, Fx=1):
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = get_model(cfg)
    train, val_loader, test_loader = get_datasets(cfg)

    # dense_performance = test(net, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    if use_log_sigma:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01, log=True)
    else:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01)
    sample_pruning_rate = trial.suggest_float("pruning_rate", 0.01, 0.99)

    # def objective_function(stochastic_performance,deter_performance, pruning_rate):
    #     return ((stochastic_performance - deter_performance)) * pruning_rate

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [sample_sigma] * number_of_layers))
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.amount = sample_pruning_rate

    pruned_model = copy.deepcopy(net)
    prune_function(pruned_model, cfg_copy)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)

    # Add small noise just to get tiny variations of the deterministic case
    det_performance = test(pruned_model, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    print("Det performance: {}".format(det_performance))

    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])
    # sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    if use_population:
        performance_of_models = []
        for individual_index in range(10):
            ############### Here I ask for pr and for sigma ###################################

            current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
            # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
            prune_function(current_model, cfg_copy)

            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
            # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
            stochastic_performance = test(current_model, use_cuda=True, testloader=val_loader, verbose=0,
                                          one_batch=one_batch)
            # Dense stochastic performance
            performance_of_models.append(stochastic_performance)
        performance_of_models = np.array(performance_of_models)
        median = np.median(performance_of_models)
        print("Median of population performance: {}".format(median))
        average_difference_performance = det_performance - performance_of_models
        fitness_function_median = objective_function(median, det_performance, sample_pruning_rate)
        # fitness_function_vector = np.array(list(map()))objective_function(performance_of_models, det_performance,sample_pruning_rate)
        # average_fitness_function = fitness_function_vector.mean()
        if Fx == 1:
            return median, fitness_function_median
        else:
            return median, sample_pruning_rate
    else:
        stochastic_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(stochastic_model, cfg_copy)

        remove_reparametrization(stochastic_model, exclude_layer_list=cfg.exclude_layers)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        stochastic_performance = test(stochastic_model, use_cuda=True, testloader=val_loader, verbose=0,
                                      one_batch=one_batch)
        fitness_function_median = objective_function(stochastic_performance, det_performance, sample_pruning_rate)
        print("Stochastic performance: {}".format(stochastic_performance))
        if Fx == 1:
            return stochastic_performance, fitness_function_median
        else:
            return stochastic_performance, sample_pruning_rate


def test_pr_sigma_combination(cfg, pr, sigma, cal_val=False):
    net = get_model(cfg)
    train, val, testloader = get_datasets(cfg)

    pruned_model = copy.deepcopy(net)
    cfg.amount = pr
    prune_function(pruned_model, cfg)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # Add small noise just to get tiny variations of the deterministic ase
    det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)
    if cal_val:
        det_performance_val = test(pruned_model, use_cuda=True, testloader=val, verbose=0)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [sigma] * number_of_layers))
    # print("Deterministic performance on test set = {}".format(det_performance))
    if cal_val:
        print("Deterministic performance on val set = {}".format(det_performance_val))
    stochastic_performance = []

    performance_of_models = []
    if cal_val:
        performance_of_models_val = []

    for individual_index in range(10):
        ############### Here I ask for pr and for sigma ###################################

        current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(current_model, cfg)

        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        stochastic_performance = test(current_model, use_cuda=True, testloader=testloader, verbose=0)
        if cal_val:
            stochastic_performance_val = test(current_model, use_cuda=True, testloader=val, verbose=0)

        performance_of_models.append(stochastic_performance)
        if cal_val:
            performance_of_models_val.append(stochastic_performance_val)

    performance_of_models = np.array(performance_of_models)
    median = np.median(performance_of_models)
    # print("Median accuracy of population: {}".format(median))
    fitness_function_median_test = objective_function(median, det_performance, pr)
    # print("Fintness function of the median on test set: {}".format(fitness_function_median))

    ##########  val functions###########################
    if cal_val:
        performance_of_models_val = np.array(performance_of_models_val)
        median_val = np.median(performance_of_models_val)
        print("Median accuracy of population valset: {}".format(median))
        fitness_function_median_val = objective_function(median, det_performance_val, pr)
        print("Fintness function of the median on val set: {}".format(fitness_function_median))
        return fitness_function_median_test, fitness_function_median_val, median

    return fitness_function_median_test, median, median - det_performance


def run_pr_sigma_search_MOO_for_cfg(cfg, arg):
    one_batch = False  # arg["one_batch"]
    one_batch_string = "whole_batch" if not one_batch else "one_batch"
    sampler = arg["sampler"]
    log_sigma = arg["log_sigma"]
    number_of_trials = arg["trials"]
    functions = arg["functions"]

    use_population = True if cfg["population"] > 1 else False
    function_string = "F1" if functions == 1 else "F2"
    if sampler == "nsga":
        # sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop",n_startup_trials=10,inc_popsize=2)
        sampler = optuna.samplers.NSGAIISampler()
    elif sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    else:
        raise Exception("Sampler {} is not suported for this experiment".format(sampler))
    # # sampler = optuna.samplers.CmaEsSampler(n_startup_trials=10,popsize=4)
    # vj
    # sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler,
                                study_name="stochastic-global-pr-and-sigma-optimisation-MOO-{}-{}-{}-{}".format(
                                    cfg.architecture,
                                    cfg.dataset, sampler, function_string),
                                storage="sqlite:///find_pr_sigma_database_MOO_{}_{}_{}_{}_{}.dep".format(
                                    cfg.architecture,
                                    cfg.dataset,
                                    sampler,
                                    one_batch, function_string),
                                load_if_exists=True)

    # study.optimize(
    #     lambda trial: find_pr_sigma_MOO_for_dataset_architecture_one_shot_GMP(trial, cfg, one_batch, use_population,
    #                                                                           use_log_sigma=log_sigma,Fx=functions),
    #     n_trials=args["trials"])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trials = study.best_trials
    print("Size of the pareto front: {}".format(len(trials)))

    sigmas_list = []
    pruning_rate_list = []
    f1_list = []
    f2_list = []
    difference_with_deterministic_list = []
    fitness_list = []

    if functions == 1:
        trial_with_highest_difference = max(study.best_trials, key=lambda t: t.values[0])
        f1, f2 = trial_with_highest_difference.values
        print("  Values: {},{}".format(f1, f2))
        print("  Params: ")
        for key, value in trial_with_highest_difference.params.items():
            print("    {}: {}".format(key, value))
        fitness_function_on_test_set, test_median_stochastic_performance = test_pr_sigma_combination(cfg,
                                                                                                     trial_with_highest_difference.params[
                                                                                                         "pruning_rate"],
                                                                                                     trial_with_highest_difference.params[
                                                                                                         "sigma"])
        print(
            "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
                fitness_function_on_test_set, test_median_stochastic_performance,
                fitness_function_on_test_set / trial_with_highest_difference.params[
                    "pruning_rate"]))
    else:
        trial_with_highest_difference = max(study.best_trials, key=lambda t: t.values[0])

        f1, f2 = trial_with_highest_difference.values
        print("  Values: {},{}".format(f1, f2))
        print("  Params: ")
        for key, value in trial_with_highest_difference.params.items():
            print("    {}: {}".format(key, value))
        fitness_function_on_test_set, test_median_stochastic_performance = test_pr_sigma_combination(cfg,
                                                                                                     trial_with_highest_difference.params[
                                                                                                         "pruning_rate"],
                                                                                                     trial_with_highest_difference.params[
                                                                                                         "sigma"])
        print(
            "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
                fitness_function_on_test_set, test_median_stochastic_performance,
                fitness_function_on_test_set / trial_with_highest_difference.params[
                    "pruning_rate"]))

    for trial in trials:
        f1, f2 = trial.values
        pr, sigma = trial.params["pruning_rate"], trial.params["sigma"]
        f1_list.append(f1)
        f2_list.append(f2)
        print("  Values: {},{}".format(f1, f2))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        fitness_function_on_test_set, test_median_stochastic_performance, difference = test_pr_sigma_combination(cfg,
                                                                                                                 trial.params[
                                                                                                                     "pruning_rate"],
                                                                                                                 trial.params[
                                                                                                                     "sigma"])

        difference_with_deterministic_list.append(difference)
        fitness_list.append(fitness_function_on_test_set)

    print(
        "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
            fitness_function_on_test_set, test_median_stochastic_performance,
            fitness_function_on_test_set / trial.params[
                "pruning_rate"]))

    sigmas_list.append(sigma)
    pruning_rate_list.append(pr)

    p = pd.read_csv("pareto_front_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler, function_string,
                                                             one_batch_string))

    p["Sigma"] = sigmas_list

    p = pd.DataFrame({"Pruning rate": pruning_rate_list, "Stochastic performance": f1_list,
                      "Fitness": fitness_list, "Sigma": sigmas_list,
                      "Difference with deterministic": difference_with_deterministic_list, "F2": f2_list})

    p.to_csv("pareto_front_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler, function_string,
                                                      one_batch_string), index=False)

    #############################################################
    #                   plotting pareto front
    #############################################################

    #
    # p = pd.read_csv("pareto_front_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler, function_string,
    #                                                          one_batch_string))
    # # p["Difference with deterministic"] = p["Fitness"]/p["Pruning rate"]
    # # plt.figure()
    # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # # g = sns.scatterplot(data=p, x="Stochastic performance", y="Pruning rate", hue="Fitness",palette="deep")
    # cmap = plt.cm.get_cmap('RdYlBu')
    # # cmap = mpl.cm.viridis
    # # # cmap = (matplotlib.colors.ListedColormap(['royalblue', 'cyan', 'orange', 'red']))
    # # cmap = matplotlib.colors.ListedColormap(['royalblue', 'cyan', 'yellow', 'orange'])
    # # diff = p["Difference with deterministic"]
    # # min_val = diff.min()
    # # q25 = diff.quantile(q=0.25)
    # # q50 = diff.quantile(q=0.50)
    # # q75 = diff.quantile(q=0.75)
    # # max_val = diff.max()
    # # bounds =[min_val,q25,0,q50,q75]
    # # bounds.sort()
    #
    # # bounds = [p["Difference with deterministic"].min(), -0.5,0,0.1 ,p["Difference with deterministic"].max()]
    # # bounds = np.linspace(p["Difference with deterministic"].min(),p["Difference with deterministic"].max(),6)
    # # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    # # # fig.colorbar(
    # # #     mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    # # #     cax=ax,
    # # #     extend='both',
    # # #     ticks=bounds,
    # # #     spacing='proportional',
    # # #     orientation='horizontal',
    # # #     label='Discrete intervals, some other units',
    # # # )
    # #
    #
    # # sc = ax.scatter(xs=p["Stochastic performance"], ys=p["Pruning rate"], c=p["Difference with deterministic"], s=15, cmap=cmap,norm=norm)
    # p_lees_than_0 = p[p["Difference with deterministic"] < 0]
    # p_more_than_0 = p[p["Difference with deterministic"] > 0]
    # sc_less_than_0 = plt.scatter(y=p_lees_than_0["Stochastic performance"], x=p_lees_than_0["Pruning rate"],
    #                              facecolors='none', edgecolors='k', s=15)
    # sc = plt.scatter(y=p_more_than_0["Stochastic performance"], x=p_more_than_0["Pruning rate"],
    #                  c=p_more_than_0["Difference with deterministic"], cmap=cmap,
    #                  norm=matplotlib.colors.PowerNorm(gamma=2), s=15)
    #
    # plt.ylabel("Stochastic performance on Val set")
    # plt.xlabel("Pruning rate")
    # plt.colorbar(sc, label="Difference with deterministic on test set")
    # # plt.legend()
    # plt.savefig("pareto_front_{}_{}_{}_{}_{}.png".format(cfg.architecture, cfg.dataset, sampler, function_string,
    #                                                      one_batch_string), bbox_inches="tight")
    # ax.zaxis.set_rotate_label(False)
    # ax.set_zlabel('$\sigma$', fontsize=35, rotation=0)
    # plt.show()
    # plt.savefig("pareto_front_{}_{}_{}_{}_{}.png".format(cfg.architecture, cfg.dataset, sampler, function_string,
    #                                                      one_batch_string), bbox_inches="tight")
    ##################### NOw we compare wit the deteminstic for the test set ##############################################

    # if functions == 1:
    #     g = optuna.visualization.plot_pareto_front(study, target_names=["Stochastic Performance", "Differce with Det."])
    #     g.update_layout(
    #         title=dict(text="{} {} {}".format(cfg.architecture, cfg.dataset,sampler), font=dict(size=20), automargin=True, yref='paper')
    #
    #     )
    #     g.show()
    # if functions == 2:
    #     g = optuna.visualization.plot_pareto_front(study, target_names=["Stochastic Performance", "Pruning rate"])
    #     g.update_layout(
    #         title=dict(text="{} {} {}".format(cfg.architecture, cfg.dataset,sampler), font=dict(size=20), automargin=True, yref='paper')
    #
    #     )
    #     g.show()

    # net = get_model(cfg)
    # train, val, testloader = get_datasets(cfg)
    #
    # dense_performance = test(net, use_cuda=True, testloader=testloader, verbose=0)

    ######################### testing one net on thes test set #########################################################

    # pruned_model = copy.deepcopy(net)
    # cfg.amount = best_pruning_rate
    # prune_function(pruned_model, cfg)
    # remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # # Add small noise just to get tiny variations of the deterministic ase
    # det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)
    # det_performance_val = test(pruned_model, use_cuda=True, testloader=val, verbose=0)
    #
    # names, weights = zip(*get_layer_dict(net))
    # number_of_layers = len(names)
    # sigma_per_layer = dict(zip(names, [best_sigma] * number_of_layers))
    # print("Deterministic performance on test set = {}".format(det_performance))
    # print("Deterministic performance on val set = {}".format(det_performance_val))
    # stochastic_performance = []
    #
    # performance_of_models = []
    # performance_of_models_val = []
    #
    # for individual_index in range(10):
    #     ############### Here I ask for pr and for sigma ###################################
    #
    #     current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
    #     # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
    #     prune_function(current_model, cfg)
    #
    #     remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
    #     stochastic_performance = test(current_model, use_cuda=True, testloader=testloader, verbose=0)
    #     stochastic_performance_val = test(current_model, use_cuda=True, testloader=val, verbose=0)
    #
    #     performance_of_models_val.append(stochastic_performance_val)
    #     # Dense stochastic performance
    #     performance_of_models.append(stochastic_performance)

    #
    # performance_of_models = np.array(performance_of_models)
    # median = np.median(performance_of_models)
    # print("Median accuracy of population: {}".format(median))
    # fitness_function_median = objective_function(median,det_performance, best_pruning_rate)
    # print("Fintness function of the median on test set: {}".format(fitness_function_median))
    # ##########  val functions###########################
    # performance_of_models_val= np.array(performance_of_models_val)
    # median = np.median(performance_of_models_val)
    # print("Median accuracy of population valset: {}".format(median))
    # fitness_function_median = objective_function(median,det_performance_val, best_pruning_rate)
    # print("Fintness function of the median on val set: {}".format(fitness_function_median))


def run_pr_sigma_search_for_cfg(cfg, arg):
    one_batch = False  # arg["one_batch"]
    sampler = arg["sampler"]
    log_sigma = arg["log_sigma"]
    number_of_trials = arg["trials"]
    use_population = True if cfg["population"] > 1 else False

    if sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop", n_startup_trials=10, popsize=10, inc_popsize=2)
    else:
        sampler = optuna.samplers.TPESampler()

    # # sampler = optuna.samplers.CmaEsSampler(n_startup_trials=10,popsize=4)
    # vj
    # sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="stochastic-global-pr-and-sigma-optimisation-{}-{}-{}".format(
                                    cfg.architecture,
                                    cfg.dataset, sampler),
                                storage="sqlite:///find_pr_sigma_database_{}_{}_{}_{}.dep".format(cfg.architecture,
                                                                                                  cfg.dataset, sampler,
                                                                                                  one_batch),
                                load_if_exists=True)

    study.optimize(
        lambda trial: find_pr_sigma_for_dataset_architecture_one_shot_GMP(trial, cfg, one_batch, use_population,
                                                                          use_log_sigma=log_sigma),
        n_trials=args["trials"])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    best_sigma = trial.params["sigma"]
    best_pruning_rate = trial.params["pruning_rate"]
    net = get_model(cfg)
    train, val, testloader = get_datasets(cfg)

    dense_performance = test(net, use_cuda=True, testloader=testloader, verbose=0)
    pruned_model = copy.deepcopy(net)
    cfg.amount = best_pruning_rate
    prune_function(pruned_model, cfg)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # Add small noise just to get tiny variations of the deterministic ase
    det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)
    det_performance_val = test(pruned_model, use_cuda=True, testloader=val, verbose=0)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [best_sigma] * number_of_layers))
    print("Deterministic performance on test set = {}".format(det_performance))
    print("Deterministic performance on val set = {}".format(det_performance_val))
    stochastic_performance = []

    performance_of_models = []
    performance_of_models_val = []
    for individual_index in range(10):
        ############### Here I ask for pr and for sigma ###################################

        current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(current_model, cfg)

        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        stochastic_performance = test(current_model, use_cuda=True, testloader=testloader, verbose=0)
        stochastic_performance_val = test(current_model, use_cuda=True, testloader=val, verbose=0)

        performance_of_models_val.append(stochastic_performance_val)
        # Dense stochastic performance
        performance_of_models.append(stochastic_performance)

    performance_of_models = np.array(performance_of_models)
    median = np.median(performance_of_models)
    print("Median accuracy of population: {}".format(median))
    fitness_function_median = objective_function(median, det_performance, best_pruning_rate)
    print("Fintness function of the median on test set: {}".format(fitness_function_median))
    ##########  val functions###########################
    performance_of_models_val = np.array(performance_of_models_val)
    median = np.median(performance_of_models_val)
    print("Median accuracy of population valset: {}".format(median))
    fitness_function_median = objective_function(median, det_performance_val, best_pruning_rate)
    print("Fintness function of the median on val set: {}".format(fitness_function_median))

    print("Manual pr = 0.9 sigma = 0.005 #################################")
    net = get_model(cfg)
    print("Dense performance on test set = {}".format(dense_performance))
    train, val, testloader = get_datasets(cfg)

    dense_performance = test(net, use_cuda=True, testloader=testloader, verbose=0)
    pruned_model = copy.deepcopy(net)
    cfg.amount = 0.9
    prune_function(pruned_model, cfg)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)

    # Add small noise just to get tiny variations of the deterministic case
    det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [0.005] * number_of_layers))
    print("Deterministic performance on test set = {}".format(det_performance))
    performance_of_models = []

    for individual_index in range(5):
        ############### Here I ask for pr and for sigma ###################################

        current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(current_model, cfg)

        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        stochastic_performance = test(current_model, use_cuda=True, testloader=testloader, verbose=0)
        # Dense stochastic performance
        performance_of_models.append(stochastic_performance)

    performance_of_models = np.array(performance_of_models)
    median = np.median(performance_of_models)
    print("Median accuracy of population: {}".format(median))
    fitness_function_median = objective_function(median, det_performance, best_pruning_rate)
    print("Fintness function of the median on test set: {}".format(fitness_function_median))


def find_pr_sigma_for_dataset_architecture_one_shot_GMP(trial: optuna.trial.Trial, cfg, one_batch=True,
                                                        use_population=True, use_log_sigma=False) -> np.ndarray:
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = get_model(cfg)
    train, val_loader, test_loader = get_datasets(cfg)

    # dense_performance = test(net, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    if use_log_sigma:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01, log=True)
    else:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01)
    sample_pruning_rate = trial.suggest_float("pruning_rate", 0.3, 0.99)

    # def objective_function(stochastic_performance,deter_performance, pruning_rate):
    #     return ((stochastic_performance - deter_performance)) * pruning_rate

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [sample_sigma] * number_of_layers))
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.amount = sample_pruning_rate

    pruned_model = copy.deepcopy(net)
    prune_function(pruned_model, cfg_copy)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)

    # Add small noise just to get tiny variations of the deterministic case
    det_performance = test(pruned_model, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    print("Det performance: {}".format(det_performance))

    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])
    # sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    if use_population:
        performance_of_models = []
        for individual_index in range(10):
            ############### Here I ask for pr and for sigma ###################################

            current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
            # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
            prune_function(current_model, cfg_copy)

            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
            # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
            stochastic_performance = test(current_model, use_cuda=True, testloader=val_loader, verbose=0,
                                          one_batch=one_batch)
            # Dense stochastic performance
            performance_of_models.append(stochastic_performance)
        performance_of_models = np.array(performance_of_models)
        median = np.median(performance_of_models)
        print("Median of population performance: {}".format(median))
        average_difference_performance = det_performance - performance_of_models
        fitness_function_median = objective_function(median, det_performance, sample_pruning_rate)
        # fitness_function_vector = np.array(list(map()))objective_function(performance_of_models, det_performance,sample_pruning_rate)
        # average_fitness_function = fitness_function_vector.mean()
        return fitness_function_median
    else:
        stochastic_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(stochastic_model, cfg_copy)

        remove_reparametrization(stochastic_model, exclude_layer_list=cfg.exclude_layers)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        stochastic_performance = test(stochastic_model, use_cuda=True, testloader=val_loader, verbose=0,
                                      one_batch=one_batch)
        fitness_function_median = objective_function(stochastic_performance, det_performance, sample_pruning_rate)
        print("Stochastic performance: {}".format(stochastic_performance))
        return fitness_function_median

    # Here is where I transfer the mask from the pruned stochastic model to the
    # original weights and put it in the ranking
    # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)


def calibrate(trial: optuna.trial.Trial) -> np.ndarray:
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = get_model(cfg)
    # sigma_add = trial.suggest_float("sigma_add", 0.0001, 0.1)
    sigma_add = cfg.sigma
    sigma_mul = trial.suggest_float("sigma_mul", 0.1, 1)

    # Prune original
    pruned_original = copy.deepcopy(net)
    weights_prune = weights_to_prune(pruned_original)
    prune.global_unstructured(
        weights_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    remove_reparametrization(model=pruned_original)
    vector_original = nn.utils.parameters_to_vector(pruned_original.parameters())
    binary_vector_original = vector_original == 0
    average_loss = []
    for i in range(20):
        add_noise_model = copy.deepcopy(net)
        mul_noise_model = copy.deepcopy(net)
        add_noise_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma_add))
        mul_noise_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma_mul))

        # Pruning of the  noisy models
        # Additive
        weights_prune = weights_to_prune(add_noise_model)
        prune.global_unstructured(
            weights_prune,
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount
        )
        remove_reparametrization(model=add_noise_model)
        # Geometric
        weights_prune = weights_to_prune(mul_noise_model)
        prune.global_unstructured(
            weights_prune,
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount
        )
        remove_reparametrization(model=mul_noise_model)
        #
        vector_added_noise = nn.utils.parameters_to_vector(add_noise_model.parameters())
        binary_vector_add_noise = vector_added_noise == 0
        vector_mul_noise = nn.utils.parameters_to_vector(mul_noise_model.parameters())
        binary_vector_mul_noise = vector_mul_noise == 0

        different_add_noise = torch.bitwise_xor(binary_vector_original, binary_vector_add_noise).sum()

        different_mul_noise = torch.bitwise_xor(binary_vector_original, binary_vector_mul_noise).sum()
        print(f"Number of changes by gaussian {different_add_noise}")
        print(f"Number of changes by geometric gaussian {different_mul_noise}")
        # I use the sum because I don't care where (in the vector) each noise changes specifically
        loss = ((different_add_noise - different_mul_noise) ** 2).item()
        #     average_loss = average_loss + (loss - average_loss) / (i + 1)
        average_loss.append(loss)
    print(f"Number of changes by gaussian {different_add_noise}")
    print(f"Number of changes by geometric gaussian {different_mul_noise}")

    return np.mean(average_loss)


def noise_calibration(cfg: omegaconf.DictConfig):
    # distributions = {
    #     "sigma_add": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
    #     "sigma_mul": optuna.distributions.FloatDistribution(0.1, 1, log=True),
    # }
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="noise-calibration",
                                storage="sqlite:///noise_cal_database.dep", load_if_exists=True)
    study.optimize(calibrate, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # fig1 = optuna.visualization.plot_optimization_history(study)
    # fig2 = optuna.plot_intermediate_values(study)
    # fig3 = optuna.plot_param_importances(study)
    # fig4 = optuna.contour_plot(study, params=["sigma_add", "sigma_mul"])
    #
    # fig1.savefig("data/figures/opt_history.png")
    # fig2.savefig("data/figures/intermediate_values.png")
    # fig3.savefig("data/figures/para_importances.png")
    # fig4.savefig("data/figures/contour_plot.png")


################################# Layer importance experiments ######################################
def plot_layer_experiments(model, exp_type="a"):
    layers = get_layer_dict(model)
    layers.reverse()
    layer_names, weights = zip(*layers)

    layer_names = list(layer_names)
    weights = list(weights)
    # layer_names.reverse()
    # weights.reverse()
    prunings_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    prunings_percentages.reverse()

    if exp_type == "a":
        matrix = np.load("data/layer_exp_A.npy")
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names)), labels=layer_names)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by depth (ascending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        plt.gcf().set_size_inches(5.1, 5.1)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/layer_V_prune_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
    if exp_type == "b":
        matrix = np.loadtxt("data/layer_exp_B.txt")
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        layer_names_by_n, _ = zip(*sorted_layer_names)
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names_by_n)), labels=layer_names_by_n)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by # parameters \n (descending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        # cbar.ax.tick_params(labelsize=7)

        plt.gcf().set_size_inches(5.5, 5)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        # data / figures / cumsum.layer_V_prune_traditional_pruned_
        # {result.tm_hour} - {result.tm_min}.pdf
        # data / figures / cumsum.layer_V_prune_
        # {result.tm_hour} - {result.tm_min}.pdf
        plt.savefig(f"data / figures / cumsum.layer_V_prune_traditional_pruned_{result.tm_hour} - {result.tm_min}.pdf")
        plt.close()


def layer_importance_experiments(cfg, model, use_cuda, test_loader, type_exp="a"):
    layers = get_layer_dict(model)
    layers.reverse()
    layer_names, weights = zip(*layers)

    # layer_names = list(layer_names)
    # weights = list(weights)
    # layer_names.reverse()
    # weights.reverse()
    prunings_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    prunings_percentages.reverse()
    matrix = np.zeros((len(layers), len(prunings_percentages)))

    if type_exp == "a":
        print("\n######################### EXPERIMENT A ##################################\n")
        for i, layer_tuple in enumerate(layers):
            for j, pruning_percentage in enumerate(prunings_percentages):
                current_model = copy.deepcopy(model)
                name, _ = layer_tuple
                weights = [get_weights_of_layer_name(current_model, layer_name=name)]
                prune.global_unstructured(
                    weights,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage
                )
                remove_reparametrization(current_model, name_module=name)
                accuracy = float(test(current_model, use_cuda=use_cuda, testloader=test_loader))
                # matrix[i, j] = int((i+j))
                matrix[i, j] = accuracy
                print(f"\n{np.matrix(matrix)}")
        np.save("data/layer_exp_A.npy", matrix)
        # matrix = np.random.randn(len(layers),len(prunings_percentages))
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names)), labels=layer_names)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by depth (ascending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        plt.gcf().set_size_inches(5.1, 5.1)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/traditional_trained_layer_V_prune_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
        percent_index = matrix.argmax(axis=1)
        best_percentage_for_layers = {}
        for i, name in enumerate(layer_names):
            best_percentage_for_layers[name] = prunings_percentages[percent_index[i]]
        print(f"Best percentage by layer {best_percentage_for_layers}")
        with open(f"data/best_per_layer_{cfg.architecture}_traditional_solution.pkl", "wb") as f:
            pickle.dump(best_percentage_for_layers, f)

    if type_exp == "b":
        # Organise the layers to accumulate with the number of weights
        # with open(f"data/best_per_layer_{cfg.architecture}.pkl", "rb") as f:
        #     prune_rate_per_layer = pickle.load(f)
        #
        print("\n######################### EXPERIMENT B ##################################\n")
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        # sorted_layer_names
        # Then im going to prune all the layers up to layer i with the pruning rate obtained from the type a experiment.
        for i, layer_tuple in enumerate(sorted_layer_names):
            for j, pruning_percentage in enumerate(prunings_percentages):
                current_model = copy.deepcopy(model)

                sub_layers = sorted_layer_names[i:]
                weights = [get_weights_of_layer_name(current_model, layer_name=n) for n, w in sub_layers]
                # name, _ = layer_tuple
                # weights = get_weights_of_layer_name(current_model, layer_name=name)
                prune.global_unstructured(
                    weights,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage
                )
                for n, w in sub_layers:
                    remove_reparametrization(current_model, name_module=n)
                accuracy = float(test(current_model, use_cuda=use_cuda, testloader=test_loader))
                # matrix[i, j] = int((i+j))
                matrix[i, j] = accuracy
                print(f"\n{np.matrix(matrix)}")
        np.save("data/layer_exp_B.npy", matrix)
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        layer_names_by_n, _ = zip(*sorted_layer_names)
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names_by_n)), labels=layer_names_by_n)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by # parameters \n (descending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        # cbar.ax.tick_params(labelsize=7)

        plt.gcf().set_size_inches(5.5, 5)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/cumsum.layer_V_prune_traditional_pruned_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
    #     # Now I'm going to do something similar to the above but just prune with the optimal rate


def run_layer_experiment(cfg):
    use_cuda = torch.cuda.is_available()
    if cfg.architecture == "resnet18" and "csghmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models import ResNet18
        net = ResNet18()
    if cfg.use_wandb:
        # os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%M:%S")
        # wandb.init(
        #     entity="luis_alfredo",
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     project="sparse_ensemble",
        #     name="layer_importance_{}".format(now),
        #     reinit=True,
        #     save_code=True,
        # )
        pass
    trainloader, valloader, testloader = get_datasets(cfg)
    load_model(net, cfg.solution)

    # plot_layer_experiments(net, exp_type="a")
    # layer_importance_experiments(cfg, net, use_cuda, testloader, type_exp="a")
    # plot_layer_experiments(net, exp_type="a")
    # layer_importance_experiments(cfg, net, use_cuda, testloader, type_exp="b")
    plot_layer_experiments(net, exp_type="b")


def get_best_pruning_strategy(cfg: omegaconf.DictConfig):
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )
    ####################### This section does compute all layer combinations of the small layers that ResNet18 has.j
    # net = None
    # if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
    #     net = ResNet18()
    # else:
    #     from alternate_models.resnet import ResNet18
    #     net = ResNet18()
    # load_model(net, cfg.solution)
    # layers = get_layer_dict(net)
    # layers.reverse()
    # layer_names, weights = zip(*layers)
    # # fifth element
    # up_until = 4
    # count = lambda w: w.nelement()
    # number_of_elements = list(map(count, weights))
    # sorted_by_n = np.argsort(number_of_elements)
    # sorted_layer_names = [layers[g] for g in sorted_by_n]
    # high_pruned = sorted_layer_names[:up_until + 1]
    # important_layer = "layer3.1.conv1"
    # not_high_pruned = dict(set(sorted_layer_names).difference(set(high_pruned)))
    # del not_high_pruned[important_layer]
    # all_groups = all_subsets(list(not_high_pruned.keys()))
    # with open(f"data/groups_small_layers_{cfg.architecture}.plk", "wb") as f:
    #     pickle.dump(tuple(all_groups), f)

    with open(f"data/groups_small_layers_{cfg.architecture}.plk", "rb") as f:
        all_groups = pickle.load(f)
    ################################################################

    study = optuna.create_study(directions=["maximize", "maximize"], pruner=pruner, study_name="Smart pruned")
    study.optimize(partial(optim_function_intelligent_pruning, cfg=cfg), n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trials = study.best_trials
    with open(f"data/best_trials_smart_pruning_{cfg.architecture}.plk", "wb") as f:
        pickle.dump(trials, f)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")
    trial_with_highest_sparsity = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest sparsity: ")
    print(f"\tnumber: {trial_with_highest_sparsity.number}")
    print(f"\tparams: {trial_with_highest_sparsity.params}")
    print(f"\tvalues: {trial_with_highest_sparsity.values}")

    optuna.visualization.plot_pareto_front(study, target_names=["Accuracy", "Compression ratio"])


def get_intelligent_pruned_network(cfg):
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet
        net = ResNet18()
    trainloader, valloader, testloader = get_datasets(cfg)
    load_model(net, cfg.solution)
    layers = get_layer_dict(net)
    layers.reverse()
    layer_names, weights = zip(*layers)
    pr = 0.9
    # fifth element
    up_until = 4
    count = lambda w: w.nelement()
    number_of_elements = list(map(count, weights))
    sorted_by_n = np.argsort(number_of_elements)
    sorted_layer_names = [layers[g] for g in sorted_by_n]
    # I reverse the layer in order to get the layers with the majority of the parameters first.
    sorted_layer_names.reverse()

    base_model = copy.deepcopy(net)
    high_pruned = sorted_layer_names[:up_until + 1]
    weights = [get_weights_of_layer_name(base_model, layer_name=n) for n, w in high_pruned]
    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pr
    )

    for n, w in high_pruned:
        remove_reparametrization(base_model, name_module=n)
    print("pruning the following layers with {} pruning rate".format(pr))
    for name in high_pruned:
        print(f"\t{name}")
    base_model_accuracy = test(base_model, use_cuda, testloader)
    print(f"Accuracy of the base model: {base_model_accuracy}")
    print(f"Base model has compression ratio of {count_zero_parameters(base_model) / count_parameters(base_model)}")
    print("Now i'm going to prune all the other layers combined")
    important_layer = "layer3.1.conv1"
    not_high_pruned = dict(set(sorted_layer_names).difference(set(high_pruned)))
    del not_high_pruned[important_layer]
    pruning_rates = [0.4]
    for pr in pruning_rates:
        full_model = copy.deepcopy(base_model)
        weights = [get_weights_of_layer_name(full_model, layer_name=n) for n, w in not_high_pruned.items()]
        prune.global_unstructured(
            weights,
            pruning_method=prune.L1Unstructured,
            amount=pr
        )
        print(f"Pruning rate {pr} for layers {not_high_pruned}")
        test_acc = test(full_model, use_cuda, testloader)
        for n, w in not_high_pruned.items():
            remove_reparametrization(full_model, name_module=n)
        if pr == 0.4:
            torch.save(full_model.state_dict(), "trained_models/cifar10/smart_pruned.pth")
        print("Test performance for the previous pruned model")
        print(f"{test_acc}")
        print(f"Model has compression ratio of {count_zero_parameters(full_model) / count_parameters(full_model)}")


def prune_function(net, cfg, pr_per_layer=None):
    target_sparsity = cfg.amount
    if cfg.pruner == "global":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    if cfg.pruner == "manual":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner="manual", pr_per_layer=pr_per_layer)
        individual_prs_per_layer = prune_with_rate(net, target_sparsity,
                                                   exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                   pruner="lamp", return_pr_per_layer=True)
        if cfg.use_wandb:
            log_dict = {}
            for name, elem in individual_prs_per_layer.items():
                log_dict["individual_{}_pr".format(name)] = elem
            wandb.log(log_dict)
    if cfg.pruner == "lamp":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)


def prune_with_rate(net: torch.nn.Module, amount: typing.Union[int, float], pruner: str = "erk",
                    type: str = "global",
                    criterion:
                    str =
                    "l1", exclude_layers: list = [], pr_per_layer: dict = {}, return_pr_per_layer: bool = False,
                    is_stochastic: bool = False, noise_type: str = "", noise_amplitude=0):
    if type == "global":
        print("Exclude layers in prun_with_rate:{}".format(exclude_layers))
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        print("Length of weigths to prune:{}".format(len(weights))
              )
        if criterion == "l1":
            prune.global_unstructured(
                weights,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        if criterion == "l2":
            prune.global_unstructured(
                weights,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2
            )
    elif type == "layer-wise":
        from layer_adaptive_sparsity.tools.pruners import get_modules, get_weights, weight_pruner_loader
        if pruner == "lamp":
            pruner = weight_pruner_loader(pruner)
            if return_pr_per_layer:
                return pruner(model=net, amount=amount, exclude_layers=exclude_layers,
                              return_amounts=return_pr_per_layer)
            else:
                pruner(model=net, amount=amount, exclude_layers=exclude_layers, is_stochastic=is_stochastic,
                       noise_type=noise_type, noise_amplitude=noise_amplitude)
        if pruner == "erk":
            pruner = weight_pruner_loader(pruner)
            pruner(model=net, amount=amount, exclude_layers=exclude_layers)
            # _, amount_per_layer, _, _ = erdos_renyi_per_layer_pruning_rate(model=net, cfg=cfg)
            # names, weights = zip(*get_layer_dict(net))
            # for name, module in net.named_modules():
            #     if name in exclude_layers or name not in names:
            #         continue
            #     else:
            #         prune.l1_unstructured(module, name="weight", amount=float(amount_per_layer[name]))
        if pruner == "manual":
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.l1_unstructured(module, name="weight", amount=float(pr_per_layer[name]))
    elif type == "random":
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        if criterion == "l1":
            prune.random_structured(
                weights,
                # pruning_method=prune.L1Unstructured,
                amount=amount
            )


    else:
        raise NotImplementedError("Not implemented for type {}".format(type))


def optim_function_intelligent_pruning(trial: optuna.trial.Trial, cfg) -> tuple:
    # Pruning rate large layers
    pruning_large_layers = trial.suggest_float("Pr_big_layers", 0.8, 0.99)
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet18
        net = ResNet18()
    trainloader, valloader, testloader = get_datasets(cfg)
    load_model(net, cfg.solution)
    n = cfg.population
    layers = get_layer_dict(net)
    layers.reverse()
    layer_names, weights = zip(*layers)
    # fifth element
    up_until = 4
    count = lambda w: w.nelement()
    number_of_elements = list(map(count, weights))
    sorted_by_n = np.argsort(number_of_elements)
    sorted_layer_names = [layers[g] for g in sorted_by_n]
    sorted_layer_names.reverse()
    base_model = copy.deepcopy(net)
    high_pruned = sorted_layer_names[:up_until + 1]
    weights = [get_weights_of_layer_name(base_model, layer_name=n) for n, w in high_pruned]
    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pruning_large_layers
    )
    for n, w in high_pruned:
        remove_reparametrization(base_model, name_module=n)
    # # Now I need to select how many layers of the other layers I'm going to group to prune
    # important_layer = "layer3.1.conv1"
    # not_high_pruned =dict( set(sorted_layer_names).difference(set(high_pruned)))
    # del not_high_pruned[important_layer]
    # all_groups = all_subsets(list(not_high_pruned.values()))
    with open(f"data/groups_small_layers_{cfg.architecture}.plk", "rb") as f:
        all_groups = pickle.load(f)
    # Layers Im going to acctually prune
    index = trial.suggest_int("all_groups index", 0, len(all_groups))
    layers_to_prune = all_groups[index]
    pr_for_this = trial.suggest_float("Pr_small_layers", 0.2, 0.7)
    weights = [get_weights_of_layer_name(base_model, layer_name=n) for n in layers_to_prune]
    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pr_for_this
    )
    for n in layers_to_prune:
        remove_reparametrization(base_model, name_module=n)
    test_accuracy = test(base_model, use_cuda, testloader)

    return test_accuracy, count_zero_parameters(base_model) / count_parameters(base_model)


######################################################################################################
def get_cifar_datasets(cfg: omegaconf.DictConfig):
    if cfg.dataset == "cifar10":
        # data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"
        current_directory = Path().cwd()
        data_path = "./datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
            data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

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

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - 5000, 5000])

        trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                                  num_workers=cfg.num_workers)
        val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
        return trainloader, val_loader, testloader
    if cfg.dataset == "cifar100":
        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\cifar100"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                                  num_workers=cfg.num_workers)
        val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
        return trainloader, val_loader, testloader


def create_beton_database_ImageNet():
    current_directory = Path().cwd()
    data_path = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\MNIST"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "datasets"
    traindir = data_path + 'imagenet/' + 'train'
    valdir = data_path + 'imagenet/' + 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)


def get_datasets(cfg: omegaconf.DictConfig):
    if "cifar" in cfg.dataset:
        return get_cifar_datasets(cfg)
    if "mnist" == cfg.dataset:
        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\MNIST"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transfos = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.MNIST(root=data_path,
                                              train=True,
                                              transform=transfos,
                                              download=True)

        testset = torchvision.datasets.MNIST(root=data_path,
                                             train=False,
                                             transform=transfos
                                             )

        MNIST_train, MNIST_val = random_split(trainset, [len(trainset) - 5000, 5000])

        trainloader = torch.utils.data.DataLoader(
            MNIST_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        valloader = torch.utils.data.DataLoader(
            MNIST_val, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        # testset = torchvision.datasets.CIFAR10(
        #     root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        return trainloader, valloader, testloader

    if 'imagenet' == cfg.dataset:
        # cfg.dataset="cifar10"
        # return get_cifar_datasets(cfg)
        # Excerpt take from https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126
        # Data loading code

        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data/"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\mnist"
        elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
            data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets/"
        traindir = data_path + '/imagenet/' + 'train'
        testdir = data_path + '/imagenet/' + 'val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        whole_train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        print(f"Length of dataset: {len(whole_train_dataset)}")

        train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset, [1231167, 50000])

        full_test_dataset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        big_test, small_test = torch.utils.data.random_split(full_test_dataset, [len(full_test_dataset) - 10000, 10000])

        # This code is to transform it into the "fast" format of ffcv

        # my_dataset = val_dataset
        # write_path = data_path + "imagenet/valSplit_dataset.beton"

        # For the validation set that I use to recover accuracy

        # # Pass a type for each data field
        # writer = DatasetWriter(write_path, {
        #     # Tune options to optimize dataset size, throughput at train-time
        #     'image': RGBImageField(
        #         max_resolution=256,
        #         jpeg_quality=90
        #     ),
        #     'label': IntField()
        # })
        # # Write dataset
        # writer.from_indexed_dataset(my_dataset)

        # For the validation set that I use to recover accuracy

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, sampler=None)
        if cfg.length_test == "small":
            test_loader = torch.utils.data.DataLoader(
                small_test,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
        if cfg.length_test == "big":
            test_loader = torch.utils.data.DataLoader(
                big_test,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
        if cfg.length_test == "whole":
            test_loader = torch.utils.data.DataLoader(
                full_test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader

    if 'small_imagenet' == cfg.dataset:

        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\mnist"
        elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
            data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "/home/luisaam/Documents/PhD/data"
        from test_imagenet import load_small_imagenet
        trainloader, valloader, testloader = load_small_imagenet(
            {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
             "num_workers": cfg.num_workers, "batch_size": cfg.batch_size,"resolution":cfg.input_resolution})
        return trainloader, valloader, testloader

    if 'tiny_imagenet' == cfg.dataset:
        from test_imagenet import load_tiny_imagenet

        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data/"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\mnist"
        elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
            data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "/home/luisaam/Documents/PhD/data"
        traindir = data_path + '/tiny_imagenet_200/' + 'train'
        testdir = data_path + '/tiny_imagenet_200/' + 'val'
        cfg.traindir = traindir
        cfg.valdir = testdir
        return load_tiny_imagenet(dict(cfg))


def main(cfg: omegaconf.DictConfig):
    print("torch version: {}".format(torch.__version__))
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet18
        net = ResNet18()
    trainloader, valloader, testloader = get_datasets(cfg)
    state_dict = torch.load(cfg.solution)
    load_model(net, cfg.solution)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)
    weights_to_prune = weights_to_prune(pruned_original)
    prune.global_unstructured(
        weights_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pop.append(pruned_original)
    pruned_performance.append(pruned_original_performance)
    stochastic_dense_performances.append(original_performance)
    # pp = pprint.PrettyPrinter(ident=4)
    for n in range(N):
        current_model = copy.deepcopy(net)
        if cfg.noise == "gaussian":
            current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))
        elif cfg.noise == "geogaussian":
            current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, testloader)

        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)
        prune.global_unstructured(
            weights_to_prune(current_model),
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))
        stochastic_pruned_performance = test(current_model, use_cuda, testloader)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
        current_model.cpu()
        pop.append(current_model)
    # proba_function = get_proba_function(cfg.amount, count_parameters(net))
    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)
    pruned_performance = np.array(pruned_performance)
    stochastic_dense_performances = np.array(stochastic_dense_performances)
    np.save("data/population_data/performances_{}.npy".format(cfg.noise), pruned_performance)
    for i, model in enumerate(pop):
        with open("data/population_data/{}/model_{}.pkl".format(cfg.noise, i), "wb") as f:
            pickle.dump(model, f)
    del pop
    cutoff = original_performance - 2

    fig = plt.figure()
    plt.axhline(y=original_performance, color="k", linestyle="-", label="Original dense pruned_performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    if cfg.noise == "geogaussian":
        plt.title("CIFAR10 Geometric Gaussian Noise", fontsize=20)

    if cfg.noise == "gaussian":
        plt.title("CIFAR10 Additive Gaussian Noise", fontsize=20)

    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    p1 = None
    for i, element in enumerate(pruned_performance[ranked_index]):
        if i == index_of_pruned_original:
            p1 = plt.scatter(i, element, c="g", marker="o")
        else:
            pruned_point = plt.scatter(i, element, c="b", marker="x")
            stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original:
            continue
            # plt.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = plt.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense), p1],
               ['Pruned stochastic', 'Dense stochastic', "original model pruned"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    result = time.localtime(time.time())
    plt.savefig(
        f"data/figures/comparison_{cfg.noise}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop_{cfg.population}_{result.tm_hour}"
        f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/comparison_{cfg.noise}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop_{cfg.population}"
                f"_{result.tm_hour}"
                f"-{result.tm_min}.png")


def manual_train(model: nn.Module, train_loader, val_loader, save_name, epochs, learning_rate, is_cyclic=False,
                 cosine_schedule=False,
                 lr_peak_epoch=5, weight_decay=1e-4, momentum=0.9, grad_clip=0.1, optim="sgd", solution_accuracy=0):
    model.cuda()
    if optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=True)
    elif optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
    iters_per_epoch = len(train_loader)
    lr_scheduler = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                             [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                             [0, 1, 0])
    if is_cyclic:
        # lr_scheduler = LambdaLR(optimizer, lr_schedule.__getitem__)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs,
                                                           steps_per_epoch=len(train_loader))

    elif cosine_schedule:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:

        lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    # trainer = create_supervised_trainer(model, optimizer, criterion, device="cuda")
    val_metrics = {
        "accuracy": igm.Accuracy(),
        "nll": igm.Loss(criterion)
    }
    accuracy = igm.Accuracy()

    if solution_accuracy:
        best_accuracy = solution_accuracy
    else:
        best_accuracy = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # first forward-backward step
            predictions = model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)
            loss.backward()
            accuracy.update((predictions, target))
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()
            # smooth_CE(model(data), target).backward()
            # optimizer.second_step(zero_grad=True)
            # if mask:
            #     mask.apply_mask()
            #     one_forward_backward_pass = mask.inference_FLOPs * 3
            #     train_flops += one_forward_backward_pass * 2
            # # L2 Regularization
            #
            # # Exp avg collection
            # _loss_collector.add_value(loss.item())
            #
            # pbar.update(1)
            # global_step += 1

            if batch_idx % 10 == 0:
                acc = accuracy.compute()
                print(f"Training Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                      f" {loss.item():.2f}")

        print("Validation results")
        performance = test(model, use_cuda=True, testloader=val_loader)
        msg_performance = f"{performance:.2f}".replace(".", ",")
        if performance > best_accuracy:
            torch.save(model.state_dict(), f"trained_models/{save_name}_val_accuracy={msg_performance}.pt")


def run_traditional_training(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        if cfg.solution:
            net = tv.models.resnet18()
            net.fc = nn.Linear(512, 10)
            load_model(net, cfg.solution)
        else:
            from torchvision.models import resnet18, ResNet18_Weights
            net = tv.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Linear(512, 10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                              num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)

    pre_train_perf = test(net, use_cuda, testloader)
    manual_train(net, trainloader, val_loader, save_name=f"traditional_trained_cylic_{cfg.cyclic_lr}_cosine_"
                                                         f"{cfg.cosineschedule}_optim_{cfg.optim}",
                 epochs=cfg.epochs,
                 learning_rate=cfg.lr,
                 is_cyclic=cfg.cyclic_lr, cosine_schedule=cfg.cosine_schedule, lr_peak_epoch=cfg.lr_peak_epoch,
                 optim=cfg.optim, solution_accuracy=pre_train_perf if cfg.solution else 0)
    print("TEST SET RESULTS")
    test(net, use_cuda, testloader)


def train(model: nn.Module, train_loader, val_loader, save_name, epochs, learning_rate, is_cyclic=False,
          cosine_schedule=False, lr_peak_epoch=5, weight_decay=1e-4, momentum=0.9):
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                                nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
    iters_per_epoch = len(train_loader)
    lr_schedule = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    if is_cyclic:
        # lr_scheduler = LambdaLR(optimizer, lr_schedule.__getitem__)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs,
                                                           steps_per_epoch=len(train_loader))
    else:
        lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    trainer = create_supervised_trainer(model, optimizer, criterion, device="cuda")
    val_metrics = {
        "accuracy": igm.Accuracy(),
        "nll": igm.Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device="cuda")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    print("\nFine tuning has began\n")

    # Setup engine &  logger
    def setup_logger(logger):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    # Force filename to model.pt to ease the rerun of the notebook
    disk_saver = DiskSaver(dirname="trained_models", require_empty=False)
    best_model_handler = Checkpoint(to_save={f'{save_name}': model},
                                    save_handler=disk_saver,
                                    filename_pattern="{name}.{ext}",
                                    n_saved=1)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    # Add early stopping
    es_patience = 10
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)
    setup_logger(es_handler.logger)

    # Clear cuda cache between training/testing
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    trainer.run(train_loader, max_epochs=epochs)


def get_stochastic_pruning_results_on(cfg):
    pass


def one_shot_iterative_sotchastic_pruning(cfg):
    # TODO implement the algorithm for one shot pruning wit optimization of the pruning rate and the
    #

    pass


###############################  Channel inspection
# ############################################
def channel_inspection_of_convolutional_layers(pruned_model: torch.nn.Module, exclude_layer_list=[]):
    histograms_of_0_per_layer = {}
    for name, m in pruned_model.named_modules():
        if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) and name not in exclude_layer_list:

            # input channels X output shanels X dim1 kernel X dim2 kernel
            weight_copy = m.weight.detach().clone().numpy()
            in_channels, out_channels, dim1k, dim2k = weight_copy.shape

            for output_index in range(out_channels):
                slice_weight = weight_copy[:, output_index, :, :]
                slice_weight.reshape((in_channels, dim1k * dim2k))

            # histograms_of_0_per_layer[]

            modules.append((m, "weight"))


############################### Experiments 25 of October # ############################################################

def select_eval_set(cfg: omegaconf.DictConfig, eval_set: str):
    trainloader, valloader, testloader = get_datasets(cfg)
    evaluation_set = None
    if eval_set == "test":
        evaluation_set = testloader
    elif eval_set == "val":
        evaluation_set = valloader
    elif eval_set == "train":
        evaluation_set = trainloader
    else:
        raise Exception("Invalid evaluation set {} is not valid".format(eval_set))
    return evaluation_set


def transfer_mask_rank_experiments(cfg: omegaconf.DictConfig, eval_set: str = "test"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    deterministic_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)
    pruned_original = copy.deepcopy(net)
    prune_with_rate(pruned_original, cfg.amount)
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)
    pop.append(pruned_original)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        sto_mask_transfer_model = copy.deepcopy(net)

        current_model = get_noisy_sample(net, cfg)

        det_mask_transfer_model = copy.deepcopy(current_model)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)

        prune_with_rate(current_model, amount=cfg.amount)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))

        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        deterministic_with_stochastic_mask_performance.append(test(sto_mask_transfer_model, use_cuda, evaluation_set,
                                                                   verbose=1))
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
        current_model.cpu()
        pop.append(current_model)

    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance) - 1))
    labels.extend(["sto mask transfer"] * len(deterministic_with_stochastic_mask_performance))
    pruned_performance.extend(deterministic_with_stochastic_mask_performance)
    stochastic_dense_performances.extend([1] * len(deterministic_with_stochastic_mask_performance))
    # Deterministic mask transfer to stochastic model
    labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    stochastic_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    assert len(labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                   f"{len(labels)}!={len(pruned_performance)}"
    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)

    pruned_performance = np.array(pruned_performance)
    stochastic_dense_performances = np.array(stochastic_dense_performances)
    result = time.localtime(time.time())
    # np.save(
    #     "data/population_data/performances_{}_transfer_t_{}-{}.npy".format(cfg.noise, result.tm_hour, result.tm_min),
    #     pruned_performance)
    # for i, model in enumerate(pop):
    #     with open("data/population_data/{}/model_{}.pkl".format(cfg.noise, i), "wb") as f:
    #         pickle.dump(model, f)
    del pop
    cutoff = original_performance - 2
    # plot_population_ranking_with_cutoff(cfg, original_performance, cutoff, pruned_performance, ranked_index,
    #                                     stochastic_dense_performances, index_of_pruned_original, labels, result,
    #                                     eval_set, "static")
    ################################# plotting the comparison #########################################################
    fig = plt.figure()
    original_line = plt.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")
    # plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    # if cfg.noise == "geogaussian":
    #     plt.title(f"CIFAR10 Geometric Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)
    # if cfg.noise == "gaussian":
    #     plt.title(f"CIFAR10 Additive Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)

    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    p1 = None

    for i, element in enumerate(pruned_performance[ranked_index]):
        if i == index_of_pruned_original:
            assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
                                                           f"in list {element} VS pruned performance:" \
                                                           f" {pruned_original_performance}"
            p1 = plt.scatter(i, element, c="g", marker="o")
        else:
            if labels[ranked_index[i]] == "sto mask transfer":
                sto_transfer_point = plt.scatter(i, element, c="tab:orange", marker="P")
                transfer_mask_models_points.append(sto_transfer_point)
            elif labels[ranked_index[i]] == "det mask transfer":
                det_transfer_point = plt.scatter(i, element, c="tab:olive", marker="X")
                stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
            else:
                pruned_point = plt.scatter(i, element, c="b", marker="x")
                stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # plt.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = plt.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([original_line, tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense), p1,
                tuple(transfer_mask_models_points), tuple(stochastic_with_deterministic_mask_models_points)],
               ["Original Performance", 'Pruned Stochastic', 'Dense Stochastic', "Original Accuracy", "Stochastic Mask "
                                                                                                      "" + r"$\rightarrow$"
                + " Original Weights", "Deterministic Mask" + r"$\rightarrow$" + "Stochastic Weights"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    plt.savefig(
        f"data/figures/transfers_comparison_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}"
        f"_t_{result.tm_hour}"
        f"-{result.tm_min}_{eval_set}.pdf")
    plt.savefig(f"data/figures/transfers_comparison_{cfg.noise}_sigma_{cfg.sigma}_pr_{cfg.amount}_batchSize"
                f"_{cfg.batch_size}_pop"
                f"_{cfg.population}"
                f"_t_{result.tm_hour}"
                f"-{result.tm_min}_{eval_set}.png")


def transfer_mask_rank_experiments_plot_adaptive_noise(cfg: omegaconf.DictConfig, eval_set: str = "test"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
                                     names=["layer", "q25", "q50", "q75"])

    sigma_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()

    evaluation_set = select_eval_set(cfg, eval_set)

    N = cfg.population
    pruned_performance = []
    stochastic_static_dense_performances = []
    stochastic_dynamic_dense_performances = []

    dynamic_pruned_performance = []
    stochastic_deltas = []
    sto_mask_to_ori_weights_deltas = []
    ori_mask_to_sto_weights_deltas = []

    original_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []
    original_performance = test(net, use_cuda, evaluation_set)
    pruned_original = copy.deepcopy(net)

    prune_with_rate(pruned_original, amount=cfg.amount, exclude_layers=cfg.exclude_layers)

    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)

    pruned_performance.append(pruned_original_performance)
    stochastic_static_dense_performances.append(original_performance)

    plot_labels = ["pruned original"]

    for n in range(N):
        sto_mask_transfer_model = copy.deepcopy(net)
        current_model_static_noise = get_noisy_sample(net, cfg)

        stochastic_static_dense_performance = test(current_model_static_noise, use_cuda, evaluation_set, verbose=1)
        stochastic_static_dense_performances.append(stochastic_static_dense_performance)
        current_model_dynamic_noise = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        stochastic_dynamic_dense_performance = test(current_model_static_noise, use_cuda, evaluation_set, verbose=1)
        stochastic_dynamic_dense_performances.append(stochastic_dynamic_dense_performance)

        det_mask_transfer_model = copy.deepcopy(current_model_static_noise)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)
        # print("Performance for dense model {}".format(n))
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)

        # prune current model static noise
        prune_with_rate(current_model_static_noise, cfg.amount, exclude_layers=cfg.exclude_layers)

        # prune current model dynamic noise

        prune_with_rate(current_model_dynamic_noise, cfg.amount, exclude_layers=cfg.exclude_layers)

        # Here is where I transfer the mask from the prune stochastic model to the
        # original weights and put it in the ranking

        copy_buffers(from_net=current_model_static_noise, to_net=sto_mask_transfer_model)

        # remove_reparametrization(current_model_static_noise)

        print("Performance for static stochastic pruned model {}".format(n))
        stochastic_pruned_performance = test(current_model_static_noise, use_cuda, evaluation_set, verbose=1)
        print("Performance for dynamic stochastic pruned model {}".format(n))
        stochastic_dynamic_pruned_performance = test(current_model_dynamic_noise, use_cuda, evaluation_set, verbose=1)

        # print("Performance for Transfer sto mask {} to model".format(n))
        # # original_with_sto_mask = test(sto_mask_transfer_model, use_cuda, evaluation_set, verbose=1)
        # original_with_stochastic_mask_performance.append(original_with_sto_mask)

        pruned_performance.append(stochastic_pruned_performance)
        dynamic_pruned_performance.append(stochastic_dynamic_pruned_performance)

        stochastic_deltas.append(original_performance - stochastic_pruned_performance)
        # sto_mask_to_ori_weights_deltas.append(original_performance - original_with_sto_mask)
        # ori_mask_to_sto_weights_deltas.append(original_performance - det_mask_transfer_model_performance)
    # len(pruned performance)-1 because the first one is the pruned original
    # Deterministic mask transfer to stochastic model
    plot_labels.extend(["static stochastic pruned"] * (len(pruned_performance) - 1))
    plot_labels.extend(["dynamic stochastic pruned"] * (len(dynamic_pruned_performance)))
    pruned_performance.extend(dynamic_pruned_performance)
    stochastic_static_dense_performances.extend(stochastic_dynamic_dense_performances)

    # Deterministic mask transfer to stochastic model
    # plot_labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    # pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    # stochastic_static_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.

    assert len(plot_labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                        f"{len(plot_labels)}!={len(pruned_performance)}"

    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)

    pruned_performance = np.array(pruned_performance)
    stochastic_static_dense_performances = np.array(stochastic_static_dense_performances)

    result = time.localtime(time.time())
    epsilon = []
    labels = []
    epsilon.extend(stochastic_deltas)
    labels.extend(["stochastic pruning"] * len(stochastic_deltas))
    epsilon.extend(sto_mask_to_ori_weights_deltas)
    labels.extend(["Sto. mask original weights"] * len(sto_mask_to_ori_weights_deltas))
    epsilon.extend(ori_mask_to_sto_weights_deltas)
    labels.extend(["Det. mask sto. weights"] * len(ori_mask_to_sto_weights_deltas))
    df_sigmas = [format_sigmas(cfg.sigma)] * len(labels)
    df_N = [cfg.population] * len(labels)
    df_pr = [cfg.amount] * len(labels)
    results = {"Epsilon": epsilon, "Population": df_N, "Type": labels, "Pruning Rate": df_pr, "sigma": df_sigmas}
    df = pd.DataFrame(results)

    # plot_population_ranking_with_cutoff(cfg, original_performance, cutoff=original_performance - 2,
    #                                     pruned_performance=pruned_performance, ranked_index=ranked_index,
    #                                     stochastic_static_dense_performances=stochastic_static_dense_performances,
    #                                     index_of_pruned_original=index_of_pruned_original,
    #                                     labels=labels, time_object=result, eval_set=eval_set, identifier="ADAPTIVE")
    cutoff = original_performance - 2
    ######################################### Plot the comparisons #####################################################
    fig = plt.figure()
    plt.axhline(y=original_performance, color="k", linestyle="-", label="Original dense pruned_performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    if cfg.noise == "geogaussian":
        plt.title(f"CIFAR10 Geometric Gaussian Noise $pr={pr}$", fontsize=10)
    if cfg.noise == "gaussian":
        plt.title(f"CIFAR10 Additive Gaussian Noise $pr={pr}$", fontsize=10)

    stochastic_static_models_points_dense = []
    stochastic_static_models_points_pruned = []

    stochastic_dynamic_models_points_dense = []
    stochastic_dynamic_models_points_pruned = []

    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    p1 = None

    for i, element in enumerate(pruned_performance[ranked_index]):
        if i == index_of_pruned_original:
            assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
                                                           f"in list {element} VS pruned performance:" \
                                                           f" {pruned_original_performance}"
            p1 = plt.scatter(i, element, c="g", marker="o")
        else:
            if plot_labels[ranked_index[i]] == "sto mask transfer":
                sto_transfer_point = plt.scatter(i, element, c="tab:orange", marker="P")
                transfer_mask_models_points.append(sto_transfer_point)
            elif plot_labels[ranked_index[i]] == "det mask transfer":
                det_transfer_point = plt.scatter(i, element, c="tab:olive", marker="X")
                stochastic_with_deterministic_mask_models_points.append(det_transfer_point)

            elif plot_labels[ranked_index[i]] == "static stochastic pruned":
                pruned_point = plt.scatter(i, element, c="b", marker="x")
                stochastic_static_models_points_pruned.append(pruned_point)
            elif plot_labels[ranked_index[i]] == "dynamic stochastic pruned":
                pruned_point = plt.scatter(i, element, c="b", marker="d")
                stochastic_dynamic_models_points_pruned.append(pruned_point)

    for i, element in enumerate(stochastic_static_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # plt.scatter(i, element, c="y", marker="o", label="original model performance")
        elif plot_labels[ranked_index[i]] == "static stochastic pruned":
            dense_point = plt.scatter(i, element, c="c", marker="1")
            stochastic_static_models_points_dense.append(dense_point)
        elif plot_labels[ranked_index[i]] == "dynamic stochastic pruned":
            dense_point = plt.scatter(i, element, c="c", marker="D")
            stochastic_dynamic_models_points_dense.append(dense_point)

    plt.legend([tuple(stochastic_static_models_points_pruned), tuple(stochastic_static_models_points_dense), p1,
                tuple(stochastic_dynamic_models_points_pruned), tuple(stochastic_dynamic_models_points_dense)],
               ['Pruned static stochastic', 'Dense static stochastic', "Original model pruned",
                "Pruned dynamic stochastic",
                "Dense dynamic stochastic"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    plt.savefig(
        f"data/figures/adaptive_V_static_comparison_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}"
        f"_t_{result.tm_hour}"
        f"-{result.tm_min}_{eval_set}.pdf")
    plt.savefig(f"data/figures/adaptive_V_static_comparison_{cfg.noise}_sigma_{cfg.sigma}_pr_{cfg.amount}_batchSize"
                f"_{cfg.batch_size}_pop"
                f"_{cfg.population}"
                f"_t_{result.tm_hour}"
                f"-{result.tm_min}_{eval_set}.png")
    return df


def transfer_mask_rank_experiments_no_plot(cfg: omegaconf.DictConfig):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    trainloader, valloader, testloader = get_datasets(cfg)
    N = cfg.population
    pruned_performance = []
    stochastic_deltas = []
    sto_mask_to_ori_weights_deltas = []
    ori_mask_to_sto_weights_deltas = []

    original_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []

    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)

    if cfg.pruner == "global":
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    # weights_to_prune = weights_to_prune(pruned_original)
    # prune.global_unstructured(
    #     weights_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=cfg.amount
    # )
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pruned_performance.append(pruned_original_performance)
    for n in range(N):
        sto_mask_transfer_model = copy.deepcopy(net)
        current_model = get_noisy_sample(net, cfg)
        det_mask_transfer_model = copy.deepcopy(current_model)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, testloader)
        stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        #
        if cfg.pruner == "global":
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)
        # Here is where I transfer the mask from the prune stochastic model to the
        # original weights and put it in the ranking

        copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        print("Performance for model {}".format(n))
        stochastic_pruned_performance = test(current_model, use_cuda, testloader)
        original_with_sto_mask = test(sto_mask_transfer_model, use_cuda, testloader)
        original_with_stochastic_mask_performance.append(original_with_sto_mask)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(original_performance - stochastic_pruned_performance)
        sto_mask_to_ori_weights_deltas.append(original_performance - original_with_sto_mask)
        ori_mask_to_sto_weights_deltas.append(original_performance - det_mask_transfer_model_performance)
    # len(pruned performance)-1 because the first one is the pruned original
    # Deterministic mask transfer to stochastic model

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    result = time.localtime(time.time())
    epsilon = []
    labels = []
    epsilon.extend(stochastic_deltas)
    if cfg.pruner == "global":
        labels.extend(["Stochastic GMP"] * len(stochastic_deltas))
    elif cfg.pruner == "lamp":
        labels.extend(["Stochastic LAMP"] * len(stochastic_deltas))
    epsilon.extend(sto_mask_to_ori_weights_deltas)
    labels.extend(["Sto. mask original weights"] * len(sto_mask_to_ori_weights_deltas))
    epsilon.extend(ori_mask_to_sto_weights_deltas)
    labels.extend(["Det. mask sto. weights"] * len(ori_mask_to_sto_weights_deltas))
    df_sigmas = [format_sigmas(cfg.sigma)] * len(labels)
    df_N = [cfg.population] * len(labels)
    df_pr = [cfg.amount] * len(labels)
    results = {"Epsilon": epsilon, "Population": df_N, "Type": labels, "Pruning Rate": df_pr, "sigma": df_sigmas}
    df = pd.DataFrame(results)
    # df = df.append(results,ignore_index=True)
    return df


def sigma_sweeps_transfer_mask_rank_experiments():
    cfg = omegaconf.DictConfig({
        "population": 10,
        "architecture": "resnet18",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
        "model": "5",
        "noise": "gaussian",
        # "sigma": 0.0021419609859022197,
        "sigma": 0.001,
        "amount": 0.8,
        "batch_size": 512,
        "num_workers": 1,
        "use_wandb": True
    })
    sigmas = np.linspace(start=0.001, stop=0.005, num=10)
    pruning_rates = [0.5, 0.6, 0.8, 0.83, 0.85, 0.87, 0.9, 0.95]
    for sig in sigmas:
        for pr in pruning_rates:
            cfg.sigma = float(sig)
            cfg.amount = pr

            transfer_mask_rank_experiments(cfg)


def single_statistics_of_epsilon_for_stochastic_pruning(filepath):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    # num_col = len(df["Pruning Rate"].unique())
    # num_row = len(df[sigma"].unique())
    # num_col = 3
    # num_row = 3
    # fig = plt.figure(figsize=(16, 10))
    # subfigs = fig.subfigures(num_row, 1, wspace=0.07)
    grped_bplot = sns.catplot(x='Population',
                              y='Epsilon',
                              hue="Type",
                              kind="box",
                              legend=False,
                              height=6,
                              aspect=1.3,
                              data=df)  # .set(title="Title")
    # make grouped stripplot
    grped_bplot = sns.stripplot(x='Population',
                                y='Epsilon',
                                hue='Type',
                                jitter=True,
                                dodge=True,
                                marker='o',
                                edgecolor="gray",
                                linewidth=1,
                                # palette="set2",
                                alpha=0.5,
                                data=df)
    # how to remove redundant legends in Python
    # Let us first get legend information from the plot object
    handles, labels = grped_bplot.get_legend_handles_labels()
    # grped_bplot.set_title("The title")
    plt.title("Pruning Rate 0.9" + r" $\sigma=0.001$", fontsize=12)
    # specify just one legend
    l = plt.legend(handles[:3], labels[:3])
    plt.savefig("grouped_boxplot_with_jittered_data_points_Seaborn_Python2.png", bbox_inches="tight")


def get_model(cfg: omegaconf.DictConfig):
    net = None
    if cfg.architecture == "resnet18":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = ResNet18()
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet18
                if cfg.dataset == "cifar10":
                    net = ResNet18()
                if cfg.dataset == "cifar100":
                    net = ResNet18(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet18()
                if cfg.dataset == "imagenet":
                    net = ResNet18(num_classes=1000)
                return net
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    from torchvision import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 100)
                    net.load_state_dict(cfg, solution)
                if cfg.dataset == "cifar10":
                    from torchvision.models import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)

                    # temp_dict = torch.load(cfg.solution)["net"]
                    # real_dict = {}
                    # for k, item in temp_dict.items():
                    #     if k.startswith('module'):
                    #         new_key = k.replace("module.", "")
                    #         real_dict[new_key] = item
                    # net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet18, ResNet18_Weights

                    net = resnet18()
                    temp_dict = torch.load(cfg.solution)
                    net.load_state_dict(temp_dict)

                    # Using pretrained weights:
                    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    # net = resnet18(weights="IMAGENET1K_V1")
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = ResNet18()
                load_model(net, cfg.solution)
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet18
                if cfg.dataset == "cifar10":
                    net = ResNet18()
                if cfg.dataset == "cifar100":
                    net = ResNet18(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet18()
                if cfg.dataset == "imagenet":
                    net = ResNet18(num_classes=1000)

                load_model(net, cfg.solution)

            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    from torchvision import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 100)
                    net.load_state_dict(cfg, solution)
                if cfg.dataset == "cifar10":
                    from torchvision.models import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)

                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet18, ResNet18_Weights

                    net = resnet18()
                    temp_dict = torch.load(cfg.solution)
                    net.load_state_dict(temp_dict)

                    # Using pretrained weights:
                    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    # net = resnet18(weights="IMAGENET1K_V1")

            return net
    if cfg.architecture == "vgg19":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = VGG(cfg.architecture)
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.vgg import VGG
                if cfg.dataset == "cifar100":
                    net = VGG("VGG19", num_classes=100)
                if cfg.dataset == "cifar10":
                    net = VGG("VGG19")
                if cfg.dataset == "imagenet":
                    net = VGG("VGG19", num_classes=1000)
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = VGG("VGG19")
                load_model(net, cfg.solution)
            if "alternative" == cfg.model_type:
                from alternate_models.vgg import VGG
                if cfg.dataset == "cifar100":
                    net = VGG("VGG19", num_classes=100)
                if cfg.dataset == "cifar10":
                    net = VGG("VGG19")
                load_model(net, cfg.solution)
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)
                if cfg.dataset == "cifar10":
                    net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)

            return net

    if cfg.architecture == "resnet50":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = ResNet50()
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet50
                if cfg.dataset == "cifar10":
                    net = ResNet50()
                if cfg.dataset == "cifar100":
                    net = ResNet50(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet50()
                if cfg.dataset == "imagenet":
                    net = ResNet50(num_classes=1000)
                return net
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    from torchvision import resnet50
                    net = resnet50()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 100)
                    net.load_state_dict(cfg, solution)
                if cfg.dataset == "cifar10":
                    from torchvision.models import resnet50
                    net = resnet50()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)
                    #
                    # temp_dict = torch.load(cfg.solution)["net"]
                    # real_dict = {}
                    # for k, item in temp_dict.items():
                    #     if k.startswith('module'):
                    #         new_key = k.replace("module.", "")
                    #         real_dict[new_key] = item
                    # net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet50, ResNet18_Weights

                    net = resnet50()
                    # temp_dict = torch.load(cfg.solution)
                    # net.load_state_dict(temp_dict)

                    # Using pretrained weights:
                    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    # net = resnet18(weights="IMAGENET1K_V1")
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = ResNet50()
                load_model(net, cfg.solution)
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet50
                if cfg.dataset == "cifar10":
                    net = ResNet50()
                if cfg.dataset == "cifar100":
                    net = ResNet50(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet50()
                load_model(net, cfg.solution)
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)
                if cfg.dataset == "cifar10":
                    # net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)
                    from torchvision.models import resnet50
                    net = resnet50()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)

                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet50, ResNet50_Weights
                    # Using pretrained weights:
                    net = resnet50(weights="IMAGENET1K_V1")

            return net

    else:
        raise NotImplementedError("Not implemented for architecture:{}".format(cfg.architecture))


######################################## Check functions ##########################################################
def layers_to_vector(layers_params: typing.List[torch.Tensor]):
    flatten_list = [tensor.flatten().detach() for tensor in layers_params]
    full_vector = torch.cat(flatten_list)
    return full_vector


def get_threshold_and_pruned_vector_from_pruning_rate(list_of_layers: typing.List[torch.Tensor], pruning_rate: float):
    full_vector = layers_to_vector(list_of_layers)
    magnitudes = torch.abs(full_vector)
    total = len(full_vector)
    sorted_index = torch.argsort(magnitudes)
    index_of_threshold = math.floor(total * pruning_rate)
    threshold = magnitudes[sorted_index[index_of_threshold]]
    full_vector[sorted_index[:index_of_threshold]] = 0
    number_of_zeros = len(torch.where(full_vector == 0)[0])
    assert number_of_zeros == index_of_threshold, "Length of 0s in full vector is not the same as the" \
                                                  f"index threshold: " \
                                                  f"{number_of_zeros}!=" \
                                                  f"{index_of_threshold}"

    return float(threshold), index_of_threshold, full_vector


def get_df_pruned_by_layer(names: typing.List[str], weights: typing.List[torch.nn.Module], function: typing.Callable,
                           cfg: omegaconf.DictConfig) -> typing.Tuple[pd.DataFrame, float, int, torch.Tensor]:
    elements = np.array(list(map(function, weights)))
    sorted_indexes_by_function = np.flip(np.argsort(elements))

    threshold, index_of_threshold, pruned_vector = get_threshold_and_pruned_vector_from_pruning_rate(weights,
                                                                                                     cfg.amount)
    new_weights = copy.deepcopy(weights)

    vector_to_parameters(pruned_vector, new_weights)

    layer_survived = []
    layer_died = []
    for w in [new_weights[i] for i in sorted_indexes_by_function]:
        survived = len(torch.where(w != 0)[0])
        died = w.nelement() - survived
        layer_survived.append(survived)
        layer_died.append(died)
    le_dictionary = {"Layer": [names[i] for i in sorted_indexes_by_function], "Survived": layer_survived,
                     "Pruned": layer_died}
    df = pd.DataFrame(le_dictionary)
    return df, threshold, index_of_threshold, pruned_vector


def get_df_changes_by_layer(names: typing.List[str], weights: typing.List[torch.nn.Module], vector1: torch.Tensor,

                            vector2: torch.Tensor,
                            function:
                            typing.Callable = None):
    if function:
        elements = np.array(list(map(function, weights)))
        sorted_indexes_by_function = np.flip(np.argsort(elements))

    difference_vector = torch.bitwise_xor(vector1.type(torch.bool), vector2.type(torch.bool))
    w = copy.deepcopy(weights)
    vector_to_parameters(difference_vector, w)
    number_different_per_layer = []
    #
    for diff_vec in w:
        number_different_per_layer.append(diff_vec.sum().item())
    if function:
        le_dictionary = {"Layer": [names[i] for i in sorted_indexes_by_function],
                         "Changed": [number_different_per_layer[i]
                                     for i in sorted_indexes_by_function]}
    else:
        le_dictionary = {"Layer": names, "Changed": number_different_per_layer}

    df = pd.DataFrame(le_dictionary)
    return df, w


def check_noise_impact_in_weights_by_layer(cfg):
    net = get_model(cfg)
    names, weights = zip(*get_layer_dict(net))
    N = cfg.population
    # BarPlot the original weights pruned at level cfg.amount
    number_param = lambda w: w.nelement()
    df, threshold, index_threshold, pruned_vector = get_df_pruned_by_layer(names=names, weights=weights,
                                                                           function=number_param, cfg=cfg)
    pruned_original = copy.deepcopy(net)
    prune_with_rate(pruned_original, amount=cfg.amount)
    deterministic_pruned_buffer_vector = parameters_to_vector(pruned_original.buffers())
    names_pruned, weights_pruned = zip(*get_layer_dict(pruned_original))

    # remove_reparametrization(pruned_original)
    # zeros = count_zero_parameters(pruned_original)
    # stacked_barplot(df,
    #                 x="Layer",
    #                 y1="Survived",
    #                 y2="Pruned",
    #                 ylabel="Count",

    #                 path="test_pr_{}.pdf".format(cfg.amount),
    #                 title="Pruning rate " f"{cfg.amount} " + f" Pruning Threshold={threshold}",
    #                 label1="Survived",
    #                 label2="Pruned",
    #                 rot=90,
    #                 logscale=True
    #                 )
    # plt.close()

    first_sample = get_noisy_sample(net, cfg)
    prune_with_rate(first_sample, amount=cfg.amount)
    stochastic_pruned_buffer_vector = parameters_to_vector(first_sample.buffers())
    # remove_reparametrization(first_sample)
    df_stochastic, changes_bool_vector_list = get_df_changes_by_layer(names, weights,
                                                                      deterministic_pruned_buffer_vector,
                                                                      stochastic_pruned_buffer_vector,
                                                                      number_param)
    elements = np.array(list(map(number_param, weights)))
    sorted_indexes_by_function = np.flip(np.argsort(elements))
    elements_sorted = elements[sorted_indexes_by_function]
    df_stochastic["Changed Percentage"] = df_stochastic["Changed"] / elements_sorted
    df_stochastic["Total Parameters"] = elements_sorted
    df_stochastic.to_csv("data/analysis_per_layer/weights_altered_per_layer_one_sample_pr_{}.csv".format(cfg.amount),
                         sep=",", index=False)
    fig = plt.figure()
    ax = df_stochastic.plot.bar(x='Layer', y='Changed Percentage', rot=90)
    ax.set_xlabel("Layers by number of parameters " + r"$+\;\rightarrow\;-$")
    # ax.set_yscale("log")
    plt.ylabel("Percentage")
    plt.title("Pruning rate = {}".format(cfg.amount))
    plt.tight_layout()
    plt.savefig(f"data/analysis_per_layer/changes_percentages_single_sample_pr_{cfg.amount}.pdf")
    plt.savefig(f"data/analysis_per_layer/changes_percentages_single_sample_pr_{cfg.amount}.png")
    plt.close()
    fig = plt.figure()
    ax = df_stochastic.plot.bar(x='Layer', y='Changed', rot=90)
    ax.set_xlabel("Layers by number of parameters " + r"$+\;\rightarrow\;-$")
    ax.set_yscale("log")
    plt.ylabel("Count")
    plt.title("Pruning rate = {}".format(cfg.amount) + "Static sigma ={}".format(cfg.sigma))
    plt.tight_layout()
    plt.savefig(f"data/analysis_per_layer/changes_single_sample_pr_{cfg.amount}_sigma_{cfg.sigma}.pdf")
    plt.savefig(f"data/analysis_per_layer/changes_single_sample_pr_{cfg.amount}_sigma_{cfg.sigma}.png")
    plt.close()
    return threshold


def heatmap_scaled_noise_per_layer(cfg: omegaconf.DictConfig):
    """
    This is for a experiment that uses stochastic noise only on
    one layer at the time and plots a heatmap for different pruning rates.
    :param cfg:
    :return:
    """
    pass


def iterative_stochastic_pruning(cfg: omegaconf.DictConfig):
    pass


def two_step_iterative_pruning(cfg: omegaconf.DictConfig):
    pass


def gradient_decent_on_sigma_pr():
    pass


############################################### CDF and 0 counting functions ###########################################
def count_zero_slices(conv_weight: torch.Tensor):
    out_features, in_features, H, W = conv_weight.size()
    in_features_list = torch.zeros(in_features)
    out_features_list = torch.zeros(out_features)

    for i in range(in_features):
        in_features_slice = torch.index_select(conv_weight, 1, torch.tensor(i))
        in_features_list[i] = in_features_slice.nelement() - torch.count_nonzero(in_features_slice)
    for i in range(out_features):
        out_features_slice = torch.index_select(conv_weight, 0, torch.tensor(i))
        out_features_list[i] = out_features_slice.nelement() - torch.count_nonzero(out_features_slice)

    return in_features_list, out_features_list


def number_of_0_analysis_stochastic_deterministic(cfg: omegaconf.DictConfig = None, cfg2: omegaconf.DictConfig = None,
                                                  config_list=[]):
    net = get_model(cfg)

    prune_function(net, cfg)
    remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
    names_det, weights_det = zip(*get_layer_dict(net))
    sigma_per_layer = dict(zip(names_det, [cfg.sigma] * len(names_det)))
    # get noisy model
    noisy_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer)
    prune_function(noisy_model, cfg)
    remove_reparametrization(noisy_model, exclude_layer_list=cfg.exclude_layers)
    names_sto, weights_sto = zip(*get_layer_dict(noisy_model))

    number_of_0s_in_features_layers_det = []
    number_of_0s_out_features_layers_det = []

    number_of_0s_in_features_layers_sto = []
    number_of_0s_out_features_layers_sto = []

    for j, det_weight in enumerate(weights_det):
        if len(det_weight.size()) < 4:
            continue
        # in_features,out_features,H,W =det_weight.size()
        # in_features_aling_det = det_weight.view(-1,in_features).detach()
        # out_features_aling_det = det_weight.view(-1,out_features).detach()
        # in_features_aling_sto =  weights_sto[j].view(-1,in_features).detach()
        # out_features_aling_sto =  weights_sto[j].view(-1,out_features).detach()

        # zero_in_features_det = in_features_aling_det.nelement()-torch.count_nonzero(in_features_aling_det,dim=0)
        #
        # zero_out_features_det =out_features_aling_det.nelement()-torch.count_nonzero(out_features_aling_det,dim=0)

        zero_in_features_det, zero_out_features_det = count_zero_slices(det_weight)

        # zero_in_features_sto = in_features_aling_sto.nelement()- torch.count_nonzero(in_features_aling_sto,dim=0)
        # zero_out_features_sto = out_features_aling_sto.nelement()-torch.count_nonzero(out_features_aling_sto,dim=0)

        zero_in_features_sto, zero_out_features_sto = count_zero_slices(weights_sto[j])

        number_of_0s_in_features_layers_det.append(zero_in_features_det.detach().cpu().numpy())
        number_of_0s_out_features_layers_det.append(zero_out_features_det.detach().cpu().numpy())

        number_of_0s_in_features_layers_sto.append(zero_in_features_sto.detach().cpu().numpy())
        number_of_0s_out_features_layers_sto.append(zero_out_features_sto.detach().cpu().numpy())
    out_det_dict = {}
    in_det_dict = {}
    out_sto_dict = {}
    in_sto_dict = {}
    for i, (name, weight) in enumerate(zip(names_det, weights_det)):

        if len(weight.size()) < 4:
            continue
        out_features, in_features, H, W = weight.size()

        # plt.figure()
        f, (ax1, ax2) = plt.subplots(2, 1)
        # Figure size
        # Width of a bar
        width = 0.2

        plt.title("Layer {}, ".format(name))
        ind1 = np.arange(in_features)
        in_det_dict[name] = number_of_0s_in_features_layers_det[i]
        out_det_dict[name] = number_of_0s_out_features_layers_det[i]
        in_sto_dict[name] = number_of_0s_in_features_layers_sto[i]
        out_sto_dict[name] = number_of_0s_out_features_layers_sto[i]
        # Plotting
        ax1.bar(ind1 - width / 2, number_of_0s_in_features_layers_det[i], width, label='Deterministic')
        ax1.bar(ind1 + width / 2, number_of_0s_in_features_layers_sto[i], width, label='Stochastic')
        ax1.set_title('Input features')
        ax1.set_ylabel("Number of 0s")

        ind2 = np.arange(out_features)
        # Plotting
        ax2.bar(ind2 - width / 2, number_of_0s_out_features_layers_det[i], width, label='Deterministic')
        ax2.bar(ind2 + width / 2, number_of_0s_out_features_layers_sto[i], width, label='Stochastic')
        ax2.set_title('Output features')
        ax2.set_ylabel("Number of 0s")
        ax2.set_xlabel("Feature index")
        plt.legend()
        plt.tight_layout()
        save_string = 'data/zero_count/{}/{}/{}/sigma{}/'.format(cfg.architecture, cfg.dataset, cfg.amount, cfg.sigma)

        path = Path(save_string)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_string + "number_0s_{}_{}_pr_{}_pruner{}_layer{}.png".format(cfg.architecture, cfg.dataset, cfg.amount,
                                                                              cfg.pruner, name))
        plt.close()

    with open(save_string + "output_sto_features_count_dict.plk", "wb") as f:
        pickle.dump(out_sto_dict, f)
    with open(save_string + "input_sto_features_count_dict.plk", "wb") as f:
        pickle.dump(in_sto_dict, f)
    with open(save_string + "output_det_features_count_dict.plk", "wb") as f:
        pickle.dump(out_det_dict, f)
    with open(save_string + "input_det_features_count_dict.plk", "wb") as f:
        pickle.dump(in_det_dict, f)

    #   read helpers
    # with open(save_string+"output_sto_features_count_dict.plk","rb")as f:
    #     pickle.dump(out_sto_dict,f)
    # with open(save_string+"input_sto_features_count_dict.plk","rb")as f:
    #     pickle.dump(in_sto_dict,f)
    # with open(save_string+"output_det_features_count_dict.plk","rb")as f:
    #     pickle.dump(out_det_dict,f)
    # with open(save_string+"input_det_features_count_dict.plk","rb")as f:
    #     pickle.dump(in_det_dict,f)


def CDF_weights_analysis_stochastic_deterministic(cfg: omegaconf.DictConfig = None, cfg2: omegaconf.DictConfig = None,
                                                  config_list=[], range: tuple = None):
    if cfg2 is None:
        net = get_model(cfg)
        param_vector = torch.abs(parameters_to_vector(net.parameters()))
        names, weights = zip(*get_layer_dict(net))

        # deterministic_pruned_model = copy.deepcopy(net)
        # if cfg.pruner == "global":
        #     prune_with_rate(deterministic_pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        # if cfg.pruner == "manual":
        #     prune_with_rate(deterministic_pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner="manual", pr_per_layer=pr_per_layer)
        #     individual_prs_per_layer = prune_with_rate(copy_of_pruned_model, target_sparsity,
        #                                                exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                                                pruner="lamp", return_pr_per_layer=True)
        #     if cfg.use_wandb:
        #         log_dict = {}
        #         for name, elem in individual_prs_per_layer.items():
        #             log_dict["individual_{}_pr".format(name)] = elem
        #         wandb.log(log_dict)
        # if cfg.pruner == "lamp":
        #     prune_with_rate(deterministic_pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
        #                     type="layer-wise",
        #                     pruner=cfg.pruner)

        sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))
        noisy_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer)
        param_vector_noisy = torch.abs(parameters_to_vector(noisy_model.parameters()))

        sigma_per_layer = dict(zip(names, [0.001] * len(names)))
        noisy_model2 = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer)
        param_vector_noisy2 = torch.abs(parameters_to_vector(noisy_model2.parameters()))
        # param_vector= param_vector/max(param_vector)
        # pruning_rates = [0.8,0.85,0.9,0.95]
        pruning_rates = [0.9]
        colors = ["m", "g", "r", "c"]
        dataset_string = cfg.dataset.upper()
        architecture_string = ""
        if cfg.architecture == "resnet18":
            architecture_string = "ResNet18"
        if cfg.architecture == "resnet50":
            architecture_string = "ResNet50"
        if cfg.architecture == "VGG19":
            architecture_string = "VGG19"

        # For net 1
        if range is None:
            count1, bin_counts1 = torch.histogram(param_vector, bins=len(param_vector))
        else:
            count1, bin_counts1 = torch.histogram(param_vector, bins=len(param_vector), range=range)
        pdf1 = count1 / torch.sum(count1)
        cdf1 = torch.cumsum(pdf1, dim=0)
        plt.plot(bin_counts1[1:].detach().numpy(), cdf1.detach().numpy(), label=f"{architecture_string}-Det.")
        names1, weights1 = zip(*get_layer_dict(net))

        for i, pr in enumerate(pruning_rates):
            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights1, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="--", label=f"Threshold @ pr {pr} for Det.")

        # For net 2
        if range is None:
            count2, bin_counts2 = torch.histogram(param_vector_noisy, bins=len(param_vector_noisy))
        else:
            count2, bin_counts2 = torch.histogram(param_vector_noisy, bins=len(param_vector_noisy), range=range)
        pdf2 = count2 / torch.sum(count2)
        cdf2 = torch.cumsum(pdf2, dim=0)
        plt.plot(bin_counts2[1:].detach().numpy(), cdf2.detach().numpy(),
                 label=f"{architecture_string}-Sto.@{cfg.sigma}")

        names2, weights2 = zip(*get_layer_dict(noisy_model))

        for i, pr in enumerate(pruning_rates):
            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights2, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="dotted",
                        label=f"Threshold @ pr {pr} for Sto.")

        plt.title(f"Deterministic and Stochastic {architecture_string} model on {dataset_string}")
        plt.legend()
        if range is not None:
            plt.savefig(
                f"cdf_{cfg.architecture}_det_vs_sto_{cfg.dataset}_s{cfg.sigma}_{cfg.pruner}_{range[1]}_range.pdf")
        else:
            plt.savefig(f"cdf_{cfg.architecture}_det_vs_sto_{cfg.dataset}_s{cfg.sigma}_{cfg.pruner}_full_range.pdf")

    if cfg2 is not None:

        net1 = get_model(cfg)
        net2 = get_model(cfg2)
        param_vector1 = torch.abs(parameters_to_vector(net1.parameters()))
        param_vector2 = torch.abs(parameters_to_vector(net2.parameters()))

        # param_vector= param_vector/max(param_vector)
        pruning_rates = [0.9, 0.95]
        dataset_string = cfg.dataset.upper()

        assert dataset_string == cfg2.dataset.upper(), "The solutions are not trained on the same dataset. {} is trained on {} and {} is trained on  {}".format(
            cfg.architecture, cfg.dataset, cfg2.architecure, cfg2.dataset)

        architecture_string1 = ""
        architecture_string2 = ""
        if cfg.architecture == "resnet18":
            architecture_string1 = "ResNet18"
        if cfg.architecture == "resnet50":
            architecture_string1 = "ResNet50"
        if cfg.architecture == "VGG19":
            architecture_string1 = "VGG19"

        if cfg2.architecture == "resnet18":
            architecture_string2 = "ResNet18"
        if cfg2.architecture == "resnet50":
            architecture_string2 = "ResNet50"
        if cfg2.architecture == "VGG19":
            architecture_string2 = "VGG19"

        colors = ["m", "g", "r", "c"]
        # For net 1
        count1, bin_counts1 = torch.histogram(param_vector1, bins=len(param_vector1), range=(0, 0.09))
        pdf1 = count1 / torch.sum(count1)
        cdf1 = torch.cumsum(pdf1, dim=0)
        plt.plot(bin_counts1[1:].detach().numpy(), cdf1.detach().numpy(), label=f"{architecture_string1}")
        # For net 2
        count2, bin_counts2 = torch.histogram(param_vector2, bins=len(param_vector2), range=(0, 0.09))
        pdf2 = count2 / torch.sum(count2)
        cdf2 = torch.cumsum(pdf2, dim=0)
        plt.plot(bin_counts2[1:].detach().numpy(), cdf2.detach().numpy(), label=f"{architecture_string2}")

        names1, weights1 = zip(*get_layer_dict(net1))
        names2, weights2 = zip(*get_layer_dict(net2))

        for i, pr in enumerate(pruning_rates):
            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights1, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="--",
                        label=f"Threshold @ pr {pr} for {architecture_string1}")

            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights2, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="dotted",
                        label=f"Threshold @ pr {pr} for {architecture_string2}")

        plt.title(f"{architecture_string1} and {architecture_string2} on {dataset_string}")
        plt.legend()
        plt.savefig(f"cdf_{cfg.architecture}_{cfg2.architecture}_{cfg.dataset}.pdf")


def weights_analysis_per_weight(cfg: omegaconf.DictConfig = None, cfg2: omegaconf.DictConfig = None, config_list=[]):
    if cfg2 is None:
        net = get_model(cfg)
        param_vector = torch.abs(parameters_to_vector(net.parameters()))
        # param_vector= param_vector/max(param_vector)
        pruning_rates = [0.8, 0.85, 0.9, 0.95]
        dataset_string = cfg.dataset.upper()
        architecture_string = ""
        if cfg.architecture == "resnet18":
            architecture_string = "ResNet18"
        if cfg.architecture == "resnet50":
            architecture_string = "ResNet50"
        if cfg.architecture == "VGG19":
            architecture_string = "VGG19"

        colors = ["m", "g", "r", "c"]
        count, bin_counts = torch.histogram(param_vector, bins=len(param_vector))
        # torch.histogram(torch.tensor([1., 2, 1]), bins=4, range=(0., 3.), weight=torch.tensor([1., 2., 4.]))
        # torch.histogram(torch.tensor([1., 2, 1]), bins=4, range=(0., 3.), weight=torch.tensor([1., 2., 4.]), density=True)
        pdf = count / torch.sum(count)
        cdf = torch.cumsum(pdf, dim=0)
        plt.plot(bin_counts[1:].detach().numpy(), cdf.detach().numpy(), label="CDF")
        names, weights = zip(*get_layer_dict(net))

        for i, pr in enumerate(pruning_rates):
            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="--",
                        label=f"Threshold for pruning rate: {pr}")

        plt.title(f"{architecture_string} on {dataset_string}")
        plt.legend()
        plt.savefig(f"cdf_{cfg.architecture}_{cfg.dataset}.pdf")
    if cfg2 is not None:

        net1 = get_model(cfg)
        net2 = get_model(cfg2)
        param_vector1 = torch.abs(parameters_to_vector(net1.parameters()))
        param_vector2 = torch.abs(parameters_to_vector(net2.parameters()))

        # param_vector= param_vector/max(param_vector)
        pruning_rates = [0.9, 0.95]
        dataset_string = cfg.dataset.upper()

        assert dataset_string == cfg2.dataset.upper(), "The solutions are not trained on the same dataset. {} is trained on {} and {} is trained on  {}".format(
            cfg.architecture, cfg.dataset, cfg2.architecure, cfg2.dataset)

        architecture_string1 = ""
        architecture_string2 = ""
        if cfg.architecture == "resnet18":
            architecture_string1 = "ResNet18"
        if cfg.architecture == "resnet50":
            architecture_string1 = "ResNet50"
        if cfg.architecture == "VGG19":
            architecture_string1 = "VGG19"

        if cfg2.architecture == "resnet18":
            architecture_string2 = "ResNet18"
        if cfg2.architecture == "resnet50":
            architecture_string2 = "ResNet50"
        if cfg2.architecture == "VGG19":
            architecture_string2 = "VGG19"

        colors = ["m", "g", "r", "c"]
        # For net 1
        count1, bin_counts1 = torch.histogram(param_vector1, bins=len(param_vector1), range=(0, 0.09))
        pdf1 = count1 / torch.sum(count1)
        cdf1 = torch.cumsum(pdf1, dim=0)
        plt.plot(bin_counts1[1:].detach().numpy(), cdf1.detach().numpy(), label=f"{architecture_string1}")
        # For net 2
        count2, bin_counts2 = torch.histogram(param_vector2, bins=len(param_vector2), range=(0, 0.09))
        pdf2 = count2 / torch.sum(count2)
        cdf2 = torch.cumsum(pdf2, dim=0)
        plt.plot(bin_counts2[1:].detach().numpy(), cdf2.detach().numpy(), label=f"{architecture_string2}")

        names1, weights1 = zip(*get_layer_dict(net1))
        names2, weights2 = zip(*get_layer_dict(net2))

        for i, pr in enumerate(pruning_rates):
            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights1, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="--",
                        label=f"Threshold @ pr {pr} for {architecture_string1}")

            threshold, index_threshold, full_vector = get_threshold_and_pruned_vector_from_pruning_rate(
                list_of_layers=weights2, pruning_rate=pr)
            plt.axvline(threshold, linewidth=1, color=colors[i], linestyle="dotted",
                        label=f"Threshold @ pr {pr} for {architecture_string2}")

        plt.title(f"{architecture_string1} and {architecture_string2} on {dataset_string}")
        plt.legend()
        plt.savefig(f"cdf_{cfg.architecture}_{cfg2.architecture}_{cfg.dataset}.pdf")

    if config_list:
        names = []

    # names, weights = zip(*get_layer_dict(net))
    # vector = torch.abs(parameters_to_vector(weights))
    # new_vector = (vector)/torch.max(torch.abs(vector))
    # del vector
    # sort_index = torch.argsort(new_vector)
    # index = torch.arange(1,len(new_vector)+1,1)/len(new_vector)
    # plt.plot(index,new_vector[sort_index])
    #
    # plt.savefig("cdf.pdf")
    # plt.figure()
    # plt.hist(new_vector, normed=True, cumulative=True, label='CDF',
    #          histtype='step', alpha=0.8)
    # plt.savefig("cdf2.pdf")
    # average_magnitude = lambda w: torch.abs(w).mean()
    # average_magnitudes_by_layer = np.array(list(map(average_magnitude, weights)))
    # number_param = lambda w: w.nelement()
    # elements = np.array(list(map(number_param, weights)))
    # ratios = average_magnitudes_by_layer / cfg.sigma
    # sorted_idexes_by_ratios = np.flip(np.argsort(ratios))
    # weights_magnitude_by_size = [np.abs(weights[i].flatten().detach().numpy()) for i in sorted_idexes_by_size]
    # names_by_size = [names[i] for i in sorted_idexes_by_size]
    # n = np.array([],dtype=str)
    # we = pyarr.array("f", [])
    # for j, w in enumerate(weights_magnitude_by_size):
    #     we.extend(w)
    #     for i in range(len(w)):
    #        n = np.append(n,names_by_size[j])
    # df = pd.DataFrame(data={"x": float(we), "g": n})
    # df.to_csv(
    #     "data/weights_by_size.csv", sep=",", index=False
    # )
    # df = pd.read_csv("data/weights_by_size.csv", header=0, sep=",")
    # # plot_ridge_plot(df, "data/figures/original_weights_ridgeplot.png".format(cfg.sigma))
    # df.rename(columns={"g": "Layer Name", "x": "Weight magnitude"}, inplace=True)
    # df["Weight magnitude"] = df['Weight magnitude'].apply(lambda x: np.abs(x))
    # print(df)
    #
    # def q25(x):
    #     return x.quantile(0.25)
    #
    # def q50(x):
    #     return x.quantile(0.50)
    #
    # def q75(x):
    #     return x.quantile(0.75)
    #
    # # vals = {'Weight magnitude': [q25, q50, q75]}
    # # quantile_df = df.groupby('Layer Name').agg(vals)
    # # quantile_df = df.groupby("").quantile([0.25,0.5,0.75])
    # plot_histograms_per_group(df, "Weight magnitude", "Layer Name")
    # fancy_bloxplot(df, x="Layer Name", y="Weight magnitude", rot=90)
    # # print(quantile_df)
    # # quantile_df.to_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", index=True)

    ########################## This is double bar plot ################################################

    # y_axes_label = r"$\frac{\bar{\mid w \mid}}{\sigma}$"
    # title = r"$\sigma = {}$".format(cfg.sigma)
    #
    # df2 = pd.DataFrame(data={"y1": ratios[sorted_idexes_by_ratios].transpose(), "y2": elements[
    #     sorted_idexes_by_ratios].transpose()})
    # xtick_labels = [names[i] for i in sorted_idexes_by_ratios]
    # plot_double_barplot(df2,y_axes_label,"Number of parameters",title,f"data/figures/sigma_"
    #                                                                  f"{cfg.sigma}_V_original_weights.png",
    #                     xtick_labels,logy2=True)


def epsilon_for_pruning_rate_given_best_sigma(filepath: str, cfg: omegaconf.DictConfig):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    epsilons = []
    pruning_rates = []
    num_col = len(df["Pruning Rate"].unique())
    num_row = len(df["sigma"].unique())
    minimun_overall = df["Epsilon"].min()
    print("The minimun epsilon is {} with the following charactersitics".format(minimun_overall))
    print(df[df["Epsilon"] == minimun_overall])
    for pr in df["Pruning Rate"].unique():
        current_df = df[df["Pruning Rate"] == pr]
        best_epsilon = current_df.minimun()
        best_epsilon_info = current_df[current_df["Epsilon"] == best_epsilon]
        print("For pruning rate {}  the best epsilon is {} obtained with the following parameters \n {}".format(pr,
                                                                                                                best_epsilon,
                                                                                                                best_epsilon_info))
        epsilons.append(best_epsilon)
        pruning_rates.append(pr)
    plt.plot(epsilons, pruning_rates)
    plt.ylabel(r"$\epsilon\mid_{\sigma_{best}}$")
    plt.xlabel("Pruning rate")
    plt.savefig("data/figures/best_epsilon.png")


def plot_specific_pr_sigma_epsilon_statistics(filepath: str, cfg: omegaconf.DictConfig, specific_sigmas: list,
                                              specific_pruning_rates: list):
    use_cuda = torch.cuda.is_available()

    net = get_model(cfg)
    trainloader, valloader, testloader = get_datasets(cfg)

    original_performance = test(net, use_cuda, testloader)

    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    # df = df_temp[df_temp["Type"]!= "Stochastic GMP"]
    num_col = len(specific_pruning_rates)
    # pylustrator.start()
    # num_col = 3
    # num_row = 3
    # For each num_row is going to be one plot with num_col subplots for each pruning rate
    for i, current_sigma in enumerate(specific_sigmas):
        # current_sigma = df["sigma"].unique()[i]
        # if num_col == 1:
        #     fig = plt.figure(figsize=(13, 13 ))
        #     axs_p = fig.subplot()
        fig, axs_p = plt.subplots(1, num_col, figsize=(9
                                                       , 10), layout="compressed")
        axs = []
        if isinstance(axs_p, SubplotBase):
            axs.append(axs_p)
        else:
            axs.extend(axs_p)

        # plt.suptitle(r"$\sigma={}$".format(current_sigma), fontsize=20)
        for j, axj in enumerate(axs):
            current_pr = specific_pruning_rates[j]
            # axj = fig.add_subplot(1, num_col, j+1,)
            current_df = df[df["sigma"] == current_sigma]
            current_df = current_df[current_df["Pruning Rate"] == current_pr]
            current_df["Accuracy"] = original_performance - current_df["Epsilon"]
            pruned_original = copy.deepcopy(net)

            if cfg.pruner == "global":
                prune_with_rate(pruned_original, float(current_pr), exclude_layers=cfg.exclude_layers, type="global")
            else:
                prune_with_rate(pruned_original, float(current_pr), exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            remove_reparametrization(pruned_original, exclude_layer_list=cfg.exclude_layers)
            pruned_original_performance = test(pruned_original, use_cuda, testloader, verbose=0)
            delta_pruned_original_performance = original_performance - pruned_original_performance
            ###############  LAMP ################################
            total_observations = len(current_df["Accuracy"][current_df["Type"] == "Stochastic GMP"])

            lamp_model = copy.deepcopy(net)

            prune_with_rate(lamp_model, float(current_pr), exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="lamp")
            print("Lamp deterministic performance")
            LAMP_deterministic_performance = test(lamp_model, use_cuda, testloader, verbose=0)
            print(LAMP_deterministic_performance)
            print("Lamp deterministic performance on val set pruning rate {}".format(current_pr))
            LAMP_deterministic_performance_val = test(lamp_model, use_cuda, valloader, verbose=0)
            print(LAMP_deterministic_performance_val)
            print("GLOBAL deterministic performance")
            print(pruned_original_performance)
            print("Global deterministic performance on val set pruning rate {}".format(current_pr))
            GLOBAL_deterministic_performance_val = test(pruned_original, use_cuda, valloader, verbose=0)
            print(GLOBAL_deterministic_performance_val)
            for type in current_df["Type"].unique():
                number_of_elements_above_LAMP_determinstic = (
                        current_df["Accuracy"][type == current_df["Type"]] > LAMP_deterministic_performance).sum()
                number_of_elements_above_GLOBAL_determinstic = (
                        current_df["Accuracy"][type == current_df["Type"]] > pruned_original_performance).sum()
                print(
                    "For pruning rate {} and sigma {} the number of elements above deterministic LAMP for type {} are {} then the fraction is {}".format(
                        current_pr, current_sigma, type, number_of_elements_above_LAMP_determinstic,
                        number_of_elements_above_LAMP_determinstic / total_observations))
                print(
                    "For pruning rate {} and sigma {} the number of elements above deterministic Global for type {} are {} then the fraction is {}".format(
                        current_pr, current_sigma, type, number_of_elements_above_GLOBAL_determinstic,
                        number_of_elements_above_GLOBAL_determinstic / total_observations))

            axj = sns.boxplot(x='Type',
                              y='Accuracy',
                              hue="Type",
                              data=current_df,
                              ax=axj
                              # width = 1
                              )

            # adjust_box_widths(fig, 2)

            if cfg.pruner == "global":
                axj.axhline(pruned_original_performance, c="purple", linewidth=2.5, label="Deterministic GMP")
            else:
                axj.axhline(pruned_original_performance, c="purple", linewidth=2.5, label="Deterministic LAMP")
            # axj.axhline(LAMP_deterministic_performance, c="xkcd:greeny yellow", linewidth=2.5, label="LAMP "
            #                                                                                          "Deterministic "
            #                                                                                          "Pruning")

            axj = sns.stripplot(x='Type',
                                y='Accuracy',
                                hue='Type',
                                jitter=True,
                                dodge=True,
                                marker='o',
                                edgecolor="gray",
                                linewidth=1,
                                # palette="set2",
                                alpha=0.5,
                                data=current_df,
                                ax=axj)
            # axj.set_title("Pruning Rate = {}".format(current_pr), fontsize=15)
            # plt.ylabel("Accuracy", fontsize=20)
            # ticks = g.get_yticks(minor=False)
            # accuracy_ticks = original_performance - ticks
            # axj.set_ylabel("Accuracy", fontsize=30)
            # axj.set_yticks(accuracy_ticks, minor=False)
            # g.tick_params(axis='y', which='major', labelsize=20)
            # plt.ylim(25, 60)
            # ax2 = g.twinx()
            # # y1_tick_positions = ax.get_ticks()
            # epsilon_ticks = original_performance - np.linspace(25, 55, len(ticks))
            # ax2.set_yticks(epsilon_ticks,minor=False)
            # ax2.set_ylabel(r"$\epsilon$", fontsize=20)
            # ax2.spines['right'].set_color('red')
            # ax2.tick_params(axis="y", colors="red",labelsize =15
            #                 )
            # ax2.yaxis.label.set_color('red')
            # ax2.invert_yaxis()

            handles, labels = axj.get_legend_handles_labels()
            new_labels = ["Stochastic GMP", "Sto. Mask on Original Weights", "Det. Mask on Sto. Weights",
                          "Deterministic GMP"]
            l = axj.legend(handles[:4], new_labels, fontsize=fs * 0.8)
            axj.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            axj.set_xlabel("")
            axj.tick_params(axis="y", labelsize=fs)
            axj.set_ylabel("Accuracy", fontsize=fs)
            # plt.ylim(25,55)
            # l = axj.get_ylim()
            # plt.ylim(l[0], l[1])
            # ax2 = axj.twinx()
            # ax2.set_ylim(l[0], l[1])
            # # l2 = ax2.
            # # f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            # # ticks = f(axj.get_yticks())
            # unnormalied_ticks = axj.get_yticks()
            # # ax2.set_yticks(axj.get_yticks(),labeget_ylim()
            # #             # ax2.set_ylim(*l)ls=[25,30,35,40,45,50,55], minor=False)
            # #
            # # ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            # # axj.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            # y2_ticks = ax2.get_yticks()
            # epsilon_ticks = original_performance - np.linspace(unnormalied_ticks[0], unnormalied_ticks[-1],
            #                                                    len(y2_ticks))
            # new_ticks = np.linspace(l[0], l[1],
            #                         len(y2_ticks))
            # if l[1] - l[0] <= 2:
            #     formatter_q = lambda n: "{:10.2f}".format(n).replace(" ", "")
            #     formatter_f = lambda n: "{:10.2f}".format(n).replace(" ", "")
            # else:
            #     formatter_q = lambda n: "{:10.0f}".format(n).replace(" ", "")
            #     formatter_f = lambda n: "{:10.1f}".format(n).replace(" ", "")
            # new_ticks = list(map(formatter_q, unnormalied_ticks))
            # axj.set_yticks(ticks=unnormalied_ticks, minor=False)
            # axj.set_yticklabels(new_ticks)
            # epsilon_ticks = list(map(formatter_f, epsilon_ticks))
            # epsilon_ticks.reverse()
            # ax2.set_yticks(ticks=unnormalied_ticks, minor=False)
            # ax2.set_yticklabels(epsilon_ticks)
            # # ax2.set_yticks([25,30,35,40,45,50,55], minor=False)
            # ax2.set_ylabel(r"$\epsilon$", fontsize=20)
            # ax2.spines['right'].set_color('red')
            # ax2.tick_params(axis="y", colors="red", labelsize=20
            #                 )
            # ax2.yaxis.label.set_color('red')
            # ax2.invert_yaxis()
            # ax2.tick_params(
            #     axis='x',  # changes apply to the x-axis
            #     which='both',  # both major and minor ticks are affected
            #     bottom=False,  # ticks along the bottom edge are off
            #     top=False,  # ticks along the top edge are off
            #     labelbottom=False)

            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.png".format(current_pr,current_sigma), bbox_inches="tight")
            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.pdf".format(current_pr,current_sigma), bbox_inches="tight")
        # plt.tight_layout()
        pr_string = "_".join(list(map(str, specific_pruning_rates)))
        # plt.show()
        plt.savefig("data/epsilon_allN_all_{}_{}_pr_{}_sigma={}_{}.pgf".format(cfg.dataset, cfg.architecture, pr_string,
                                                                               current_sigma, cfg.pruner),
                    bbox_inches="tight")
        plt.savefig("data/epsilon_allN_all_{}_{}_pr_{}_sigma={}_{}.pdf".format(cfg.dataset, cfg.architecture, pr_string,
                                                                               current_sigma, cfg.pruner),
                    bbox_inches="tight")


def plot_val_accuracy_wandb(filepath, save_path, x_variable, y_variable, xlabel, ylabel):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    df.plot(x=x_variable, y=y_variable, legend=False)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.savefig(save_path)


def plot_test_accuracy_wandb(filepaths, legends, save_path, x_variable, y_variable, y_min, y_max, xlabel, ylabel,
                             title=""):
    figure, ax = plt.subplots()
    for filepath in filepaths:
        df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
        plt.plot(df[x_variable].ewm().mean(), df[y_variable].ewm())
        plt.fill_between(df[x_variable], df[y_min].ewm().mean(), df[y_max].man(), alpha=0.2)

    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    ax.legend(legends)
    plt.title(title, fontsize=20)
    # plt.savefig(save_path)
    plt.show()


def statistics_of_epsilon_for_stochastic_pruning(filepath: str, cfg: omegaconf.DictConfig, identifier: str):
    use_cuda = torch.cuda.is_available()

    net = get_model(cfg)

    _, _, testloader = get_datasets(cfg)

    original_performance = test(net, use_cuda, testloader)

    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    num_col = len(df["Pruning Rate"].unique())
    num_row = len(df["sigma"].unique())
    # num_col = 3
    # num_row = 3
    # For each num_row is going to be one plot with num_col subplots for each pruning rate
    for i in range(num_row):
        current_sigma = df["sigma"].unique()[i]
        fig, axs = plt.subplots(1, num_col, figsize=(17, 10), layout="constrained")
        # plt.suptitle(r"$\sigma={}$".format(current_sigma), fontsize=20)
        for j, axj in enumerate(axs.flat):
            current_pr = df["Pruning Rate"].unique()[j]
            # axj = fig.add_subplot(1, num_col, j+1,)
            current_df = df[df["sigma"] == current_sigma]
            current_df = current_df[current_df["Pruning Rate"] == current_pr]

            pruned_original = copy.deepcopy(net)
            prune_with_rate(pruned_original, float(current_pr))
            remove_reparametrization(pruned_original)
            pruned_original_performance = test(pruned_original, use_cuda, testloader, verbose=0)
            delta_pruned_original_performance = original_performance - pruned_original_performance
            g = sns.boxplot(x='Type',
                            y='Epsilon',
                            hue="Type",
                            data=current_df,
                            ax=axj
                            )
            g.axhline(delta_pruned_original_performance, c="purple", label="Deterministic Pruning")
            g = sns.stripplot(x='Type',
                              y='Epsilon',
                              hue='Type',
                              jitter=True,
                              # dodge=True,
                              marker='o',
                              edgecolor="gray",
                              linewidth=1,
                              # palette="set2",
                              alpha=0.5,
                              data=current_df,
                              ax=axj)
            axj.set_title("Pruning Rate = {}".format(current_pr), fontsize=15)
            axj.set_ylabel("Epsilon".format(current_pr), fontsize=15)
            handles, labels = g.get_legend_handles_labels()
            l = axj.legend(handles[:4], labels[:4])
            axj.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            axj.set_xlabel("")

            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.png".format(current_pr,current_sigma), bbox_inches="tight")
            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.pdf".format(current_pr,current_sigma), bbox_inches="tight")
        plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.png".format(identifier, current_sigma), bbox_inches="tight")
        plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.pdf".format(identifier, current_sigma), bbox_inches="tight")

    # subfigs = fig.subfigures(num_row, 1)
    # for i, subfig in enumerate(subfigs):
    #     axes = subfig.subplots(1, num_col, sharey=True)
    #     string_formatted = ""
    #     current_sigma = df["sigma"].unique()[i]
    #     subfig.suptitle(r"$\sigma={}$".format(df["sigma"].unique()[i]))
    #     # subfig.suptitle(r"$\sigma={}$".format(0.00214))
    #     for j, ax in enumerate(axes):
    #         current_pr = df["Pruning Rate"].unique()[j]
    #         ax.set_title(f'Pruning Rate={df["Pruning Rate"].unique()[j]}', fontsize=12)
    #         # ax.set_title(f'Pruning Rate=0.8', fontsize=12)
    #         current_df = df[df["sigma"] == current_sigma]
    #         current_df = current_df[current_df["Pruning Rate"] == current_pr]
    #         sns.boxplot(ax=ax, data=current_df, x='Population', y='Epsilon', hue="Type")
    #
    #         # sns.catplot(ax=ax,x='Population',
    #         #             y='Epsilon',
    #         #             hue="Type",
    #         #             kind="box",
    #         #             legend=False,
    #         #             # height=6,
    #         #             # aspect=1.3,
    #         #             data=current_df)  # .set(title="Title")
    #         # # make grouped stripplot
    #         # sns.stripplot(ax=ax,x='Population',
    #         #               y='Epsilon',
    #         #               hue='Type',
    #         #               jitter=True,
    #         #               dodge=True,
    #         #               marker='o',
    #         #               # palette="Set2",
    #         #               alpha=0.5,
    #         #               data=current_df)
    #
    # plt.savefig("test.png")
    # ########## For N = 1000 ###########
    # all1000_performances = np.load("data/population_data/performances_gaussian_transfer_t_20-34-.npy")
    # labels1000 = np.load("data/population_data/labels_gaussian_transfer_t_20-34.npy")
    # deltas_det_to_sto1000 = np.load("data/population_data/deltas_gaussian_det_mask_to_sto_weights_deltas "
    #                                "_N_1000_t_20-34.npy")
    #
    # N = len(deltas_det_to_sto1000)
    # stochastic_deltas1000 = []
    # for i, label in enumerate(labels1000):
    #
    #     if label == "stochastic pruned":
    #         stochastic_deltas1000.append(original_performance - all1000_performances[i])
    #         if len(stochastic_deltas1000) == N:
    #             break


def test_figure():
    exercise = sns.load_dataset("exercise")
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    g = sns.boxplot(x="time", y="pulse", hue="kind", data=exercise, ax=ax1)  # pass ax1
    g = sns.stripplot(x="time", y="pulse", hue="kind",
                      jitter=True,
                      dodge=True,
                      marker='o',
                      edgecolor="gray",
                      linewidth=1,
                      # palette="set2",
                      alpha=0.5,
                      data=exercise, ax=ax1)  # pass ax1

    ax2 = fig.add_subplot(122)
    g = sns.boxplot(x="time", y="pulse", hue="kind", data=exercise, ax=ax2)  # pass ax2
    g = sns.stripplot(x="time",
                      y="pulse", hue="kind",
                      jitter=True,
                      dodge=True,
                      marker='o',
                      edgecolor="gray",
                      linewidth=1,
                      # palette="set2",
                      alpha=0.5,
                      data=exercise,
                      ax=ax2)  # pass ax1

    plt.close(2)
    plt.close(3)
    plt.tight_layout()
    plt.savefig("test.png")


def population_sweeps_transfer_mask_rank_experiments(cfg, identifier=""):
    # cfg = omegaconf.DictConfig({
    #     "population": 1,
    #     "generations": 10,
    #     "epochs": 100,
    #     "short_epochs": 10,
    #     # "architecture": "VGG19",
    #     "architecture": "resnet18",
    #     "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
    #     # "solution": "trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth",
    #     "noise": "gaussian",
    #     "pruner": "lamp",
    #     "model_type": "alternative",
    #     "exclude_layers": ["conv1", "linear"],
    #     # "exclude_layers": ["features.0", "classifier"],
    #     "fine_tune_exclude_layers": True,
    #     "fine_tune_non_zero_weights": True,
    #     "sampler": "tpe",
    #     "flop_limit": 0,
    #     "one_batch": True,
    #     "measure_gradient_flow":True,
    #     "full_fine_tune": False,
    #     "use_stochastic": True,
    #     # "sigma": 0.0021419609859022197,
    #     "sigma": 0.011,
    #     "noise_after_pruning":0,
    #     "amount": 0.9,
    #     "dataset": "cifar10",
    #     "batch_size": 512,
    #     # "batch_size": 128,
    #     "num_workers": 0,
    #     "save_model_path": "stochastic_pruning_models/",
    #     "save_data_path": "stochastic_pruning_data/",
    #     "use_wandb": True
    # })
    Ns = [10, 50, 100]
    sigmas = np.linspace(start=0.001, stop=0.005, num=3)
    # sigmas = [0.005]
    if cfg.architecture == "resnet18":
        pruning_rates = [0.5, 0.8, 0.9]
    if cfg.architecture == "VGG19":
        pruning_rates = [0.72, 0.88, 0.94]

    df = pd.DataFrame(columns=["Epsilon", "Population", "Type", "Pruning Rate", "sigma"])
    # result = time.localtime(time.time())
    file_path = f"data/epsilon_experiments_{cfg.dataset}_{cfg.architecture}_{cfg.pruner}_{identifier}"
    for pop in Ns:
        for sig in sigmas:
            for pr in pruning_rates:
                cfg.population = pop
                cfg.sigma = float(sig)
                cfg.amount = pr
                df_result = transfer_mask_rank_experiments_no_plot(cfg)
                df_result.to_csv(file_path + f"pop_{pop}_sig_{sig}_pr_{pr}.csv", sep=",", index=False)
                df = pd.concat((df, df_result))
    df.to_csv(file_path + "_full.csv", sep=",", index=False)
    # full_dataset :pd.DataFrame = None

    # for file in glob.glob('data/epsilon_experiments_1678152343.18253_t_1-25_lamppop*.csv'):
    #     temp_df = pd.read_csv(filepath_or_buffer=file, sep=",", header=0)
    #     if full_dataset is None:
    #         full_dataset = temp_df
    #     else:
    #         full_dataset = pd.concat((full_dataset,temp_df))
    #
    # full_dataset.to_csv("epsilon_experiments_lamp_full.csv", sep=",", index=False)
    return file_path


############################# Stochastic pruning with sigma optimization ###########################################
def get_sigma_sample_per_layer_optuna(trial: optuna.Trial, lower_limit: int, upper_limit_per_layer: dict,
                                      exclude_layer_list: list = []):
    # trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
    new_dict = {}
    for key, value in upper_limit_per_layer.items():
        if key in exclude_layer_list:
            continue
        else:
            new_dict[key] = trial.suggest_float("{}_sigma".format(key), lower_limit, value, log=True)

    return trial, new_dict


def get_sigma_pr_sample_per_layer_optuna(trial: optuna.Trial, lower_limit: int, upper_limit_per_layer: dict,
                                         number_weights_per_layer: dict, total_number_weights: int):
    # trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
    sigma_dict = {}
    pr_dict = {}
    sum_of_pruning_weights_by_layer = 0
    for key, value in upper_limit_per_layer.items():
        sigma_dict[key] = trial.suggest_float("{}_sigma".format(key), lower_limit, value, log=True)
        pr_dict[key] = trial.suggest_float("{}_pr".format(key), 0.05, 0.99)
        sum_of_pruning_weights_by_layer += pr_dict[key] * number_weights_per_layer[key]
    # Store the constraints as user attributes so that they can be restored after optimization.
    c0 = total_number_weights - sum_of_pruning_weights_by_layer
    trial.set_user_attr("constraint", (c0,))
    return trial, sigma_dict, pr_dict


def get_pr_sample_per_layer_optuna(trial: optuna.Trial, layer_names: dict,
                                   number_weights_per_layer: dict, lower_bounds: dict, total_number_weights_pruned: int,
                                   exclude_layers=[]):
    # trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
    pr_dict = {}
    sum_of_pruning_weights_by_layer = 0
    for key in layer_names:
        if key in exclude_layers:
            sum_of_pruning_weights_by_layer += number_weights_per_layer[key]
            continue
        pr_dict[key] = trial.suggest_float("{}_pr".format(key), lower_bounds[key], 0.99)
        sum_of_pruning_weights_by_layer += pr_dict[key] * number_weights_per_layer[key]
    # Store the constraints as user attributes so that they can be restored after optimization.
    c0 = total_number_weights_pruned - sum_of_pruning_weights_by_layer
    trial.set_user_attr("constraint", (c0,))
    return trial, pr_dict, c0


def stochastic_pruning_with_sigma_and_pr_optimization(cfg: omegaconf.DictConfig):
    print("Config: \n{}".format(omegaconf.OmegaConf.to_yaml(cfg)))
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    N = cfg.population
    first_sparsity = 0.5
    original_model = get_model(cfg)
    if cfg.use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"pruning_rate_sigma_optimization",
            reinit=True,
            save_code=True,
        )

    _, pr_per_layer, baseline_non_zero, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)

    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    count = lambda w: w.nelement()
    number_per_layer = list(map(count, weights))
    number_per_layer = dict(zip(names, number_per_layer))

    ################# Optuna study###########################
    def constraints(trial):
        return trial.user_attrs["constraint"]

    sampler = optuna.integration.BoTorchSampler(
        constraints_func=constraints,
        n_startup_trials=10,
    )
    study = optuna.create_study(study_name="stochastic pruning with pr and sigma optimisation", sampler=sampler,
                                direction="maximize")
    total_surviving_weights = count_parameters(original_model) * cfg.amount

    quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
                                     names=["layer", "q25", "q50", "q75"])
    sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigmas_found = None
    best_pruning_rates_found = None
    current_best_model = original_model

    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        current_gen_prs = []
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)
            ############### Here I ask for pr and for sigma ###################################
            individual_trial, sigmas_for_individual, pr_for_individual = get_sigma_pr_sample_per_layer_optuna(
                individual_trial,
                lower_bound,
                sigma_upper_bound_per_layer,
                number_weights_per_layer=number_per_layer,
                total_number_weights=total_surviving_weights)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            ############ Prune with optuna prs #######################################
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_per_layer)
            ##########################################################################################
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)
            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {}:{}".format(gen, individual_index, noisy_sample_performance))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)
            current_gen_prs.append(pr_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_sigmas_found = current_gen_sigmas[best_index]
            best_pruning_rates_found = current_gen_prs[best_index]
            if cfg.use_wandb:
                wandb.log({"val_set_accuracy": best_accuracy_found, "generation": gen,
                           "sparsity": sparsity(best_model_found)})

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))

    print("Set of sigmas and pruning rates: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    accuracy_string = "{:10.2f}%".format(performance_best_model_found)
    result = time.localtime(time.time())
    ######Saving the model the pruning
    with open(cfg.save_model_path + "stochastic_pruning_pr_optim_test_accuracy={}_time_{}-{}.pth".format(
            accuracy_string, result.tm_hour, result.tm_min),
              "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "sigmas_per_layers_pr_optim_time_{}-{}.pth".format(result.tm_hour,
                                                                                      result.tm_min),
              "wb") as f:
        pickle.dump(best_sigmas_found, f)
    with open(cfg.save_data_path + "pruning_rate_per_layers_pr_optim_time_{}-{}.pth".format(result.tm_hour,
                                                                                            result.tm_min),
              "wb") as f:
        pickle.dump(best_pruning_rates_found, f)

    if cfg.use_wandb:
        wandb.join()


def stochastic_pruning_with_sigma_optimization_with_erk_layer_wise_prunig_rates(cfg: omegaconf.DictConfig):
    print("Config: \n{}".format(omegaconf.OmegaConf.to_yaml(cfg)))
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"erk_pr_sigma_optimization",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    first_sparsity = 0.5
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    noise = [cfg.sigma] * len(names)
    noise_per_layer = dict(zip(names, noise))
    ############## Craete the study###############################
    study = optuna.create_study(study_name="stochastic pruning "
                                           "with sigma optimization with erk layer wise prunig rates",
                                direction="maximize")

    quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
                                     names=["layer", "q25", "q50", "q75"])
    sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigma_found = None
    current_best_model = original_model
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)

            individual_trial, sigmas_for_individual = get_sigma_sample_per_layer_optuna(individual_trial, lower_bound,
                                                                                        sigma_upper_bound_per_layer,
                                                                                        exclude_layer_list=cfg.exclude_layers)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="erk")
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)
            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))
            # print("Sigmas used:")
            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_sigma_found = current_gen_sigmas[best_index]
            if cfg.use_wandb:
                wandb.log({"val_set_accuracy": best_accuracy_found, "generation": gen,
                           "sparsity": sparsity(best_model_found)})

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp
        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))

    print("Set of sigmas: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    accuracy_string = "{:10.2f}%".format(performance_best_model_found)
    result = time.localtime(time.time())
    with open(cfg.save_model_path + "stochastic_pruning_erdos_renyi_dist_test_accuracy={}_time_{}-{}.pth".format(
            accuracy_string, result.tm_hour, result.tm_min),
              "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "sigmas_per_layers_erdos_renyi_dist_time_{}-{}.pth".format(result.tm_hour,
                                                                                              result.tm_min),
              "wb") as f:
        pickle.dump(best_sigma_found, f)
    if cfg.use_wandb:
        wandb.join()


def enqueue_sigmas(study: optuna.Study, initial_sigmas_per_layer: dict):
    first_trial_dict = {}
    for key, value in initial_sigmas_per_layer.items():
        first_trial_dict["{}_sigma".format(key)] = value
    study.enqueue_trial(first_trial_dict, user_attrs={"memo": "good global value fond by grid "
                                                              "search"})


def enqueue_pr(study: optuna.Study, initial_pr_per_layer: dict):
    first_trial_dict = {}
    for key, value in initial_pr_per_layer.items():
        first_trial_dict[key] = float(value)
    print(omegaconf.OmegaConf.to_yaml(first_trial_dict))
    study.enqueue_trial(first_trial_dict, user_attrs={"memo": "good pr found from literature"})


############################# Ablation experiments #####################################################################


def dynamic_sigma_per_layer_one_shot_pruning(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"

        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"one-shot_{cfg.pruner}_pr_{target_sparsity}_sampler_{cfg.sampler}_sigma_optimization",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))
    print("Deterministic pruning sparsity:  {}".format(sparsity(deterministic_pruning)))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    noise_upper_bound = [0.01] * len(names)
    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good
    noise_initial_value = [0.005] * len(names)
    noise_upper_bound_dict = dict(zip(names, noise_upper_bound))
    noise_initail_value_dict = dict(zip(names, noise_initial_value))
    ############## Craete the study###############################
    sampler = None
    sampler = get_sampler(cfg)
    study = optuna.create_study(study_name="One shot stochastic pruning "
                                           "with sigma optimization per layer",
                                direction="maximize", sampler=sampler)
    ######### Here I set the intial values I want the trial to have which is a global value
    enqueue_sigmas(study, noise_initail_value_dict)
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigma_found = None
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        best_gen_accuracy = 0
        best_gen_model = None
        best_gen_sigma = None

        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(original_model)

            individual_trial, sigmas_for_individual = get_sigma_sample_per_layer_optuna(individual_trial, lower_bound,
                                                                                        noise_upper_bound_dict,
                                                                                        exclude_layer_list=cfg.exclude_layers)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)
            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))

            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))
            if noisy_sample_performance > best_gen_accuracy:
                best_gen_accuracy = noisy_sample_performance
                best_gen_model = noisy_sample
                best_gen_sigma = sigmas_for_individual

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)

        # best_index = np.argmax(current_gen_accuracies)
        # gen_best_accuracy = current_gen_accuracies[best_index]

        if cfg.use_wandb:
            log_dict = {"val_set_accuracy": best_gen_accuracy, "generation": gen,
                        "sparsity": sparsity(best_gen_model),
                        "Deterministic performance": deterministic_pruning_performance}
            log_dict.update(best_gen_sigma)
            wandb.log(log_dict)
        if best_gen_accuracy > best_accuracy_found:
            best_accuracy_found = best_gen_accuracy
            best_model_found = best_gen_model
            best_sigma_found = best_gen_sigma

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

        print("Current best accuracy: {}".format(best_accuracy_found))

    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {}"
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))

    print("Set of sigmas: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_name = cfg.save_model_path + "one_shot_dynamic_sigmas_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, result.tm_hour, result.tm_min)
    model_name = model_name.replace(" ", "")
    with open(model_name, "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "one_shot_dynamic_sigmas_per_layers_{}_dist_pr_{}_sampler_{}.pth".format(cfg.pruner,
                                                                                                            cfg.amount,
                                                                                                            cfg.sampler),
              "wb") as f:
        pickle.dump(best_sigma_found, f)
    if cfg.use_wandb:
        wandb.save(model_name)
        wandb.join()


def static_sigma_per_layer_manually_iterative_process(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_sigma_manual_10_percentile",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    sigmas_for_experiment = get_percentile_per_layer(original_model, 0.1)

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    total_sparse_flops = 0
    first_time = 1
    iter_val_loader = cycle(iter(valloader))
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        batch_for_gen = [next(iter_val_loader)]
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            if cfg.pruner == "global":
                prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
            noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, batch_for_gen, verbose=0,
                                                                     count_flops=True, batch_flops=unit_sparse_flops,
                                                                     one_batch=cfg.one_batch)
            total_sparse_flops += individual_sparse_flops
            print("Generation {} Individual {} sparsity {:0.3f} FLOPS {} Accuracy {}".format(gen, individual_index,
                                                                                             sparsity(noisy_sample),
                                                                                             sparse_flops,
                                                                                             noisy_sample_performance))

            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy}
            wandb.log(log_dict)
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))

    accuracy_string = "{:10.2f}%".format(performance_best_model_found).replace(" ", "")
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "one_shot_manual_sigma_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)

    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def static_sigma_per_layer_optimized_iterative_process(cfg: omegaconf.DictConfig):
    sigmas_for_experiment = None
    try:
        print(cfg.save_data_path + "one_shot_dynamic_sigmas_per_layers_{}_dist_pr_{}_sampler_{}.pth".format(cfg.pruner,
                                                                                                            cfg.amount,
                                                                                                            cfg.sampler))
        with open(cfg.save_data_path + "one_shot_dynamic_sigmas_per_layers_{}_dist_pr_{}_sampler_{}.pth".format(
                cfg.pruner,
                cfg.amount, cfg.sampler),
                  "rb") as f:

            sigmas_for_experiment = pickle.load(f)
    except:
        raise Exception("The respective sigmas file for pruning rate {} and pr per layer {} and sampler {} hasn't been "
                        "generated, "
                        "You need to "
                        "run first \n"
                        "dynamic_sigma_per_layer_one_shot_pruning function with the desired pruning rate and pr per layer".format(
            cfg.amount, cfg.pruner, cfg.sampler))
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    print("Cuda: {}".format(use_cuda))
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_sigma_optimized",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    iter_val_loader = cycle(iter(valloader))
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    total_sparse_flops = 0
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        batch_for_gen = [next(iter_val_loader)]
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################# Pruning ##################################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
            noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, batch_for_gen, verbose=0,
                                                                     count_flops=True, batch_flops=unit_sparse_flops,
                                                                     one_batch=cfg.one_batch)
            total_sparse_flops += individual_sparse_flops
            print("Generation {} Individual {} sparsity {:0.3f} FLOPS {} Accuracy {}".format(gen, individual_index,
                                                                                             sparsity(noisy_sample),
                                                                                             sparse_flops,
                                                                                             noisy_sample_performance))

            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy}
            wandb.log(log_dict)

        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))

    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "iterative_static_sigma_optimized_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner,
        accuracy_string, result.tm_hour, result.tm_min).replace(" ", "")
    with open(model_file_name,
              "wb") as f:
        pickle.dump(best_model_found, f)
    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def static_global_sigma_iterative_process(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    one_batch_string = "_one_batch_per_generation" if cfg.one_batch else "_whole_valset_per_generation "
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_global_static_sigma{one_batch_string}",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    data_loader_iterator = cycle(iter(valloader))
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    data, y = next(data_loader_iterator)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    iter_val_loader = cycle(iter(valloader))
    total_sparse_flops = 0
    first_generation = True
    evaluation_set = valloader
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []

        if cfg.one_batch:
            evaluation_set = [next(iter_val_loader)]
        ################################################################################################################
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)

            noisy_sample = get_noisy_sample(individual, cfg)

            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            noisy_sample_performance, sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                          count_flops=True, batch_flops=unit_sparse_flops)
            total_sparse_flops += sparse_flops
            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))
            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
        ################################################################################################################
        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy,
                        "sigma": cfg.sigma
                        }
            wandb.log(log_dict)

        if first_generation:
            cfg.sigma = cfg.sigma / 2
            first_generation = False
        if gen == int((1 / 3) * cfg.generations):
            cfg.sigma = cfg.sigma / 2
        if gen == int((2 / 3) * cfg.generations):
            cfg.sigma = cfg.sigma / 2

        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ## so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))

    accuracy_string = "{:10.2f}%".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "global_sigma_iterative_stochastic_pruning_pr_{}_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.amount, cfg.pruner,
        accuracy_string, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)
    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def dynamic_sigma_iterative_process(cfg: omegaconf.DictConfig, print_exclude_layers):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    N = cfg.population
    generations = cfg.generations
    original_model = get_model(cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"

        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{target_sparsity}_sampler_{cfg.sampler}_dynamic_sigma",
            reinit=True,
            save_code=True,
        )
    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    noise_upper_bound = [0.01] * len(names)
    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good
    noise_initial_value = [0.005] * len(names)
    noise_upper_bound_dict = dict(zip(names, noise_upper_bound))
    noise_initail_value_dict = dict(zip(names, noise_initial_value))
    ############## Craete the study###############################
    study = optuna.create_study(study_name="Iterative stochastic pruning "
                                           "with sigma optimization per layer",
                                direction="maximize")
    ######### Here I set the intial values I want the trial to have which is a global value
    enqueue_sigmas(study, noise_initail_value_dict)
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigma_found = None
    current_best_model = original_model
    for gen in range(generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)
            individual_trial, sigmas_for_individual = get_sigma_sample_per_layer_optuna(individual_trial, lower_bound,
                                                                                        noise_upper_bound_dict,
                                                                                        exclude_layer_list=cfg.exclude_layers)
            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)

            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {}:{}".format(gen, individual_index, noisy_sample_performance))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance}
            log_dict.update(current_gen_sigmas[best_index])
            wandb.log(log_dict)

        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_sigma_found = current_gen_sigmas[best_index]

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp
        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {}".format(
        performance_best_model_found, cfg.amount))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    ############################################################################
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))
    print("Set of sigmas: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    ######################### Saving the model and the sigmas ##########################################################
    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "iterative_dynamic_sigma_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "iterative_dynamic_sigmas_per_layers_{}_dist_pr_{}.pth".format(cfg.pruner,
                                                                                                  cfg.amount),
              "wb") as f:
        pickle.dump(best_sigma_found, f)
    if cfg.use_wandb:
        wandb.save(
            cfg.save_model_path + "iterative_dynamic_sigma_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
                cfg.pruner, accuracy_string, result.tm_hour, result.tm_min))
        wandb.join()


def compare_weights(weight_list_1: typing.List[torch.TensorType], weight_list_2: typing.List[torch.TensorType]):
    list_equals = []
    list_of_elem = []
    for i, weight1 in enumerate(weight_list_1):
        weight2 = weight_list_2[i]
        list_equals.append((weight1 == weight2).sum())
        list_of_elem.append(weight2.numel())
    return list_equals, list_of_elem


def run_fine_tune_experiment(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    post_pruning_noise_string = "_post_training_noise" if bool(
        cfg.noise_after_pruning) * cfg.measure_gradient_flow else ""

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"deterministic_fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}{post_pruning_noise_string}",
            reinit=True,
        )
    pruned_model = get_model(cfg)
    if cfg.pruner == "global":
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # Add small noise just to get tiny variations of the deterministic case
    initial_performance = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    print("Original version performance: {}".format(initial_performance))
    if cfg.noise_after_pruning and cfg.measure_gradient_flow:
        mask_dict = get_mask(pruned_model)
        names, weights = zip(*get_layer_dict(pruned_model))
        sigma_per_layer = dict(zip(names, [cfg.noise_after_pruning] * len(names)))
        p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        print("p2_model version 1 performance: {}".format(initial_performance))
        apply_mask(p2_model, mask_dict)
        initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
        print("p2_model version 2 performance: {} with sparsity {}".format(initial_performance, sparsity(p2_model)))
        pruned_model = p2_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    # mask_dict = get_mask(pruned_model)
    # p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
    # print("p2_model version 1 performance: {}".format(initial_performance))
    # apply_mask(p2_model,mask_dict)
    # initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    # print("p2_model version 2 performance: {}".format(initial_performance))
    # return
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": initial_performance, "initial_accuracy": initial_performance})
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.2f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/{}/deterministic_LAMP/{}/{}/sigma{}/pr{}/{}/".format(cfg.dataset,
                                                                                                            cfg.architecture,
                                                                                                            cfg.model_type,
                                                                                                            cfg.sigma,
                                                                                                            cfg.amount,
                                                                                                            identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/{}/deterministic_GLOBAL/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"

    restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       cfg=cfg,
                                       gradient_flow_file_prefix=filepath_GF_measure)

    if cfg.use_wandb:
        wandb.join()


@torch.no_grad()
def preditions_on_batch(model1, model2, batch):
    model1.eval()
    model2.eval()
    predictions_1 = model1(batch)
    predictions_2 = model2(batch)
    return predictions_1, predictions_2


def check_correctness(outputs, targets):
    total = 0
    correct = 0
    correct_soft_max = 0
    soft_max_outputs = F.softmax(outputs, dim=1)
    print("soft_max:{}".format(soft_max_outputs))
    _, predicted = torch.max(outputs.data, 1)
    soft_max_pred, predicted_soft_max = torch.max(soft_max_outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    correct_soft_max += predicted_soft_max.eq(targets.data).cpu().sum()

    return total, correct, correct_soft_max, predicted.eq(targets.data).cpu(), soft_max_pred


def run_fine_tune_mask_transfer_experiment(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    ################################## WANDB configuration ############################################
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"mask_transfer_stochastic_fine_tuning",

            notes="",
            reinit=True,
        )
    ################################## Gradient flow measure ############### test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)#############################
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.5f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/mask_transfer_det_sto/{}/stochastic_LAMP/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/mask_transfer_det_sto/{}/stochastic_GLOBAL/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
    pruned_model = get_model(cfg)
    pruned_model.cuda()
    best_model = None
    best_accuracy = -1
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    data, y = data.cuda(), y.cuda()
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = valloader
    if cfg.one_batch:
        evaluation_set = [(data, y)]
    names, weights = zip(*get_layer_dict(pruned_model))
    sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))

    deterministic_pruned_model = copy.deepcopy(pruned_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    if cfg.pruner == "manual":
        prune_with_rate(deterministic_pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner="manual", pr_per_layer=pr_per_layer)
        individual_prs_per_layer = prune_with_rate(copy_of_pruned_model, target_sparsity,
                                                   exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                   pruner="lamp", return_pr_per_layer=True)
        if cfg.use_wandb:
            log_dict = {}
            for name, elem in individual_prs_per_layer.items():
                log_dict["individual_{}_pr".format(name)] = elem
            wandb.log(log_dict)

    if cfg.pruner == "lamp":
        prune_with_rate(deterministic_pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)
    p = test(deterministic_pruned_model, use_cuda, testloader, verbose=0, count_flops=False,
             batch_flops=unit_sparse_flops)
    print("Deterministic model performance: {}".format(p))

    # Go over the population t
    for n in range(cfg.population):
        # current_model = get_noisy_sample(pruned_model, cfg)
        current_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        copy_of_pruned_model = copy.deepcopy(current_model)
        deterministic_pruned_model.cuda()
        copy_buffers(from_net=deterministic_pruned_model, to_net=current_model)
        prune_function(copy_of_pruned_model, cfg)
        remove_reparametrization(copy_of_pruned_model, exclude_layer_list=cfg.exclude_layers)
        det_pred, sto_pred = preditions_on_batch(deterministic_pruned_model, copy_of_pruned_model, data)
        record_predictions(deterministic_pruned_model, testloader, "one_shot_resnet18_det_prediction")
        record_predictions(copy_of_pruned_model, testloader, "one_shot_resnet18_sto_prediction")

        torch.save(det_pred, "deterministic_outputs")
        torch.save(sto_pred, "stochastic_outputs")
        torch.save(y, "labels_of_batch")
        print("Predictions determinsitic: \n {}".format(det_pred))
        print("Predictions stochastic: \n {}".format(sto_pred))
        total, correct_det, correct_soft_det = check_correctness(det_pred, y)
        _, correct_sto, correct_soft_sto = check_correctness(sto_pred, y)
        print("Correct stochastic: {}".format(correct_sto))
        print("Correct stochastic soft_max: {}".format(correct_soft_sto))
        print("Correct deterministic: {}".format(correct_det))
        print("Correct deterministic soft_max: {}".format(correct_soft_det))
        # remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        return
        if first_iter:
            _, unit_sparse_flops = flops(current_model, data)
            first_iter = 0
        noisy_sample_performance, individual_sparse_flops = test(current_model, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)

        print("Deterministic mask on noísy weights model performance on test: {}".format(noisy_sample_performance))
        check_for_layers_collapse(current_model)
        initial_flops += individual_sparse_flops
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = current_model

    initial_performance = test(best_model, use_cuda=use_cuda, testloader=valloader, verbose=1)

    end = time.time()
    initial_test_performance = test(best_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    print("Deterministic mask on noísy weights model performance on test: {}".format(initial_performance))
    total = time.time() - end
    print("Time for testing: {} s".format(total))

    # torch.save({"model_state":best_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_{cfg.pruner}_s{cfg.sigma}_pr{cfg.amount}.pth")

    # if cfg.pruner == "global":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    #
    # if cfg.pruner == "lamp":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
    #                     type="layer-wise",
    #                     pruner=cfg.pruner)
    #
    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    # perfor = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    # torch.save({"model_state":pruned_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_deterministic_{cfg.pruner}_pr{cfg.amount}.pth")

    if cfg.use_wandb:
        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": initial_flops, "initial_test_performance":
            initial_test_performance})

    # remove_reparametrization(best_model,exclude_layer_list=cfg.exclude_layers)

    restricted_fine_tune_measure_flops(best_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       initial_flops=initial_flops,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       gradient_flow_file_prefix=filepath_GF_measure,
                                       cfg=cfg)


def static_sigma_per_layer_manually_iterative_process_flops_counts(cfg: omegaconf.DictConfig, FLOP_limit: float = 1e15):
    FLOP_limit = cfg.flop_limit
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    one_batch_string = "_one_batch" if cfg.one_batch else ""
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_sigma_manual_10_percentile_flops_count{one_batch_string}"
                 f"{cfg.one_batch}",
            notes="This run use the global iterative first generation uses global sigma 0.005 and then uses the "
                  "10th percentile "
                  "of each layer",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    sigmas_for_experiment = get_percentile_per_layer(original_model, 0.1)

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    sparse_flops = 0
    first_time = 1
    iter_val_loader = cycle(iter(valloader))
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    evaluation_set = valloader
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        if cfg.one_batch:
            evaluation_set = [next(iter_val_loader)]
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)
            ###### If the pruning is vanila global then the first gen is with global sigma
            #### after that we do it with the 10th percentile iterative process
            if gen == 0 and cfg.pruner == "global":
                noisy_sample = get_noisy_sample(individual, cfg)
            else:
                noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################## we prune #######################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            #######################################################33
            remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)

            noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                                     count_flops=True, batch_flops=unit_sparse_flops)
            sparse_flops += individual_sparse_flops
            print("Generation {} Individual {} sparsity {:0.3f} FLOPS {} Accuracy {}".format(gen, individual_index,
                                                                                             sparsity(noisy_sample),
                                                                                             sparse_flops,
                                                                                             noisy_sample_performance))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            if FLOP_limit != 0:
                if sparse_flops > FLOP_limit:
                    break

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": sparse_flops,
                        "test_set_accuracy": test_accuracy
                        }
            wandb.log(log_dict)
            if FLOP_limit != 0:
                if sparse_flops > FLOP_limit:
                    break
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            # remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance,
                   "sparse_flops": sparse_flops
                   })
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    #
    accuracy_string = "{:10.2f}".format(performance_best_model_found).replace(" ", "")
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "iterative_manual_sigma_{}_dist_test_accuracy={}_flops_{}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, sparse_flops, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)

    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def stochastic_pruning_with_static_sigma_and_pr_optimization(cfg: omegaconf.DictConfig):
    print("Config: \n{}".format(omegaconf.OmegaConf.to_yaml(cfg)))
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    N = cfg.population
    first_sparsity = 0.5
    original_model = get_model(cfg)
    if cfg.use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_dynamic_prpl_pr_{cfg.amount}_sigma_manual_10_percentile_flops_count",
            reinit=True,
            save_code=True,
        )

    _, pr_lowerbound_per_layer, baseline_non_zero, total_param = erdos_renyi_per_layer_pruning_rate(
        model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    ###############  Sigmas of 10th percentile  #################
    sigmas_for_experiment = get_percentile_per_layer(original_model, 0.1)

    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    count = lambda w: w.nelement()
    number_per_layer = list(map(count, weights))
    number_per_layer = dict(zip(names, number_per_layer))

    ################# Optuna study###########################
    #### Good values of PR per layer dictated by LAMP strategy
    initial_pr_per_layer = prune_with_rate(copy.deepcopy(original_model), target_sparsity,
                                           exclude_layers=cfg.exclude_layers, type="layer-wise", pruner="lamp",
                                           return_pr_per_layer=True)
    initial_pr_per_layer = {f'{k}_pr': v for k, v in initial_pr_per_layer.items()}

    def constraints(trial):
        return trial.user_attrs["constraint"]

    sampler = optuna.samplers.NSGAIISampler(
        constraints_func=constraints,
    )
    study = optuna.create_study(study_name="stochastic pruning with pr optimisation", sampler=sampler,
                                # directions=["maximize","minimize"])
                                direction="maximize")

    ###### Put the first pr in initial #########
    enqueue_pr(study, initial_pr_per_layer)
    total_pruned_weights = count_parameters(original_model) * (cfg.amount - 0.02)

    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])
    # sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    # lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigmas_found = None
    best_pruning_rates_found = None
    current_best_model = original_model
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    total_sparse_flops = 0
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_prs = []
        sample_for_generation = [next(data_loader_iterator)]
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)
            ############### Here I ask for pr ###################################
            individual_trial, pr_for_individual, constrain_value = get_pr_sample_per_layer_optuna(
                individual_trial,
                names, lower_bounds=pr_lowerbound_per_layer,
                number_weights_per_layer=number_per_layer,
                total_number_weights_pruned=total_pruned_weights, exclude_layers=cfg.exclude_layers)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)

            ############ Prune with optuna prs #######################################

            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_for_individual)
            ##########################################################################################
            noisy_sample_performance, sparse_flops = test(noisy_sample, use_cuda, sample_for_generation, verbose=0,
                                                          count_flops=True, batch_flops=unit_sparse_flops,
                                                          one_batch=cfg.one_batch)
            total_sparse_flops += sparse_flops
            # study.tell(individual_trial, [noisy_sample_performance, sparsity(noisy_sample)])
            study.tell(individual_trial, noisy_sample_performance)

            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))
            print("Constrain value:{}".format(constrain_value))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_prs.append(pr_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy
                        }
            log_dict.update(current_gen_prs[best_index])
            wandb.log(log_dict)
        if gen_best_accuracy >= best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_pruning_rates_found = current_gen_prs[best_index]

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trials[0]

    print("  Performance on validation set for set of sigmas: {}".format(trial.values))

    print("Set of sigmas and pruning rates: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    pruning_rate_string = "{:0.3f}".format(sparsity(best_model_found))
    result = time.localtime(time.time())
    ######   Saving the model the pruning
    with open(cfg.save_model_path + "stochastic_pruning_pr_optim_{}_test_accuracy={}_time_{}-{}.pth".format(
            pruning_rate_string, accuracy_string, result.tm_hour, result.tm_min).replace(" ", ""),
              "wb") as f:
        pickle.dump(best_model_found, f)
    auxiliary_list_1 = list(best_pruning_rates_found.values())
    auxiliary_list_2 = list(best_pruning_rates_found.keys())

    dict_1 = {"layer": auxiliary_list_2, "pruning_rate": auxiliary_list_1}
    best_pr_csv = pd.DataFrame(dict_1)
    best_pr_csv.to_csv(cfg.save_data_path + "pruning_rate_per_layers_pr_optim_time_{}-{}.csv".format(result.tm_hour,
                                                                                                     result.tm_min),
                       sep=",", index=False)
    # with open(cfg.save_data_path + "pruning_rate_per_layers_pr_optim_time_{}-{}.pth".format(result.tm_hour,
    #                                                                                         result.tm_min),
    #           "wb") as f:
    #     pickle.dump(best_pruning_rates_found, f)
    #
    if cfg.use_wandb:
        wandb.join()


def generation_of_stochastic_prune_with_efficient_evaluation(solution, target_sparsity, sigmas_for_experiment,
                                                             population, dataloader, image_flops, total_flops, cfg,
                                                             previews_dict_of_image: dict = None):
    surviving_models = []
    first_iteration = 1
    image, y = None, None
    sorted_images = []
    dict_of_images = None
    if previews_dict_of_image is None:
        image, y = get_random_image_label(dataloader)
        dict_of_images = {}
    else:
        sorted_percentages = list(previews_dict_of_image.keys())
        sorted_percentages.sort(reverse=True)
        for percentage in sorted_percentages:
            sorted_images.append(previews_dict_of_image[percentage])
        image, y = sorted_images.pop()
    while len(surviving_models) == 0:
        for i in range(population):
            individual = copy.deepcopy(solution)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################## we prune #######################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            #######################################################33
            # remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)

            prediction = torch.argmax(noisy_sample(image).detach())
            total_flops += image_flops
            if prediction.eq(y.data):
                surviving_models.append(noisy_sample)

        if len(surviving_models) == 0:
            image, y = get_random_image_label(dataloader)

    if len(sorted_images) == 0 and previews_dict_of_image is None:
        dict_of_images[1 - len(surviving_models) / population] = (image, y)
        image, y = get_random_image_label(dataloader)
    if len(sorted_images) == 0 and previews_dict_of_image is not None:
        if len(previews_dict_of_image.keys()) == 1:
            image, y = get_random_image_label(dataloader)
        # else:
        #     previews_dict_of_image[1-len(surviving_models)/population] = (image,y)
        #     image, y = get_random_image_label(dataloader)
    if len(sorted_images) != 0:
        image, y = sorted_images.pop()
    index_to_remove: typing.List[int] = []
    ######## While there
    total_for_image = len(surviving_models)
    while len(surviving_models) > 1:
        for ind in surviving_models:
            prediction = torch.argmax(ind(image).detach())
            total_flops += image_flops
            if not prediction.eq(y.data):
                index_to_remove.append(surviving_models.index(ind))

        if len(index_to_remove) == len(surviving_models):
            if len(sorted_images) == 0:
                image, y = get_random_image_label(dataloader)
            else:
                image, y = sorted_images.pop()
            index_to_remove = []

        else:
            if len(sorted_images) == 0 and previews_dict_of_image is not None:
                previews_dict_of_image[1 - len(surviving_models) / total_for_image] = (image, y)
            if len(sorted_images) == 0 and previews_dict_of_image is None:
                dict_of_images[1 - len(surviving_models) / population] = (image, y)
            for indx in index_to_remove:
                surviving_models.pop(indx)
            total_for_image = len(surviving_models)
    if previews_dict_of_image is None:
        return surviving_models[0], dict_of_images
    else:
        return surviving_models[0]


def efficient_evaluation_random_images(solution, target_sparsity, sigmas_for_experiment,
                                       population, dataloader, image_flops, total_flops, cfg,
                                       previews_dict_of_image: dict = None):
    surviving_models = []

    image, y = get_random_image_label(dataloader)
    image, y = image.cuda(), y.cuda()
    images_used = 1
    solution.cuda()
    print("First loop evaluation")
    while len(surviving_models) == 0:
        for i in range(population):

            individual = copy.deepcopy(solution)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################## we prune #######################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            #######################################################33
            # remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)

            prediction = torch.argmax(noisy_sample(image).detach())
            total_flops += image_flops
            if prediction.eq(y.data):
                surviving_models.append(noisy_sample)

        if len(surviving_models) == 0:
            image, y = get_random_image_label(dataloader)
            images_used += 1
            image, y = image.cuda(), y.cuda()

    print(f"At the end {len(surviving_models)} individuals survived the first loop out of {population} using "
          f"{images_used} images")
    image, y = get_random_image_label(dataloader)
    images_used += 1
    image, y = image.cuda(), y.cuda()

    index_to_remove: typing.List[int] = []
    ######## While there
    total_for_image = len(surviving_models)
    while len(surviving_models) > 1:
        for index, ind in enumerate(surviving_models):
            prediction = torch.argmax(ind(image).detach())
            total_flops += image_flops
            if not prediction.eq(y.data):
                index_to_remove.append(index)
        print("Index to remove {} {}".format(len(index_to_remove), index_to_remove))
        print("Surviving Models {}".format(len(surviving_models)))
        if len(index_to_remove) >= len(surviving_models):
            image, y = get_random_image_label(dataloader)
            images_used += 1
            image, y = image.cuda(), y.cuda()
            index_to_remove = []

        if len(index_to_remove) < len(surviving_models):
            for index in index_to_remove:
                surviving_models.pop(index)

            index_to_remove = []
    print(f"Used in total {images_used} images")
    return surviving_models[0]


def fine_tune_after_stochatic_pruning_experiment(cfg: omegaconf.DictConfig, print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    ################################## WANDB configuration ############################################
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    one_batch_string = "_one_batch_per_generation" if cfg.one_batch else "_whole_dataset_per_generation"
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"fine_tune_base_stochastic_pruning_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
                 f"{non_zero_string}{one_batch_string}",
            notes="This run is to see if gradient clipping is hindering stochastic pruning",
            reinit=True,
        )
    ################################## Gradient flow measure###############test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)#############################
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.5f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/{}/stochastic_LAMP/{}/{}/sigma{}/pr{}/{}/".format(cfg.dataset,
                                                                                                         cfg.architecture,
                                                                                                         cfg.model_type,
                                                                                                         cfg.sigma,
                                                                                                         cfg.amount,
                                                                                                         identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/{}/stochastic_GLOBAL/{}/{}/sigma{}/pr{}/{}/".format(cfg.dataset,
                                                                                                           cfg.architecture,
                                                                                                           cfg.model_type,
                                                                                                           cfg.sigma,
                                                                                                           cfg.amount,
                                                                                                           identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
    pruned_model = get_model(cfg)
    best_model = None
    best_accuracy = -1
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = valloader
    if cfg.one_batch:
        evaluation_set = [(data, y)]
    names, weights = zip(*get_layer_dict(pruned_model))
    sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))

    pr_per_layer = prune_with_rate(copy.deepcopy(pruned_model), target_sparsity, exclude_layers=cfg.exclude_layers,
                                   type="layer-wise",
                                   pruner="lamp", return_pr_per_layer=True)
    if cfg.use_wandb:
        log_dict = {}
        for name, elem in pr_per_layer.items():
            log_dict["deterministic_{}_pr".format(name)] = elem
        wandb.log(log_dict)
    # Go over the population t
    for n in range(cfg.population):
        # current_model = get_noisy_sample(pruned_model, cfg)
        current_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        copy_of_pruned_model = copy.deepcopy(current_model)
        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        if cfg.pruner == "global":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")

        if cfg.pruner == "manual":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_per_layer)
            individual_prs_per_layer = prune_with_rate(copy_of_pruned_model, target_sparsity,
                                                       exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                       pruner="lamp", return_pr_per_layer=True)
            if cfg.use_wandb:
                log_dict = {}
                for name, elem in individual_prs_per_layer.items():
                    log_dict["individual_{}_pr".format(name)] = elem
                wandb.log(log_dict)

        if cfg.pruner == "lamp":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers,
                            type="layer-wise",
                            pruner=cfg.pruner)
        # prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                 pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        if first_iter:
            _, unit_sparse_flops = flops(current_model, data)
            first_iter = 0
        noisy_sample_performance, individual_sparse_flops = test(current_model, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)
        check_for_layers_collapse(current_model)
        initial_flops += individual_sparse_flops
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = current_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    initial_performance = test(best_model, use_cuda=use_cuda, testloader=valloader, verbose=1)

    end = time.time()
    initial_test_performance = test(best_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    total = time.time() - end
    print("Time for testing: {} s".format(total))

    # torch.save({"model_state":best_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_{cfg.pruner}_s{cfg.sigma}_pr{cfg.amount}.pth")

    # if cfg.pruner == "global":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    #
    # if cfg.pruner == "lamp":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
    #                     type="layer-wise",
    #                     pruner=cfg.pruner)
    #
    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    # perfor = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    # torch.save({"model_state":pruned_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_deterministic_{cfg.pruner}_pr{cfg.amount}.pth")

    if cfg.use_wandb:
        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": initial_flops, "initial_test_performance":
            initial_test_performance})

    restricted_fine_tune_measure_flops(best_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       initial_flops=initial_flops,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       gradient_flow_file_prefix=filepath_GF_measure,
                                       cfg=cfg)


#################################### ACCELERATOR STOCHASTIC AND DETERMINISTIC FINE-TUNING ##############################
def fine_tune_after_stochatic_pruning_ACCELERATOR_experiment(cfg: omegaconf.DictConfig, print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    ################################## WANDB configuration ############################################
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    one_batch_string = "_one_batch_per_generation" if cfg.one_batch else "_whole_dataset_per_generation"
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"fine_tune_stochastic",
            notes="This is for imageNet experiments",
            reinit=True,
        )
    ################################## Gradient flow measure###############test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)#############################
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.5f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/ACCELERATOR/{}/stochastic_LAMP/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/ACCELERATOR/{}/stochastic_GLOBAL/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
    pruned_model = get_model(cfg)
    best_model = None
    best_accuracy = -1
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = valloader
    if cfg.one_batch:
        evaluation_set = [(data, y)]
    names, weights = zip(*get_layer_dict(pruned_model))
    sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))

    pr_per_layer = prune_with_rate(copy.deepcopy(pruned_model), target_sparsity, exclude_layers=cfg.exclude_layers,
                                   type="layer-wise",
                                   pruner="lamp", return_pr_per_layer=True)
    if cfg.use_wandb:
        log_dict = {}
        for name, elem in pr_per_layer.items():
            log_dict["deterministic_{}_pr".format(name)] = elem
        wandb.log(log_dict)
    del pr_per_layer
    # Go over the population t
    for n in range(cfg.population):
        # current_model = get_noisy_sample(pruned_model, cfg)
        current_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        if cfg.pruner == "global":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")

        if cfg.pruner == "manual":
            copy_of_pruned_model = copy.deepcopy(current_model)
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_per_layer)
            individual_prs_per_layer = prune_with_rate(copy_of_pruned_model, target_sparsity,
                                                       exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                       pruner="lamp", return_pr_per_layer=True)
            if cfg.use_wandb:
                log_dict = {}
                for name, elem in individual_prs_per_layer.items():
                    log_dict["individual_{}_pr".format(name)] = elem
                wandb.log(log_dict)

        if cfg.pruner == "lamp":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers,
                            type="layer-wise",
                            pruner=cfg.pruner)
        # prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                 pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

        if first_iter:
            _, unit_sparse_flops = flops(current_model, data)
            print("Counted the flops!!")
            print("Flops for batch: {}".format(unit_sparse_flops))
            torch.cuda.empty_cache()
            first_iter = 0
        noisy_sample_performance, individual_sparse_flops = test(current_model, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)
        check_for_layers_collapse(current_model)

        initial_flops += individual_sparse_flops
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = current_model
            del current_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    # torch.save({"model_state":best_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_{cfg.pruner}_s{cfg.sigma}_pr{cfg.amount}.pth")

    # if cfg.pruner == "global":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    #
    # if cfg.pruner == "lamp":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
    #                     type="layer-wise",
    #                     pruner=cfg.pruner)
    #
    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    # perfor = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    # torch.save({"model_state":pruned_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_deterministic_{cfg.pruner}_pr{cfg.amount}.pth")
    del pruned_model
    if cfg.use_wandb:
        initial_performance = test(best_model, use_cuda=use_cuda, testloader=valloader, verbose=1)

        end = time.time()
        initial_test_performance = test(best_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        total = time.time() - end

        print("Time for testing: {} s".format(total))

        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": initial_flops, "initial_test_performance":
            initial_test_performance})

    restricted_IMAGENET_fine_tune_ACCELERATOR_measure_flops(best_model, valloader, testloader,
                                                            FLOP_limit=cfg.flop_limit,
                                                            use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                            exclude_layers=cfg.exclude_layers,
                                                            initial_flops=initial_flops,
                                                            fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                            fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                            gradient_flow_file_prefix=filepath_GF_measure,
                                                            cfg=cfg)


def fine_tune_deterministic_pruning_ACCELERATOR_experiment(cfg: omegaconf.DictConfig, print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    post_pruning_noise_string = "_post_training_noise" if bool(
        cfg.noise_after_pruning) * cfg.measure_gradient_flow else ""

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"deterministic_fine_tune",
            reinit=True,
        )
    pruned_model = get_model(cfg)
    if cfg.pruner == "global":
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # Add small noise just to get tiny variations of the deterministic case
    initial_performance = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    print("Original version performance: {}".format(initial_performance))
    if cfg.noise_after_pruning and cfg.measure_gradient_flow:
        remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
        mask_dict = get_mask(pruned_model)
        names, weights = zip(*get_layer_dict(pruned_model))
        sigma_per_layer = dict(zip(names, [cfg.noise_after_pruning] * len(names)))
        p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        print("p2_model version 1 performance: {}".format(initial_performance))
        apply_mask(p2_model, mask_dict)
        initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
        print("p2_model version 2 performance: {} with sparsity {}".format(initial_performance, sparsity(p2_model)))
        pruned_model = p2_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    # mask_dict = get_mask(pruned_model)
    # p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
    # print("p2_model version 1 performance: {}".format(initial_performance))
    # apply_mask(p2_model,mask_dict)
    # initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    # print("p2_model version 2 performance: {}".format(initial_performance))
    # return
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": initial_performance, "initial_accuracy": initial_performance})
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.2f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/ACCELERATOR/{}/deterministic_LAMP/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/ACCELERATOR/{}/deterministic_GLOBAL/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.architecture, cfg.model_type, cfg.sigma, cfg.amount, identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"

    restricted_IMAGENET_fine_tune_ACCELERATOR_measure_flops(pruned_model, valloader, testloader,
                                                            FLOP_limit=cfg.flop_limit,
                                                            use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                            exclude_layers=cfg.exclude_layers,
                                                            fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                            fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                            gradient_flow_file_prefix=filepath_GF_measure,
                                                            cfg=cfg)


########################################################################################################################
def one_shot_static_sigma_stochastic_pruning(cfg, eval_set="test", print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers" if print_exclude_layers else ""
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"one_shot_stochastic_pruning_static_sigma_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}",
            notes="This experiment is to test if iterative global stochastic pruning, compares to one-shot stochastic pruning",
            reinit=True,
        )

    pruned_model = get_model(cfg)
    best_model = None
    best_accuracy = 0
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = None
    if cfg.one_batch:
        evaluation_set = [data]
    else:
        if eval_set == "test":
            evaluation_set = testloader
        if eval_set == "val":
            evaluation_set = valloader

    for n in range(cfg.population):

        noisy_sample = get_noisy_sample(pruned_model, cfg)

        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        if cfg.pruner == "global":
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)
        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
        if first_iter:
            _, unit_sparse_flops = flops(noisy_sample, data)
            first_iter = 0

        noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)

        initial_flops += individual_sparse_flops
        if cfg.use_wandb:
            test_accuracy = test(noisy_sample, use_cuda, [get_random_batch(testloader)], verbose=0)
            log_dict = {"val_set_accuracy": noisy_sample_performance, "individual": n,
                        "sparse_flops": initial_flops,
                        "test_set_accuracy": test_accuracy
                        }
            wandb.log(log_dict)
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = noisy_sample
            test_accuracy = test(best_model, use_cuda, [get_random_batch(testloader)], verbose=0)
            if cfg.use_wandb:
                log_dict = {"best_val_set_accuracy": best_accuracy, "individual": n,
                            "sparsity": sparsity(best_model),
                            "sparse_flops": initial_flops,
                            "test_set_accuracy": test_accuracy
                            }
                wandb.log(log_dict)


def select_pruning(pruned_model, cfg, target_sparsity, use_stochastic, valloader, sigmas_for_experiment, image_flops,
                   total_flops,
                   dict_of_images=None):
    if use_stochastic:

        return efficient_evaluation_random_images(pruned_model, target_sparsity,
                                                  sigmas_for_experiment,
                                                  cfg.population, valloader,
                                                  image_flops,
                                                  total_flops=total_flops,
                                                  cfg=cfg,
                                                  previews_dict_of_image=dict_of_images)
        # if dict_of_images is None:
        #     best_model,dict_image = generation_of_stochastic_prune_with_efficient_evaluation(pruned_model,
        #                                                                                    target_sparsity,
        #                                                                           sigmas_for_experiment,
        #                                                                           cfg.population, valloader,
        #                                                                           image_flops,
        #                                                                           total_flops=total_flops,
        #                                                                           cfg=cfg,
        #                                                                           previews_dict_of_image=dict_of_images)
        #     return best_model,dict_image
        # else:
        #     best_model = generation_of_stochastic_prune_with_efficient_evaluation(pruned_model,
        #                                                                                       target_sparsity,
        #                                                                                       sigmas_for_experiment,
        #                                                                                       cfg.population, valloader,
        #                                                                                       image_flops,
        #                                                                                       total_flops=total_flops,
        #                                                                                       cfg=cfg,
        #                                                                                       previews_dict_of_image=dict_of_images)
        #     return best_model

    else:
        if cfg.pruner == "global":
            prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)
        return pruned_model


def lamp_scenario_2_cheap_evaluation(cfg):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    original_epoch_number = cfg.epochs
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    full_fine_tune_string = "_full_fine_tune" if cfg.full_fine_tune else "_single_step_fine_tune"
    stochastic_pruning_string = "_stochastic_pruning" if cfg.use_stochastic else "_deterministic_pruning"

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"progressive_pruning_fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
                 f"{non_zero_string}{full_fine_tune_string}{stochastic_pruning_string}",
            reinit=True,
        )
    pruned_model = get_model(cfg)
    for_flops_model = copy.deepcopy(pruned_model)
    if cfg.pruner == "global":
        prune_with_rate(for_flops_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(for_flops_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    sigmas_for_experiment = None
    names, weights = zip(*get_layer_dict(pruned_model))
    list_of_sigma = [cfg.sigma] * len(names)
    sigmas_for_experiment = dict(zip(names, list_of_sigma))

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    image, y = get_random_image_label(valloader)
    _, image_flops = flops(for_flops_model, image)
    TOTAL_FLOPS = 0
    dict_of_images = None
    if cfg.use_stochastic:
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=target_sparsity,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS)
    else:
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=target_sparsity,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS)

    initial_performance = test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
    initial_performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": TOTAL_FLOPS,
                   "test_set_accuracy": initial_performance_test_set, "G": -1,
                   "sparsity": sparsity(model=pruned_model)})
    if cfg.full_fine_tune:

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 0,
                       "sparsity": sparsity(model=pruned_model)})

        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.5,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)

        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 1,
                       "sparsity": sparsity(model=pruned_model)})
        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.8,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 2,
                       "sparsity": sparsity(model=pruned_model)})
    else:

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.short_epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 0,
                       "sparsity": sparsity(model=pruned_model)})
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.5,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)
        performance_val_set = test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "val_set_accuracy": performance_val_set,
                       "sparsity": sparsity(model=pruned_model)})
        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader,
                                                         FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.short_epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 1,
                       "sparsity": sparsity(model=pruned_model)})
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.8,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)
        performance_val_set = test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "val_set_accuracy": performance_val_set,
                       "sparsity": sparsity(model=pruned_model)})
        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        # cfg.epoch = original_epoch_number
        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 2,
                       "sparsity": sparsity(model=pruned_model)})

    last_performance = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    accuracy_string = "{:10.2f}".format(last_performance).replace(" ", "")
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + f"progressive_pruning_fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
    f"{non_zero_string}{full_fine_tune_string}{stochastic_pruning_string}_test_accuracy={accuracy_string}_time_" \
    f"{result.tm_hour}-{result.tm_min}.pth"
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)


def experiment_selector(cfg: omegaconf.DictConfig, args, number_experiment: int = 1):
    if number_experiment == 1:
        dynamic_sigma_per_layer_one_shot_pruning(cfg)
    if number_experiment == 2:
        static_sigma_per_layer_manually_iterative_process(cfg)
    if number_experiment == 3:
        static_sigma_per_layer_optimized_iterative_process(cfg)
    if number_experiment == 4:
        static_global_sigma_iterative_process(cfg)
    if number_experiment == 5:
        dynamic_sigma_iterative_process(cfg)
    if number_experiment == 6:
        run_fine_tune_experiment(cfg)
    if number_experiment == 7:
        static_sigma_per_layer_manually_iterative_process_flops_counts(cfg)
    if number_experiment == 8:
        stochastic_pruning_with_static_sigma_and_pr_optimization(cfg)
    if number_experiment == 9:
        # one_shot_iterative_sotchastic_pruning(cfg)
        pass
    if number_experiment == 10:
        one_shot_static_sigma_stochastic_pruning(cfg, eval_set="val")
    if number_experiment == 11:
        fine_tune_after_stochatic_pruning_experiment(cfg)
    if number_experiment == 12:
        lamp_scenario_2_cheap_evaluation(cfg)
    if number_experiment == 13:
        fine_tune_after_stochatic_pruning_ACCELERATOR_experiment(cfg)
    if number_experiment == 14:
        fine_tune_deterministic_pruning_ACCELERATOR_experiment(cfg)
    if number_experiment == 15:
        print("Began experiment 15")
        sigma_values = [0.001, 0.003, 0.005]
        pruning_rate_values = [0.8, 0.85, 0.9, 0.95]
        architecture_values = ["resnet18", "resnet50", "VGG19"]
        dataset_values = ["cifar100", "cifar10"]

        cfg2 = omegaconf.DictConfig({
            "sigma": 0.001,
            "amount": 0.9,
            "architecture": "resnet18",
            "model_type": "alternative",
            "dataset": "imagenet",
            "set": "test",
            "solution": "",
            # "batch_size": 512,
            "batch_size": 128,
            "num_workers": 10,
        })
        create_ensemble_dataframe(cfg2, sigma_values, architecture_values, pruning_rate_values, dataset_values)
    if number_experiment == 16:
        print("Began experiment 16")
        solution = "/nobackup/sclaam/trained_models/resnet18_imagenet.pth"
        exclude_layers = ["conv1", "fc"]
        cfg2 = omegaconf.DictConfig({
            "population": 5,
            "generations": 10,
            "epochs": 100,
            "short_epochs": 10,
            # "dataset": "mnist",
            # "dataset": "cifar10",
            "dataset": "imagenet",
            "length_test": "small",
            "architecture": "resnet18",
            # "architecture": "VGG19",
            "solution": "/nobackup/sclaam/trained_models/resnet18_imagenet.pth",
            # "solution": "trained_models/mnist/resnet18_MNIST_traditional_train.pth",
            # "solution" : "trained_models/cifar10/resnet50_cifar10.pth",
            # "solution" : "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth",
            # "solution": "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth",
            # "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
            # "solution": "trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth",
            #  "solution": "trained_models/cifar100/vgg19_cifar100_traditional_train.pth",
            # "solution": "trained_models/cifar100/resnet18_cifar100_traditional_train.pth",
            # "solution": "cifar10_resnet20",
            "exclude_layers": ["conv1", "fc"],
            # "exclude_layers": [],
            # "exclude_layers": ["conv1", "linear"],
            # "exclude_layers": ["features.0", "classifier"],
            "noise": "gaussian",
            "pruner": "global",
            # "model_type": "alternative",
            "model_type": "hub",
            "fine_tune_exclude_layers": True,
            "fine_tune_non_zero_weights": True,
            "sampler": "tpe",
            "flop_limit": 0,
            "one_batch": True,
            "measure_gradient_flow": True,
            "full_fine_tune": False,
            "use_stochastic": True,
            # "sigma": 0.0021419609859022197,
            "sigma": 0.001,
            "noise_after_pruning": 0,
            # "amount": 0.944243158936, # for VGG19 to mach 0.9 pruning rate on Resnet 18
            "amount": 0.75,  # For resnet18
            "batch_size": 128,
            # "batch_size": 128,
            "num_workers": 4,
            "save_model_path": "stochastic_pruning_models/",
            "save_data_path": "stochastic_pruning_data/",
            "use_wandb": True
        })

        ### Deterministic pruner vs stochastic pruner based on pruner, dataset, sigma, and pruning rate present on cfg #####
        # cfg = omegaconf.DictConfig({
        #     "sigma":0.001,
        #     "amount":0.9,
        #     "architecture":"resnet18",
        #     "model_type": "hub",
        #     "dataset": "imagenet",
        #     "set":"test"
        # })
        sigmas_ = [0.001, 0.003]
        pruning_rates_ = [0.55, 0.65, 0.7, 0.75]
        pruners_ = ["global", "lamp"]

        best_delta = -float("Inf")
        best_params = None
        # df.to_csv(filepath, mode="a", header=False, index=False)
        df = None
        le_deltas = np.zeros(len(sigmas_) * len(pruning_rates_) * len(pruners_))
        le_deterministic_performances = np.zeros(len(sigmas_) * len(pruning_rates_) * len(pruners_))
        le_quantil_50_performances = np.zeros(len(sigmas_) * len(pruning_rates_) * len(pruners_))

        le_sigmas = np.zeros(len(sigmas_) * len(pruning_rates_) * len(pruners_))
        le_pruning_rates = np.zeros(len(sigmas_) * len(pruning_rates_) * len(pruners_))
        le_pruners = [None] * (len(sigmas_) * len(pruning_rates_) * len(pruners_))

        evaluation_set = select_eval_set(cfg2, "test")
        print("Loaded the dataset {}! ".format(cfg2.dataset))
        i = 0
        for s in sigmas_:
            for pr in pruning_rates_:
                for p in pruners_:
                    cfg2.sigma = s
                    cfg2.amount = pr
                    cfg2.pruner = p
                    t = {"sigma": s, "amount": pr, "pruner": p}
                    print(t)
                    delta, deterministic_performance, quantil_50_performance = stochastic_pruning_against_deterministic_pruning_mean_diference(
                        cfg2, evaluation_set, name="")
                    torch.cuda.empty_cache()
                    le_deltas[i] = delta
                    le_deterministic_performances[i] = deterministic_performance
                    le_quantil_50_performances[i] = quantil_50_performance
                    le_sigmas[i] = s
                    le_pruners[i] = p
                    le_pruning_rates[i] = pr
                    i += 1

                    t.update({"delta": [delta], "Determinstic performance": [deterministic_performance],
                              "quantil_50_performance": [quantil_50_performance], "Population": [cfg2.population]})

                    if delta > best_delta:
                        best_params = t
                        best_delta = delta
                        print("Best delta so far {}".format(best_delta))
                        print("Parameters: {}".format(t))
                        print("Delta {} , Deterministic Performance {} , Quantil 50 performance {}".format(delta,
                                                                                                           deterministic_performance,
                                                                                                           quantil_50_performance))

        print("Best delta is {}".format(best_delta))
        print("For params {}".format(best_params))
        dict = {"sigma": le_sigmas, "pruning_rate": le_pruning_rates, "pruner": le_pruners, "delta": le_deltas,
                "Determinstic performance": le_deterministic_performances,
                "quantil_50_performance": le_quantil_50_performances,
                "Population": [cfg2.population] * len(le_quantil_50_performances)}
        df = pd.DataFrame(dict)

        df.to_csv("{}_pr_sigma_combination_pop_{}.csv".format(cfg2.dataset, cfg2.population), index=False)
    if number_experiment == 17:
        run_fine_tune_mask_transfer_experiment(cfg)
    if number_experiment == 18:
        run_pr_sigma_search_for_cfg(cfg, args)
    if number_experiment == 19:
        run_pr_sigma_search_MOO_for_cfg(cfg, args)
    # if number_experiment == 13:


def test_sigma_experiment_selector():
    test_cfg = omegaconf.DictConfig({
        "population": 2,
        "generations": 2,
        "architecture": "resnet18",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
        "noise": "gaussian",
        "pruner": "erk",
        "type": "alternative",
        "exclude_layers": ["conv1", "linear"],
        # "sigma": 0.0021419609859022197,
        "sigma": 0.005,
        "amount": 0.9,
        "batch_size": 512,
        "num_workers": 0,
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "use_wandb": False
    })
    experiment_selector(test_cfg, 1)
    experiment_selector(test_cfg, 2)
    experiment_selector(test_cfg, 3)
    experiment_selector(test_cfg, 4)
    experiment_selector(test_cfg, 5)


############################### Plot of stochastic pruning against deterministic pruning ###############################

def stochastic_pruning_against_deterministic_pruning_mean_diference(cfg: omegaconf.DictConfig,
                                                                    evaluation_set: torch.utils.data.DataLoader,
                                                                    name: str = ""):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    # accelerator = accelerate.Accelerator(mixed_precision="fp16")
    # evaluation_set , net = accelerator.prepare(evaluation_set,net)

    # t0 = time.time()
    # original_performance = test_with_accelerator(net,evaluation_set, verbose=0,accelerator=accelerator)
    pruned_original_performance = test(net, use_cuda, evaluation_set, verbose=0)
    # t1 = time.time()
    # print("Time for test: {}".format(t1-t0))
    pruned_original = copy.deepcopy(net)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [cfg.sigma] * number_of_layers))

    if cfg.pruner == "global":
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    remove_reparametrization(pruned_original, exclude_layer_list=cfg.exclude_layers)
    print("pruned_performance of pruned original")
    # t0 = time.time()
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=0)
    # t1 = time.time()
    # print("Time for test: {}".format(t1-t0))
    del pruned_original
    # pop.append(pruned_original)
    # pruned_performance.append(pruned_original_performance)
    labels = []
    # stochastic_dense_performances.append(original_performance)
    for n in range(N):
        current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # print("Stochastic dense performance")
        # t0 = time.time()
        # StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        # t1 = time.time()
        # print("Time for test: {}".format(t1-t0))
        # Dense stochastic performance
        # stochastic_dense_performances.append(StoDense_performance)

        if cfg.pruner == "global":
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

        torch.cuda.empty_cache()
        # print("Stocastic pruning performance")
        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=0)
        # print("Time for test: {}".format(t1-t0))

        pruned_performance.append(stochastic_pruned_performance)
        # stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
        del current_model
        torch.cuda.empty_cache()

    quantil_50 = np.quantile(pruned_performance, 0.5)

    quantil_50_delta = (quantil_50 - pruned_original_performance)

    return quantil_50_delta, pruned_original_performance, quantil_50


def stochastic_pruning_against_deterministic_pruning(cfg: omegaconf.DictConfig, eval_set: str = "test", name: str = ""):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    # accelerator = accelerate.Accelerator(mixed_precision="fp16")
    # evaluation_set, net = accelerator.prepare(evaluation_set, net)

    t0 = time.time()
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)
    t1 = time.time()
    print("Time for test: {}".format(t1 - t0))
    pruned_original = copy.deepcopy(net)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [cfg.sigma] * number_of_layers))

    if cfg.pruner == "global":
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    remove_reparametrization(pruned_original, exclude_layer_list=cfg.exclude_layers)
    # record_predictions(pruned_original, evaluation_set,
    #                    "{}_one_shot_det_{}_predictions_{}".format(cfg.architecture, cfg.model_type, cfg.dataset))
    print("pruned_performance of pruned original")
    t0 = time.time()
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)
    print("Det_performance in function: {}".format(pruned_original_performance))
    t1 = time.time()
    print("Time for test: {}".format(t1 - t0))
    del pruned_original
    # pop.append(pruned_original)
    # pruned_performance.append(pruned_original_performance)
    labels = []
    # stochastic_dense_performances.append(original_performance)
    for n in range(N):
        current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        t0 = time.time()
        StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        t1 = time.time()
        print("Time for test: {}".format(t1 - t0))
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)

        if cfg.pruner == "global":
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        # record_predictions(current_model, evaluation_set,
        #                    "{}_one_shot_sto_{}_predictions_{}".format(cfg.architecture, cfg.model_type, cfg.dataset))
        torch.cuda.empty_cache()
        print("Stocastic pruning performance")
        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        print("Time for test: {}".format(t1 - t0))

        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
        del current_model
        torch.cuda.empty_cache()

    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance)))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    assert len(labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                   f"{len(labels)}!={len(pruned_performance)}"

    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)
    all_index = np.ones(len(ranked_index), dtype=bool)
    all_index[index_of_pruned_original] = False
    ranked_index = ranked_index[all_index]

    pruned_performance = np.array(pruned_performance)
    stochastic_dense_performances = np.array(stochastic_dense_performances)
    result = time.localtime(time.time())

    del pop
    cutoff = original_performance - 2
    ################################# plotting the comparison #########################################################
    fig, ax = plt.subplots(figsize=fig_size, layout="compressed")

    original_line = ax.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")

    deterministic_pruning_line = ax.axhline(y=pruned_original_performance, c="purple", label="Deterministic Pruning")
    plt.xlabel("Ranking Index", fontsize=fs)
    plt.ylabel("Accuracy", fontsize=fs)
    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    for i, element in enumerate(pruned_performance[ranked_index]):
        # if i == index_of_pruned_original:
        #     # assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
        #     #                                                f"in list {element} VS pruned performance:" \
        #     #                                                f" {pruned_original_performance}"
        #     # p1 = ax.scatter(i, element, c="g", marker="o")
        # else:
        if labels[ranked_index[i]] == "sto mask transfer":
            sto_transfer_point = ax.scatter(i, element, c="tab:orange", marker="P")
            transfer_mask_models_points.append(sto_transfer_point)
        elif labels[ranked_index[i]] == "det mask transfer":
            det_transfer_point = ax.scatter(i, element, c="tab:olive", marker="X")
            stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
        else:
            pruned_point = ax.scatter(i, element, c="steelblue", marker="x")
            stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # ax.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = ax.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([original_line, tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense),
                deterministic_pruning_line],
               ['Original Performance', 'Pruned Stochastic', 'Dense Stochastic', "Deterministic Pruning"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.grid(ls='--', alpha=0.5)
    # plt.ylim(0, 100)
    #
    # ax2 = ax.twinx()
    #
    # ytickslocs = ax.get_yticks()
    # _, ymx = ax.get_ylim()
    #
    #
    # y_ticks = ax.transData.transform([(tick,_) for tick in ytickslocs])
    #
    #
    # epsilon_ticks =  np.linspace(original_performance, 0, len(y_ticks)-1)
    #
    # ax2.set_yticks(ticks=epsilon_ticks, minor=False)
    # ax2.set_yticklabels(epsilon_ticks)
    # ax2.set_ylabel(r"Accuracy degradation-$\epsilon$ ", fontsize=15)
    # ax2.spines['right'].set_color('red')
    # ax2.tick_params(axis="y", colors="red")
    # ax2.yaxis.label.set_color('red')
    # ax2.invert_yaxis()

    # plt.tight_layout()
    plt.savefig(
        f"data/figures/_{cfg.dataset}_{cfg.pruner}_{cfg.architecture}_stochastic_deterministic_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}_{eval_set}_{name}.pdf")
    plt.savefig(
        f"data/figures/_{cfg.dataset}_{cfg.pruner}_{cfg.architecture}_stochastic_deterministic_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}_{eval_set}_{name}.pgf")


# def stochastic_pruning_global_against_LAMP_deterministic_pruning():
def stochastic_pruning_global_against_LAMP_deterministic_pruning(cfg: omegaconf.DictConfig,
                                                                 eval_set: str = "test") -> object:
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    deterministic_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)

    lamp_pruned_original: typing.Union[ResNet, None, VGG, nn.Module] = copy.deepcopy(net)

    prune_with_rate(lamp_pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                    pruner="lamp")
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(lamp_pruned_original, use_cuda, evaluation_set, verbose=1)
    pop.append(lamp_pruned_original)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        current_model = copy.deepcopy(net)

        # current_model = get_noisy_sample(net, cfg)

        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        # StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        # Dense stochastic performance
        # stochastic_dense_performances.append(StoDense_performance)

        prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner="lamp", is_stochastic=True, noise_type="gaussian", noise_amplitude=cfg.sigma)
        # prune_with_rate(current_model, amount=cfg.amount, exclude_layers=cfg.exclude_layers)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        pruned_performance.append(stochastic_pruned_performance)
        # stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)

    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance) - 1))
    # labels.extend(["sto mask transfer"] * len(deterministic_with_stochastic_mask_performance))
    # pruned_performance.extend(deterministic_with_stochastic_mask_performance)
    # stochastic_dense_performances.extend([1] * len(deterministic_with_stochastic_mask_performance))
    # # Deterministic mask transfer to stochastic model
    # labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    # pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    # stochastic_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    assert len(labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                   f"{len(labels)}!={len(pruned_performance)}"
    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)

    pruned_performance = np.array(pruned_performance)
    # stochastic_dense_performances = np.array(stochastic_dense_performances)
    result = time.localtime(time.time())

    del pop
    cutoff = original_performance - 2
    ################################# plotting the comparison #########################################################
    fig, ax = plt.subplots()

    original_line = ax.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")

    deterministic_pruning_line = ax.axhline(y=pruned_original_performance, c="purple", label="Deterministic Pruning")
    # plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking Index", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    # if cfg.noise == "geogaussian":
    #     plt.title(f"CIFAR10 Geometric Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)
    # if cfg.noise == "gaussian":
    #     plt.title(f"CIFAR10 Additive Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)

    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    p1 = None

    for i, element in enumerate(pruned_performance[ranked_index]):
        # if i == index_of_pruned_original:
        #     assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
        #                                                    f"in list {element} VS pruned performance:" \
        #                                                    f" {pruned_original_performance}"
        # p1 = ax.scatter(i, element, c="g", marker="o")
        # else:
        # if labels[ranked_index[i]] == "sto mask transfer":
        #     sto_transfer_point = ax.scatter(i, element, c="tab:orange", marker="P")
        #     transfer_mask_models_points.append(sto_transfer_point)
        # elif labels[ranked_index[i]] == "det mask transfer":
        #     det_transfer_point = ax.scatter(i, element, c="tab:olive", marker="X")
        #     stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
        # else:
        pruned_point = ax.scatter(i, element, c="steelblue", marker="x")
        stochastic_models_points_pruned.append(pruned_point)
    # for i, element in enumerate(stochastic_dense_performances[ranked_index]):
    #     if i == index_of_pruned_original or element == 1:
    #         continue
    #         # ax.scatter(i, element, c="y", marker="o", label="original model performance")
    #     else:
    #         dense_point = ax.scatter(i, element, c="c", marker="1")
    #         stochastic_models_points_dense.append(dense_point)

    plt.legend([original_line, tuple(stochastic_models_points_pruned),
                deterministic_pruning_line],
               ['Original Performance', 'Stochastic Lamp scores', "LAMP deterministic Pruned"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    # plt.ylim(5, 100)
    # ax2 = ax.twinx()
    # # y1_tick_positions = ax.get_ticks()
    # epsilon_ticks = original_performance - np.linspace(20, 100, 9)
    # ax2.set_yticks(epsilon_ticks, minor=False)
    # ax2.set_ylabel(r"$\epsilon$", fontsize=20)
    # ax2.spines['right'].set_color('red')
    # ax2.tick_params(axis="y", colors="red")
    # ax2.yaxis.label.set_color('red')
    # ax2.invert_yaxis()
    # ax2 = ax.twinx()
    # y1_tick_positions = ax.get_ticks()
    # epsilon_ticks = original_performance - np.linspace(88,100, 7)
    # ax2.set_yticks(np.flip(epsilon_ticks), minor=False)
    # ax2.tick_params(axis="y", colors="red")

    plt.tight_layout()
    plt.savefig(
        f"data/figures/LAMP_stochastic_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}_{eval_set}_{cfg.architecture}.pdf")
    plt.savefig(f"data/figures/LAMP_stochastic_{cfg.noise}_sigma_"
                f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
                f"_{cfg.population}_{eval_set}_{cfg.architecture}.png")


############################# Scatter plots #############################################################

def scatter_plot_sigmas(dataFrame1: pd.DataFrame, dataFrame2: pd.DataFrame, deterministic_dataframe1: pd.DataFrame,
                        deterministic_dataframe2: pd.DataFrame, det_label1: str, det_label2: str, title: str = "",
                        file: str = "", use_set="val", sigmas_to_show=[]):
    all_df1: pd.DataFrame = None
    all_df2: pd.DataFrame = None
    for sigma in dataFrame1["sigma"].unique():
        sigma_temp_df = dataFrame1[dataFrame1["sigma"] == sigma]
        gradient_flow = []
        accuracy = []
        type = []
        sigma_list = []
        for elem in sigma_temp_df["individual"].unique():
            # One-shot data
            temp_df = sigma_temp_df[sigma_temp_df["individual"] == elem]
            if use_set == "val":
                gradient_flow.append(float(temp_df['val_set_gradient_magnitude'][temp_df["Epoch"] == -1].iloc[0]))
                accuracy.append(float(temp_df["val_accuracy"][temp_df["Epoch"] == -1]))
                type.append("One-Shot")
                sigma_list.append(sigma)
                # Now the fine-tuned data
                gradient_flow.append(float(temp_df.iloc[len(temp_df) - 1]["val_set_gradient_magnitude"]))
                print(temp_df["Epoch"])
                accuracy.append(float(temp_df["val_accuracy"][temp_df["Epoch"] == temp_df["Epoch"].max()].iloc[0]))
                type.append("Fine-tuned")
                sigma_list.append(sigma)
            elif use_set == "test":
                gradient_flow.append(float(temp_df['test_set_gradient_magnitude'][temp_df["Epoch"] == -1].iloc[0]))
                accuracy.append(float(temp_df["test_accuracy"][temp_df["Epoch"] == -1]))
                type.append("One-Shot")
                sigma_list.append(sigma)
                # Now the fine-tuned data
                gradient_flow.append(float(temp_df.iloc[len(temp_df) - 1]["test_set_gradient_magnitude"]))
                print(temp_df["Epoch"])
                accuracy.append(float(temp_df["test_accuracy"][temp_df["Epoch"] == temp_df["Epoch"].max()].iloc[0]))
                type.append("Fine-tuned")
                sigma_list.append(sigma)

        d = pd.DataFrame(
            {"Gradient Magnitude": gradient_flow,
             "Accuracy": accuracy,
             "Type": type,
             "Sigma": sigma

             }
        )

        if all_df1 is None:
            all_df1 = d
        else:
            all_df1 = pd.concat((all_df1, d), ignore_index=True)
    plt.figure()
    plt.subplots(figsize=fig_size, layout="compressed")
    if not sigmas_to_show:
        g = sns.scatterplot(data=all_df1, x="Gradient Magnitude", y="Accuracy", hue="Sigma", style="Type",
                            palette="deep", legend="full", edgecolor=None, linewidth=0)
    else:
        # bool_index_vector = all_df1["Sigma"] == sigmas_to_show.pop()
        bool_index_vector = np.zeros(len(all_df1))
        for s in sigmas_to_show:
            bool_index_vector = np.logical_or(bool_index_vector, all_df1["Sigma"] == s)
        sigma_df = all_df1[bool_index_vector]
        g = sns.scatterplot(data=sigma_df, x="Gradient Magnitude", y="Accuracy", hue="Sigma", style="Type",
                            palette="deep", legend="full", edgecolor="black", linewidth=0.1)

    # plt.xlabel(fontsize=20)
    # plt.ylabel(fontsize=20)
    # plt.scatter(,label="")
    plt.title("")

    # Deterministic dataframe 1

    # Get first row using row position
    individual_1 = deterministic_dataframe1.iloc[0]["individual"]
    deterministic_dataframe = deterministic_dataframe1[deterministic_dataframe1["individual"] == individual_1]
    if use_set == "val":

        deterministic_initial_gradient_fow = float(
            deterministic_dataframe['val_set_gradient_magnitude'][deterministic_dataframe["Epoch"] == -1].iloc[0])
        deterministic_final_gradient_fow = float(
            deterministic_dataframe.iloc[len(deterministic_dataframe) - 1]["val_set_gradient_magnitude"])
        deterministic_initial_accuracy = float(
            deterministic_dataframe["val_accuracy"][deterministic_dataframe["Epoch"] == -1])
        deterministic_final_accuracy = float(deterministic_dataframe["val_accuracy"][
                                                 deterministic_dataframe["Epoch"] == deterministic_dataframe[
                                                     "Epoch"].max()].iloc[0])

    elif use_set == "test":

        deterministic_initial_gradient_fow = float(
            deterministic_dataframe['test_set_gradient_magnitude'][deterministic_dataframe["Epoch"] == -1].iloc[0])
        deterministic_final_gradient_fow = float(
            deterministic_dataframe.iloc[len(deterministic_dataframe) - 1]["test_set_gradient_magnitude"])
        deterministic_initial_accuracy = float(
            deterministic_dataframe["test_accuracy"][deterministic_dataframe["Epoch"] == -1])
        deterministic_final_accuracy = float(deterministic_dataframe["test_accuracy"][
                                                 deterministic_dataframe["Epoch"] == deterministic_dataframe[
                                                     "Epoch"].max()].iloc[0])

    plt.scatter(x=deterministic_initial_gradient_fow, y=deterministic_initial_accuracy, marker='^', s=40, c='crimson',
                edgecolors='crimson', label=f"One-shot {det_label1}")
    plt.scatter(x=deterministic_final_gradient_fow, y=deterministic_final_accuracy, marker='x', s=40, c='crimson',
                label=f"Fine-Tuned {det_label1}")

    # Deterministic dataframe 2
    individual_2 = deterministic_dataframe2.iloc[0]["individual"]
    deterministic_dataframe = deterministic_dataframe2[deterministic_dataframe2["individual"] == individual_2]

    if use_set == "val":
        deterministic_initial_gradient_fow = float(
            deterministic_dataframe['val_set_gradient_magnitude'][deterministic_dataframe["Epoch"] == -1].iloc[0])
        deterministic_final_gradient_fow = float(
            deterministic_dataframe.iloc[len(deterministic_dataframe) - 1]["val_set_gradient_magnitude"])
        deterministic_initial_accuracy = float(
            deterministic_dataframe["val_accuracy"][deterministic_dataframe["Epoch"] == -1])
        deterministic_final_accuracy = float(deterministic_dataframe["val_accuracy"][
                                                 deterministic_dataframe["Epoch"] == deterministic_dataframe[
                                                     "Epoch"].max()].iloc[0])
    elif use_set == "test":
        deterministic_initial_gradient_fow = float(
            deterministic_dataframe['test_set_gradient_magnitude'][deterministic_dataframe["Epoch"] == -1].iloc[0])
        deterministic_final_gradient_fow = float(
            deterministic_dataframe.iloc[len(deterministic_dataframe) - 1]["test_set_gradient_magnitude"])
        deterministic_initial_accuracy = float(
            deterministic_dataframe["test_accuracy"][deterministic_dataframe["Epoch"] == -1])
        deterministic_final_accuracy = float(deterministic_dataframe["test_accuracy"][
                                                 deterministic_dataframe["Epoch"] == deterministic_dataframe[
                                                     "Epoch"].max()].iloc[0])

    plt.scatter(x=deterministic_initial_gradient_fow, y=deterministic_initial_accuracy, marker='v', s=40,
                c='dodgerblue', edgecolors='dodgerblue', label=f"One-shot {det_label2}")
    plt.scatter(x=deterministic_final_gradient_fow, y=deterministic_final_accuracy, marker='x', s=40, c='dodgerblue',
                label=f"Fine-Tuned {det_label2}")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)
    plt.grid(ls="--", alpha=0.5)
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(10, 10)
    # plt.xlim(0,2.5)
    plt.savefig(file, bbox_inches="tight")


def gradient_flow_especific_combination_dataframe_generation_stochastic_only(prefix: str, cfg, min_epochs=11,
                                                                             surname=""):
    '''
    This function is for unifying the results of a particular exp
    @rtype: object
    @param prefix:This is the folder were deterministic LAMP/GLOBAL and stochastic LAMP/GLOBAL folders (with all the subfolder struceture) reside. Should contain the same dataset name as in cfg.
    @param cfg: Configuration for a particular architceture X sigma X pruning rate combination to create a dataframe with all individuals results. The dataset is
    present in the prefix string
    '''
    # prefix = Path(prefix)
    middle_string = cfg.model_type
    # if cfg.model_type == "hub" :
    #     middle_string = "/hub"

    assert cfg.dataset in prefix, "Prefix does not contain the name of the dataset: {}!={}".format(cfg.dataset, prefix)
    # If I want to unify the mask transfer.
    stochastic_global_root = prefix + "stochastic_GLOBAL/" + f"{cfg.architecture}/{cfg.model_type}/sigma{cfg.sigma}/pr{cfg.amount}/"
    stochastic_lamp_root = prefix + "stochastic_LAMP/" + f"{cfg.architecture}/{cfg.model_type}/sigma{cfg.sigma}/pr{cfg.amount}/"

    combine_stochastic_GLOBAL_DF: pd.DataFrame = None
    combine_stochastic_LAMP_DF: pd.DataFrame = None

    ########################### Global Determinisitc ########################################
    ########################### Lamp Deterministic  ########################################
    ########################## First Global stochatic #######################################
    for index, individual in enumerate(glob.glob(stochastic_global_root + "*/", recursive=True)):
        individual_df = pd.read_csv(individual + "recordings.csv", sep=",", header=0, index_col=False)
        len_df = individual_df.shape[0]
        if len_df < min_epochs:
            continue
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        individual_df["Architecture"] = [cfg.architecture] * len_df
        individual_df["Pruner"] = ["GMP"] * len_df
        individual_df["Dataset"] = [cfg.dataset] * len_df
        individual_df["Pruning Rate"] = [cfg.amount] * len_df
        if combine_stochastic_GLOBAL_DF is None:
            combine_stochastic_GLOBAL_DF = individual_df
        else:
            combine_stochastic_GLOBAL_DF = pd.concat((combine_stochastic_GLOBAL_DF, individual_df), ignore_index=True)

    combine_stochastic_GLOBAL_DF.to_csv(
        f"gradientflow_stochastic_global_{surname}{cfg.architecture}_{cfg.dataset}_sigma_{cfg.sigma}_pr{cfg.amount}.csv",
        header=True, index=False)
    ########################## Second LAMP stochatic #######################################

    for index, individual in enumerate(glob.glob(stochastic_lamp_root + "*/", recursive=True)):
        individual_df = pd.read_csv(individual + "recordings.csv", sep=",", header=0, index_col=False)
        len_df = individual_df.shape[0]
        if len_df < min_epochs:
            continue
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        individual_df["Pruner"] = ["LAMP"] * len_df
        individual_df["Architecture"] = [cfg.architecture] * len_df
        individual_df["Dataset"] = [cfg.dataset] * len_df
        individual_df["Pruning Rate"] = [cfg.amount] * len_df
        if combine_stochastic_LAMP_DF is None:
            combine_stochastic_LAMP_DF = individual_df
        else:
            combine_stochastic_LAMP_DF = pd.concat((combine_stochastic_LAMP_DF, individual_df), ignore_index=True)

    combine_stochastic_LAMP_DF.to_csv(
        f"gradientflow_stochastic_lamp_{surname}{cfg.architecture}_{cfg.dataset}_sigma_{cfg.sigma}_pr{cfg.amount}.csv",
        header=True, index=False)


def gradient_flow_especific_combination_dataframe_generation(prefix: str, cfg, min_epochs=11, is_mask_transfer=False):
    '''
    This function is for unifying the results of a particular exp
    @rtype: object
    @param prefix:This is the folder were deterministic LAMP/GLOBAL and stochastic LAMP/GLOBAL folders (with all the subfolder struceture) reside. Should contain the same dataset name as in cfg.
    @param cfg: Configuration for a particular architceture X sigma X pruning rate combination to create a dataframe with all individuals results. The dataset is
    present in the prefix string
    '''
    # prefix = Path(prefix)
    middle_string = cfg.model_type
    # if cfg.model_type == "hub" :
    #     middle_string = "/hub"

    assert cfg.dataset in prefix, "Prefix does not contain the name of the dataset: {}!={}".format(cfg.dataset, prefix)
    deterministic_lamp_root = prefix + "deterministic_LAMP/" + f"{cfg.architecture}/{cfg.model_type}/sigma0.0/pr{cfg.amount}/"

    deterministic_global_root = prefix + "deterministic_GLOBAL/" + f"{cfg.architecture}/{cfg.model_type}/sigma0.0/pr{cfg.amount}/"

    stochastic_global_root = prefix + "stochastic_GLOBAL/" + f"{cfg.architecture}/{cfg.model_type}/sigma{cfg.sigma}/pr{cfg.amount}/"
    stochastic_lamp_root = prefix + "stochastic_LAMP/" + f"{cfg.architecture}/{cfg.model_type}/sigma{cfg.sigma}/pr{cfg.amount}/"

    combine_stochastic_GLOBAL_DF: pd.DataFrame = None
    combine_stochastic_LAMP_DF: pd.DataFrame = None
    combine_deterministic_GOBAL_DF: pd.DataFrame = None
    combine_deterministic_LAMP_DF: pd.DataFrame = None

    ########################### Global Determinisitc ########################################

    for index, individual in enumerate(glob.glob(deterministic_global_root + "*/", recursive=True)):
        individual_df = pd.read_csv(individual + "recordings.csv", sep=",", header=0, index_col=False)
        len_df = individual_df.shape[0]
        if len_df < min_epochs:
            continue
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [0] * len_df
        individual_df["Pruner"] = ["GMP"] * len_df
        individual_df["Architecture"] = [cfg.architecture] * len_df
        individual_df["Dataset"] = [cfg.dataset] * len_df
        individual_df["Pruning Rate"] = [cfg.amount] * len_df
        if combine_deterministic_GOBAL_DF is None:
            combine_deterministic_GOBAL_DF = individual_df
        else:
            combine_deterministic_GOBAL_DF = pd.concat((combine_deterministic_GOBAL_DF, individual_df),
                                                       ignore_index=True)

    combine_deterministic_GOBAL_DF.to_csv(
        f"gradientflow_deterministic_global_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv", header=True,
        index=False)

    ########################### Lamp Deterministic  ########################################

    for index, individual in enumerate(glob.glob(deterministic_lamp_root + "*/", recursive=True)):
        individual_df = pd.read_csv(individual + "recordings.csv", sep=",", header=0, index_col=False)
        len_df = individual_df.shape[0]
        if len_df < min_epochs:
            continue
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [0] * len_df
        individual_df["Pruner"] = ["LAMP"] * len_df
        individual_df["Architecture"] = [cfg.architecture] * len_df
        individual_df["Dataset"] = [cfg.dataset] * len_df
        individual_df["Pruning Rate"] = [cfg.amount] * len_df
        if combine_deterministic_LAMP_DF is None:
            combine_deterministic_LAMP_DF = individual_df
        else:
            combine_deterministic_LAMP_DF = pd.concat((combine_deterministic_LAMP_DF, individual_df), ignore_index=True)

    combine_deterministic_LAMP_DF.to_csv(
        f"gradientflow_deterministic_lamp_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv", header=True,
        index=False)

    ########################## first Global stochatic #######################################
    for index, individual in enumerate(glob.glob(stochastic_global_root + "*/", recursive=True)):
        individual_df = pd.read_csv(individual + "recordings.csv", sep=",", header=0, index_col=False)
        len_df = individual_df.shape[0]
        if len_df < min_epochs:
            continue
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        individual_df["Architecture"] = [cfg.architecture] * len_df
        individual_df["Pruner"] = ["GMP"] * len_df
        individual_df["Dataset"] = [cfg.dataset] * len_df
        individual_df["Pruning Rate"] = [cfg.amount] * len_df
        if combine_stochastic_GLOBAL_DF is None:
            combine_stochastic_GLOBAL_DF = individual_df
        else:
            combine_stochastic_GLOBAL_DF = pd.concat((combine_stochastic_GLOBAL_DF, individual_df), ignore_index=True)

    combine_stochastic_GLOBAL_DF.to_csv(
        f"gradientflow_stochastic_global_{cfg.architecture}_{cfg.dataset}_sigma_{cfg.sigma}_pr{cfg.amount}.csv",
        header=True, index=False)
    ########################## Second LAMP stochatic #######################################

    for index, individual in enumerate(glob.glob(stochastic_lamp_root + "*/", recursive=True)):
        individual_df = pd.read_csv(individual + "recordings.csv", sep=",", header=0, index_col=False)
        len_df = individual_df.shape[0]
        if len_df < min_epochs:
            continue
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        individual_df["Pruner"] = ["LAMP"] * len_df
        individual_df["Architecture"] = [cfg.architecture] * len_df
        individual_df["Dataset"] = [cfg.dataset] * len_df
        individual_df["Pruning Rate"] = [cfg.amount] * len_df
        if combine_stochastic_LAMP_DF is None:
            combine_stochastic_LAMP_DF = individual_df
        else:
            combine_stochastic_LAMP_DF = pd.concat((combine_stochastic_LAMP_DF, individual_df), ignore_index=True)

    combine_stochastic_LAMP_DF.to_csv(
        f"gradientflow_stochastic_lamp_{cfg.architecture}_{cfg.dataset}_sigma_{cfg.sigma}_pr{cfg.amount}.csv",
        header=True, index=False)


def unify_sigma_datasets(sigmas: list, cfg: omegaconf.DictConfig, surname=""):
    combine_stochastic_GLOBAL_DF: pd.DataFrame = None
    combine_stochastic_LAMP_DF: pd.DataFrame = None
    first_sigma = sigmas.pop()

    combine_stochastic_LAMP_DF = pd.read_csv(
        f"gradientflow_stochastic_lamp_{surname}{cfg.architecture}_{cfg.dataset}_sigma_{first_sigma}_pr{cfg.amount}.csv",
        sep=",", header=0, index_col=False)
    combine_stochastic_GLOBAL_DF = pd.read_csv(
        f"gradientflow_stochastic_global_{surname}{cfg.architecture}_{cfg.dataset}_sigma_{first_sigma}_pr{cfg.amount}.csv",
        sep=",", header=0, index_col=False)
    for sigma in sigmas:
        lamp_tem_df = pd.read_csv(
            f"gradientflow_stochastic_lamp_{cfg.architecture}_{cfg.dataset}_sigma_{sigma}_pr{cfg.amount}.csv", sep=",",
            header=0, index_col=False)

        global_tem_df = pd.read_csv(
            f"gradientflow_stochastic_global_{cfg.architecture}_{cfg.dataset}_sigma{_sigma}_pr{cfg.amount}.csv",
            sep=",", header=0, index_col=False)

        combine_stochastic_LAMP_DF = pd.concat((combine_stochastic_LAMP_DF, lamp_tem_df), ignore_index=True)
        combine_stochastic_GLOBAL_DF = pd.concat((combine_stochastic_GLOBAL_DF, global_tem_df), ignore_index=True)

    combine_stochastic_LAMP_DF.to_csv(
        f"gradientflow_stochastic_lamp_all_sigmas_{surname}{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv",
        header=True, index=False)
    combine_stochastic_GLOBAL_DF.to_csv(
        f"gradientflow_stochastic_global_all_sigmas_{surname}{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv",
        header=True, index=False)


def unify_all_variables_datasets(sigmas: list, architectures: list, pruning_rates: list, datasets: list):
    combine_stochastic_GLOBAL_DF: pd.DataFrame = None
    combine_stochastic_LAMP_DF: pd.DataFrame = None
    # First of everything
    # first_sigma = sigmas.pop()
    # first_architecture = architectures.pop()
    # first_pruning_rate =pruning_rates.pop()
    # first_dataset =datasets.pop()

    # combine_stochastic_LAMP_DF = pd.read_csv(f"gradientflow_stochastic_lamp_{first_architecture}_{first_dataset}_sigma_{first_sigma}_pr{first_pruning_rate}.csv",sep= ",",header=0,index_col=False)
    # combine_stochastic_GLOBAL_DF = pd.read_csv(f"gradientflow_stochastic_global_{first_architecture}_{first_dataset}_sigma_{first_sigma}_pr{first_pruning_rate}.csv",sep= ",",header=0,index_col=False)
    # Loop over all values of everything
    for pr in pruning_rates:
        for arch in architectures:
            for dataset in datasets:
                for sigma in sigmas:

                    lamp_tem_df = pd.read_csv(f"gradientflow_stochastic_lamp_{arch}_{dataset}_sigma_{sigma}_pr{pr}.csv",
                                              sep=",", header=0, index_col=False)

                    global_tem_df = pd.read_csv(
                        f"gradientflow_stochastic_global_{arch}_{dataset}_sigma_{sigma}_pr{pr}.csv", sep=",", header=0,
                        index_col=False)
                    if combine_stochastic_GLOBAL_DF is None:
                        combine_stochastic_GLOBAL_DF = global_tem_df
                    else:
                        combine_stochastic_GLOBAL_DF = pd.concat((combine_stochastic_GLOBAL_DF, global_tem_df),
                                                                 ignore_index=True)
                    if combine_stochastic_LAMP_DF is None:
                        combine_stochastic_LAMP_DF = lamp_tem_df
                    else:
                        combine_stochastic_LAMP_DF = pd.concat((combine_stochastic_LAMP_DF, lamp_tem_df),
                                                               ignore_index=True)

                deter_lamp_df = pd.read_csv(f"gradientflow_deterministic_lamp_{arch}_{dataset}_pr{pr}.csv", sep=",",
                                            header=0, index_col=False)
                combine_stochastic_LAMP_DF = pd.concat((combine_stochastic_LAMP_DF, deter_lamp_df), ignore_index=True)
                deter_global_df = pd.read_csv(f"gradientflow_deterministic_global_{arch}_{dataset}_pr{pr}.csv", sep=",",
                                              header=0, index_col=False)
                combine_stochastic_GLOBAL_DF = pd.concat((combine_stochastic_GLOBAL_DF, deter_global_df),
                                                         ignore_index=True)

    combine_stochastic_LAMP_DF.to_csv(f"gradientflow_stochastic_lamp_all_sigmas_architectures_datasets_pr.csv",
                                      header=True, index=False)
    combine_stochastic_GLOBAL_DF.to_csv(f"gradientflow_stochastic_global_all_sigmas_architectures_datasets_pr.csv",
                                        header=True, index=False)

    combine_all = pd.concat((combine_stochastic_GLOBAL_DF, combine_stochastic_LAMP_DF), ignore_index=True)
    combine_all.to_csv(f"gradientflow_stochastic_all_sigmas_architectures_datasets_pr.csv", header=True, index=False)


def bar_plot_function(dataFrame1: pd.DataFrame, name="barplot.pdf"):
    """
    This Functions plots
    @param dataFrame1: datafrme with the following column names: Accuracy, Pruning rate, $\sigma$, Pruner, ARchitecture, Dataset, Stage
    @return: None
    """
    g = so.Plot(
        d, y="Accuracy", x="Pruning rate",
        edgestyle=r"$\sigma$", alpha="Pruner", color="Architecture",

    ).add(so.Bar(edgewidth=0.9, edgecolor="k"), so.Agg(), so.Dodge())
    # .add(so.Agg("median"),so.Dodge()).add(so.Est("sd"),so.Doge())
    # g = g.add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    g = g.scale(color="bright")
    g = g.facet("Dataset", "Stage")

    #
    # # g = sns.barplot(d,y="Accuracy",x="Pruning rate",
    # #        hue="sigma", errorbar="sd")
    # # f=(
    # #     so.Plot(
    # #         sns.load_dataset("penguins"), x="species",
    # #         color="sex", alpha="sex", edgestyle="sex",
    # #     )
    # #     .add(so.Bar(edgewidth=2), so.Hist(), so.Dodge("fill"))
    # # )
    # # f.save("barplot_test.pdf",bbox_inches='tight')
    # # g.set_yscale("log")
    # # plt.savefig("barplot.pdf", bbox_inches="tight")
    g.save(name, bbox_inches='tight')


def change_to_upper(string: str):
    return string.upper()


def bar_plot_every_experiment(dataFrame1: pd.DataFrame, dataFrame2: pd.DataFrame = None, use_set="val",
                              sigmas_to_show=[0, 0.001]):
    # Dataframe 1 has every training trace for every individual for every combination of dataset, architecture pruning rate and sigma
    # I just want to have a dataframe that has an extra colum with one-shot and fine-tuned values called stage
    accuracy = []
    stage = []
    id = []
    sigma_list = []
    gradient_flow = []
    pruner_list = []
    arch_list = []
    dataset_list = []
    pr_list = []
    for sigma in dataFrame1["sigma"].unique():
        for pruner in dataFrame1["Pruner"].unique():
            for arch in dataFrame1["Architecture"].unique():
                for dataset in dataFrame1["Dataset"].unique():
                    for pr in dataFrame1["Pruning Rate"].unique():
                        if sigma not in sigmas_to_show:
                            continue
                        first_temp_df = dataFrame1[dataFrame1["sigma"] == sigma]
                        first_temp_df = first_temp_df[first_temp_df["Pruner"] == pruner]
                        first_temp_df = first_temp_df[first_temp_df["Architecture"] == arch]
                        first_temp_df = first_temp_df[first_temp_df["Dataset"] == dataset]
                        first_temp_df = first_temp_df[first_temp_df["Pruning Rate"] == pr]

                        for elem in first_temp_df["individual"].unique():
                            # One-shot data
                            temp_df = first_temp_df[first_temp_df["individual"] == elem]
                            if use_set == "val":
                                gradient_flow.append(
                                    float(temp_df['val_set_gradient_magnitude'][temp_df["Epoch"] == -1].iloc[0]))
                                accuracy.append(float(temp_df["val_accuracy"][temp_df["Epoch"] == -1]))
                                stage.append("One-Shot")
                                id.append(elem)
                                sigma_list.append(sigma)
                                pruner_list.append(pruner)
                                arch_list.append(arch)
                                dataset_list.append(dataset)
                                pr_list.append(pr)
                                # Now the fine-tuned data
                                gradient_flow.append(
                                    float(temp_df.iloc[len(temp_df) - 1]["val_set_gradient_magnitude"]))
                                accuracy.append(
                                    float(temp_df["val_accuracy"][temp_df["Epoch"] == temp_df["Epoch"].max()].iloc[0]))
                                stage.append("Fine-tuned")
                                sigma_list.append(sigma)
                                id.append(elem)
                                sigma_list.append(sigma)
                                pruner_list.append(pruner)
                                arch_list.append(arch)
                                dataset_list.append(dataset)
                                pr_list.append(pr)

                            elif use_set == "test":

                                gradient_flow.append(
                                    float(temp_df['test_set_gradient_magnitude'][temp_df["Epoch"] == -1].iloc[0]))
                                accuracy.append(float(temp_df["test_accuracy"][temp_df["Epoch"] == -1].iloc[0]))
                                stage.append("One-Shot")
                                id.append(elem)
                                sigma_list.append(sigma)
                                pruner_list.append(pruner)
                                arch_list.append(arch.upper())
                                dataset_list.append(dataset.upper())
                                pr_list.append(pr)
                                # Now the fine-tuned data
                                gradient_flow.append(
                                    float(temp_df.iloc[len(temp_df) - 1]["test_set_gradient_magnitude"]))
                                accuracy.append(
                                    float(temp_df["test_accuracy"][temp_df["Epoch"] == temp_df["Epoch"].max()].iloc[0]))
                                stage.append("Fine-tuned")
                                sigma_list.append(sigma)
                                pruner_list.append(pruner)
                                arch_list.append(arch.upper())
                                dataset_list.append(dataset.upper())
                                pr_list.append(pr)

    d = pd.DataFrame(
        {
            "Accuracy": accuracy,
            "Stage": stage,
            r"$\sigma$": sigma_list,
            "Pruner": pruner_list,
            "Architecture": arch_list,
            "Dataset": dataset_list,
            "Pruning rate": pr_list,

        }
    )
    columns = ["Stage", r"$\sigma$", "Pruner", "Architecture", "Dataset", "Pruning rate"]
    test_labels = ["One-Shot", 0, "GMP", "RESNET18", "CIFAR10", 0.9]
    sub_dataframe = d[d["Pruning rate"] == 0.9]
    sub_dataframe = sub_dataframe[sub_dataframe["Dataset"] == "CIFAR10"]
    sub_dataframe = sub_dataframe[sub_dataframe["Architecture"] == "RESNET18"]
    sub_dataframe = sub_dataframe[sub_dataframe["Pruner"] == "GMP"]
    sub_dataframe = sub_dataframe[sub_dataframe["$\sigma$"] == 0.001]
    sub_dataframe = sub_dataframe[sub_dataframe["Stage"] == "One-Shot"]

    new_d = d.groupby(columns).median().reset_index()  # .rename(columns={0:'Median Performance'})
    d.to_csv("organized_all_individuals_gradient_flow_dataset_sigmas{}.csv".format(sigmas_to_show), index=False)
    new_d.to_csv("organized_median_gradient_flow_dataset_sigmas{}.csv".format(sigmas_to_show), index=False)

    if dataFrame2 is not None:
        d2 = dataFrame2[dataFrame2[r"$\sigma$"].isin(sigmas_to_show)]
        d2["Architecture"] = d2["Architecture"].apply(change_to_upper)
        d2["Dataset"] = d2["Dataset"].apply(change_to_upper)
        d2["Accuracy"] = d2["Accuracy"] * 100
        d = pd.concat((d, d2), ignore_index=True)
    #
    # temp_rn05 = d[d["Stage"]=="One-Shot"]
    # temp_rn05 = temp_rn05[temp_rn05[r"$\sigma$"]==0.001]
    # temp_rn05 = temp_rn05[temp_rn05["Architecture"]=="RESNET50"]
    # temp_rn05 = temp_rn05[temp_rn05["Dataset"] =="CIFAR10"]
    # temp_rn05 = temp_rn05[temp_rn05["Pruning rate"]==0.95]
    #
    # thing = temp_rn05.groupby('Pruner').agg(['mean','std','count'])
    #
    # print("STOCHASTIC ONE-SHOT")
    #
    #
    # ci95_lo= []
    # ci95_hi= []
    # for i,_ in enumerate(thing.index):
    #     m,s,c =thing.iloc[i]["Accuracy","mean"],thing.iloc[i]["Accuracy","std"],thing.iloc[i]["Accuracy","count"]
    #     if not c:
    #         ci95_hi.append(0)
    #         ci95_lo.append(0)
    #     else:
    #         ci95_hi.append(m + 1.96 * s / math.sqrt(c))
    #         ci95_lo.append(m - 1.96 * s / math.sqrt(c))
    #
    # thing['ci95_lo'] = ci95_lo
    # thing['ci95_hi'] = ci95_hi
    # print(thing)
    # # Deterministic
    #
    # temp_rn05 = d[d["Stage"]=="One-Shot"]
    # temp_rn05 = temp_rn05[temp_rn05[r"$\sigma$"]==0]
    # temp_rn05 = temp_rn05[temp_rn05["Architecture"]=="RESNET50"]
    # temp_rn05 = temp_rn05[temp_rn05["Dataset"] =="CIFAR10"]
    # temp_rn05 = temp_rn05[temp_rn05["Pruning rate"]==0.95]
    # thing = temp_rn05.groupby('Pruner').agg(['mean','std','count'])
    # print("DETERMINISTIC ONE-SHOT")
    # ci95_lo= []
    # ci95_hi= []
    # for i,_ in enumerate(thing.index):
    #     m,s,c =thing.iloc[i]["Accuracy","mean"],thing.iloc[i]["Accuracy","std"],thing.iloc[i]["Accuracy","count"]
    #     if not c:
    #         ci95_hi.append(0)
    #         ci95_lo.append(0)
    #     else:
    #         ci95_hi.append(m + 1.96 * s / math.sqrt(c))
    #         ci95_lo.append(m - 1.96 * s / math.sqrt(c))
    #
    # thing['ci95_lo'] = ci95_lo
    # thing['ci95_hi'] = ci95_hi
    # print(thing)

    #
    # temp_rn05 = d[d["Stage"]=="Fine-tuned"]
    # temp_rn05 = temp_rn05[temp_rn05[r"$\sigma$"]==0.001]
    # temp_rn05 = temp_rn05[temp_rn05["Architecture"]=="RESNET50"]
    # temp_rn05 = temp_rn05[temp_rn05["Dataset"] =="CIFAR10"]
    # temp_rn05 = temp_rn05[temp_rn05["Pruning rate"]==0.95]
    #
    #
    #
    # thing = temp_rn05.groupby('Pruner').agg(['mean','std','count'])
    #
    # print("STOCHASTIC FINE-TUNED")
    #
    # print(thing)
    #
    # # Deterministic
    #
    # temp_rn05 = d[d["Stage"]=="Fine-tuned"]
    # temp_rn05 = temp_rn05[temp_rn05[r"$\sigma$"]==0]
    # temp_rn05 = temp_rn05[temp_rn05["Architecture"]=="RESNET50"]
    # temp_rn05 = temp_rn05[temp_rn05["Dataset"] =="CIFAR10"]
    # temp_rn05 = temp_rn05[temp_rn05["Pruning rate"]==0.95]
    # thing = temp_rn05.groupby('Pruner').agg(['mean','std','count'])
    # print("DETERMINISTIC FINE-TUNED")
    # print(thing)
    #
    #
    #
    g = so.Plot(
        d, y="Accuracy", x="Pruning rate",
        edgestyle=r"$\sigma$", alpha="Pruner", color="Architecture",

    ).add(so.Bar(edgewidth=0.9, edgecolor="k"), so.Agg(), so.Dodge())
    # .add(so.Agg("median"),so.Dodge()).add(so.Est("sd"),so.Doge())
    # g = g.add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    g = g.scale(color="bright")
    g = g.facet("Dataset", "Stage")

    #
    # # g = sns.barplot(d,y="Accuracy",x="Pruning rate",
    # #        hue="sigma", errorbar="sd")
    # # f=(
    # #     so.Plot(
    # #         sns.load_dataset("penguins"), x="species",
    # #         color="sex", alpha="sex", edgestyle="sex",
    # #     )
    # #     .add(so.Bar(edgewidth=2), so.Hist(), so.Dodge("fill"))
    # # )
    # # f.save("barplot_test.pdf",bbox_inches='tight')
    # # g.set_yscale("log")
    # # plt.savefig("barplot.pdf", bbox_inches="tight")
    g.save("barplot.pdf", bbox_inches='tight')
    g.save("barplot.eps", bbox_inches='tight')


@torch.no_grad()
def get_predictions_of_individual(folder: str, batch: typing.Tuple[torch.Tensor, torch.Tensor], model: nn.Module, cfg):
    model.eval()
    model.cuda()
    data, target = batch
    # print("T")
    # print("")
    batch_prediction = model(data)
    return batch_prediction


# def save_ensemble_predictions():
def calc_ece(confidences, accuracies, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)

    bin_lowers = bin_boundaries[:-1]

    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        # Calculated |confidence - accuracy| in each bin

        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()

            avg_confidence_in_bin = confidences[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item() * 100))

    return ece.item() * 100


def record_predictions(model: nn.Module, data_loader: torch.utils.data.DataLoader, file_path_predictions: str,
                       record_dataset=False, file_path_x: str = "",
                       file_path_y: str = ""):  # now we go through all the test set
    model.cuda()
    all_predictions = None
    all_y = None
    all_x = None
    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_prediction = model(inputs)
        if all_predictions is None:
            all_predictions = batch_prediction.detach().cpu().numpy()
        else:
            all_predictions = np.concatenate((all_predictions, batch_prediction.detach().cpu().numpy()), axis=0)
        if record_dataset:
            if all_x is None:
                all_x = inputs.detach().cpu().numpy()
            else:
                all_x = np.concatenate((all_x, inputs.detach().cpu().numpy()), axis=0)
            if all_y is None:
                all_y = targets.detach().cpu().numpy()
            else:
                all_y = np.concatenate((all_y, targets.detach().cpu().numpy()), axis=0)

    with open(file_path_predictions, "wb") as f:
        pickle.dump(all_predictions, f)
    if record_dataset:
        with open(file_path_x, "wb") as f:
            pickle.dump(all_x, f)
        with open(file_path_y, "wb") as f:
            pickle.dump(all_y, f)


def load_predictions(file):
    t = None
    with open(file, "rb") as f:
        t = pickle.load(file)
    return t


def record_predictions_of_individual(prefix: str, datasets_tuple, cfg):
    stochastic_global_root = prefix + "stochastic_GLOBAL/" + f"{cfg.architecture}/sigma{cfg.sigma}/pr{cfg.amount}/"
    stochastic_lamp_root = prefix + "stochastic_LAMP/" + f"{cfg.architecture}/sigma{cfg.sigma}/pr{cfg.amount}/"
    prediction_prefix = f"/nobackup/sclaam/prediction_storage/{cfg.dataset}/{cfg.architecture}/{cfg.model_type}/sigma{cfg.sigma}/pr{cfg.amount}/"
    # Create the folder structure if is not there already
    path = Path(prediction_prefix)
    path.mkdir(parents=True, exist_ok=True)

    max_individuals = 10

    train, val, test = datasets_tuple

    labels = None
    ########################## first Global stochatic #######################################

    model_place_holder: nn.Module = get_model(cfg)

    index_batch = 0
    ind_number = 0
    model_place_holder.cuda()
    model_place_holder.eval()
    for index, individual in enumerate(glob.glob(stochastic_global_root + "*/", recursive=True)):
        all_predictions = None
        # Load the individuals
        try:
            if Path(individual + "weigths/epoch_90.pth").is_file():
                model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_90.pth"))
                # print("I loaded the weights!")
            if Path(individual + "weigths/epoch_100.pth").is_file():
                model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_100.pth"))
                # print("I loaded the weights!")
            if Path(individual + "weigths/epoch_101.pth").is_file():
                model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_101.pth"))
                # print("I loaded the weights!")
        except Exception as err:
            print("There was the follwing error but im going to continue because I only need 10 individuals")
            print(err)
            continue
        # now we go through all the test set
        for inputs, targets in test:
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_prediction = model_place_holder(inputs)
            if all_predictions is None:
                all_predictions = batch_prediction.detach().cpu().numpy()
            else:
                all_predictions = np.concatenate((all_predictions, batch_prediction.detach().cpu().numpy()), axis=0)

        print(index)
        with open(prediction_prefix + f"global_predictions_{index}.npy", "wb") as f:
            pickle.dump(all_predictions, f)

        if ind_number > max_individuals:
            print("He completado 10 individuos para el batch {}".format(index_batch))
            break
        ind_number += 1
    ############################ Lamp now ############################
    del inputs
    del targets
    torch.cuda.empty_cache()
    # gc.collect()
    print("Now LAMP")
    print(glob.glob(stochastic_lamp_root + "*/", recursive=True))
    index_batch = 0
    ind_number = 0
    for index, individual in enumerate(glob.glob(stochastic_lamp_root + "*/", recursive=True)):
        all_predictions = None
        # Load the individuals
        try:
            if Path(individual + "weigths/epoch_90.pth").is_file():
                model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_90.pth"))
                # print("I loaded the weights!")
            if Path(individual + "weigths/epoch_100.pth").is_file():
                model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_100.pth"))
                # print("I loaded the weights!")
            if Path(individual + "weigths/epoch_101.pth").is_file():
                model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_101.pth"))
                # print("I loaded the weights!")
        except Exception as err:
            print("There was the follwing error but im going to continue because I only need 10 individuals")
            print(err)
            continue

        # now we go through all the test set
        print(index)
        for inputs, targets in test:
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_prediction = model_place_holder(inputs)
            if all_predictions is None:
                all_predictions = batch_prediction.detach().cpu().numpy()
            else:
                all_predictions = np.concatenate((all_predictions, batch_prediction.detach().cpu().numpy()), axis=0)

        with open(prediction_prefix + f"lamp_predictions_{index}", "wb") as f:
            pickle.dump(all_predictions, f)

        if ind_number > max_individuals:
            print("He completado 10 individuos para el batch {}".format(index_batch))
            break
        ind_number += 1
    print("FINISH")
    del model_place_holder


def ensemble_predictions(prefix: str, cfg):
    stochastic_global_root = prefix + "stochastic_GLOBAL/" + f"{cfg.architecture}/sigma{cfg.sigma}/pr{cfg.amount}/"

    stochastic_lamp_root = prefix + "stochastic_LAMP/" + f"{cfg.architecture}/sigma{cfg.sigma}/pr{cfg.amount}/"

    max_individuals = 10

    train, val, test = get_datasets(cfg)

    labels = None

    ########################## first Global stochatic #######################################

    model_place_holder: nn.Module = get_model(cfg)
    full_mean_mean_accuracy = None
    full_voting_mean_accuracy = None
    counter_for_mean_mean = 2
    counter_for_mean_voting = 2
    if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
        accuracy_mean = Accuracy(task="multiclass", num_classes=10).to("cuda")
        accuracy_voting = Accuracy(task="multiclass", num_classes=10).to("cuda")
    if cfg.dataset == "cifar100":
        accuracy_mean = Accuracy(task="multiclass", num_classes=100).to("cuda")
        accuracy_voting = Accuracy(task="multiclass", num_classes=100).to("cuda")
    if cfg.dataset == "imagenet":
        accuracy = Accuracy(task="multiclass", num_classes=1000).to("cuda")
    index_batch = 0
    for inputs, targets in test:
        inputs, targets = inputs.cuda(), targets.cuda()
        predictions_mean = None
        predictions_voting = None
        counter_for_mean_individuals = 2
        ind_number = 0
        for index, individual in enumerate(glob.glob(stochastic_global_root + "*/", recursive=True)):
            # Load the individuals
            # print("Individual:{}".format(individual))
            # print("Contents of the weight folder")
            # p = Path(individual).glob('**/*')
            # files = [x for x in p if x.is_file()]
            # print("{}".format(files))

            # torch.cuda.empty_cache()
            try:
                if Path(individual + "weigths/epoch_90.pth").is_file():
                    model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_90.pth"))
                    # print("I loaded the weights!")
                if Path(individual + "weigths/epoch_100.pth").is_file():
                    model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_100.pth"))
                    # print("I loaded the weights!")
                if Path(individual + "weigths/epoch_101.pth").is_file():
                    model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_101.pth"))
                    # print("I loaded the weights!")
            except Exception as err:
                print("There was the follwing error but im going to continue because I only need 10 individuals")
                print(err)
                continue
                # break
            # Aqui predigo solo un individuo, tengo que acumular estas predicciones para luego hace el recuento total
            # despues del for.
            try:
                individual_predictions: torch.Tensor = get_predictions_of_individual(individual, (inputs, targets),
                                                                                     model_place_holder, cfg)
            except Exception as e:
                print("There was the follwing error but im going to continue because I only need 10 individuals")
                print(e)
                continue

            if predictions_mean is None:
                predictions_mean = individual_predictions
            else:
                predictions_mean = predictions_mean + (
                        individual_predictions - predictions_mean) / counter_for_mean_individuals
                counter_for_mean_individuals += 1
            if predictions_voting is None:
                predictions_voting = torch.reshape(torch.argmax(individual_predictions, dim=1), (-1, 1))
            else:
                predictions_voting = torch.cat(
                    (predictions_voting, torch.reshape(torch.argmax(individual_predictions, dim=1), (-1, 1))), dim=1)

            if ind_number > max_individuals:
                print("He completado 10 individuos para el batch {}".format(index_batch))
                break
            ind_number += 1
        assert predictions_mean is not None, " the predictions for batch {} for all individuals were skipped.".format(
            index)
        index_batch += 1

        # Now I'm going to actually make the predictions first by averaging and second by voting
        temp_variable = torch.mode(predictions_voting, dim=1)
        pred_voting = temp_variable.values
        pred_mean = torch.argmax(predictions_mean, dim=1)

        accuracy_mean.update(preds=pred_mean, target=targets)
        mean_accuracy = accuracy_mean.compute()
        accuracy_voting.update(preds=pred_voting, target=targets)
        voting_accuracy = accuracy_voting.compute()
        # Update the mean of the whole dataset for this particular batch for the two ensemble methods
        # For mean method
        if full_mean_mean_accuracy is None:
            full_mean_mean_accuracy = mean_accuracy
        else:
            full_mean_mean_accuracy = full_mean_mean_accuracy + (
                    mean_accuracy - full_mean_mean_accuracy) / counter_for_mean_mean
            counter_for_mean_mean += 1
        # For voting method
        if full_voting_mean_accuracy is None:
            full_voting_mean_accuracy = voting_accuracy
        else:
            full_voting_mean_accuracy = full_voting_mean_accuracy + (
                    voting_accuracy - full_voting_mean_accuracy) / counter_for_mean_voting
            counter_for_mean_voting += 1

    global_results = {"voting": full_voting_mean_accuracy.detach().cpu().numpy(),
                      "mean": full_mean_mean_accuracy.detach().cpu().numpy()}
    print("Global results")
    print(global_results)
    # torch.cuda.empty_cache()
    with open(stochastic_global_root + "global_ensemble_results", "wb") as f:
        pickle.dump(global_results, f)

    ########################## LAMP stochatic #######################################

    print(cfg)
    print("Now for lamp")
    full_mean_mean_accuracy = None
    full_voting_mean_accuracy = None
    counter_for_mean_mean = 2
    counter_for_mean_voting = 2
    if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
        accuracy = Accuracy(task="multiclass", num_classes=10).to("cuda")
    if cfg.dataset == "cifar100":
        accuracy = Accuracy(task="multiclass", num_classes=100).to("cuda")
    if cfg.dataset == "imagenet":
        accuracy = Accuracy(task="multiclass", num_classes=1000).to("cuda")

    for inputs, targets in test:
        inputs, targets = inputs.cuda(), targets.cuda()
        predictions_mean = None
        predictions_voting = None
        counter_for_mean_individuals = 2
        ind_number = 0
        t0 = time.time()
        for index, individual in enumerate(glob.glob(stochastic_lamp_root + "*/", recursive=True)):
            # Load the individuals
            # print("Individual:{}".format(individual))
            # print("Contents of the weight folder")
            #           p = Path(individual).glob('**/*')
            #           files = [x for x in p if x.is_file()]
            #           print("{}".format(files))
            try:
                if Path(individual + "weigths/epoch_90.pth").is_file():
                    model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_90.pth"))
                    # print("I loaded the weights!")

                elif Path(individual + "weigths/epoch_100.pth").is_file():
                    model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_100.pth"))
                    # print("I loaded the weights!")
                elif Path(individual + "weigths/epoch_101.pth").is_file():
                    model_place_holder.load_state_dict(torch.load(individual + "weigths/epoch_101.pth"))
                    # print("I loaded the weights!")
            except Exception as err:
                print(individual)
                print(err)
                return 0

                # break
            # Aqui predigo solo un individuo, tengo que acumular estas predicciones para luego hace el recuento total
            # despues del for.

            # individual_predictions: torch.Tensor= get_predictions_of_individual(individual,(inputs,targets) , model_place_holder, cfg)

            # Aqui predigo solo un individuo, tengo que acumular estas predicciones para luego hace el recuento total
            # despues del for.
            try:
                individual_predictions: torch.Tensor = get_predictions_of_individual(individual, (inputs, targets),
                                                                                     model_place_holder, cfg)
            except Exception as e:
                print("There was the follwing error but im going to continue because I only need 10 individuals")
                print(e)
                continue

            if predictions_mean is None:
                predictions_mean = individual_predictions
            else:
                predictions_mean = predictions_mean + (
                        individual_predictions - predictions_mean) / counter_for_mean_individuals
                counter_for_mean_individuals += 1
            if predictions_voting is None:
                predictions_voting = torch.reshape(torch.argmax(individual_predictions, dim=1), (-1, 1))
            else:
                predictions_voting = torch.cat(
                    (predictions_voting, torch.reshape(torch.argmax(individual_predictions, dim=1), (-1, 1))), dim=1)

            if ind_number > max_individuals:
                print("I break the individuals for-loop")
                break
            ind_number += 1
        t1 = time.time()
        print("Time to process one batch {}".format(t1 - t0))
        # Now I'm going to actually make the predictions first by averaging and second by voting
        temp_variable = torch.mode(predictions_voting, dim=1)
        pred_voting = temp_variable.values
        pred_mean = torch.argmax(predictions_mean, dim=1)

        accuracy.update(preds=pred_mean, target=targets)
        mean_accuracy = accuracy.compute()
        print("mean accuracy for batch: {} ".format(voting_accuracy))
        accuracy.update(preds=pred_voting, target=targets)
        voting_accuracy = accuracy.compute()
        print("voting accuracy for batch: {} ".format(voting_accuracy))
        # Update the mean of the whole dataset for this particular batch for the two ensemble methods
        # For mean method
        if full_mean_mean_accuracy is None:
            full_mean_mean_accuracy = mean_accuracy
        else:
            full_mean_mean_accuracy = full_mean_mean_accuracy + (
                    mean_accuracy - full_mean_mean_accuracy) / counter_for_mean_mean
            counter_for_mean_mean += 1
        # For voting method
        if full_voting_mean_accuracy is None:
            full_voting_mean_accuracy = voting_accuracy
        else:
            full_voting_mean_accuracy = full_voting_mean_accuracy + (
                    voting_accuracy - full_voting_mean_accuracy) / counter_for_mean_voting
            counter_for_mean_voting += 1

    assert predictions_mean is not None, " the predictions for batch {} for all individuals were skipped.".format(index)
    lamp_results = {"voting": full_voting_mean_accuracy.detach().cpu().numpy(),
                    "mean": full_mean_mean_accuracy.cpu().detach().numpy()}
    print("LAMP")
    print(lamp_results)
    with open(stochastic_lamp_root + "lamp_ensemble_results", "wb") as f:
        pickle.dump(lamp_results, f)

    return global_results, lamp_results


def mock_function():
    dict_1 = {"voting": 94, "mean": 94}
    dict_2 = {"voting": 95, "mean": 95}
    return dict_1, dict_2


def create_ensemble_dataframe(cfg: omegaconf.DictConfig, sigma_values: list, architecture_values: list,
                              pruning_rate_values: list, dataset_values: list):
    import gc
    combine_stochastic_GLOBAL_DF: pd.DataFrame = None
    combine_stochastic_LAMP_DF: pd.DataFrame = None
    accuracy = []
    stage = []
    sigma_list = []
    pruner_list = []
    arch_list = []
    dataset_list = []
    pr_list = []

    # Loop over all values of everything
    for dataset in dataset_values:
        cfg.dataset = dataset
        datasets_tuple = get_datasets(cfg)

        for pruning_rate in pruning_rate_values:
            cfg.amount = pruning_rate
            for arch in architecture_values:
                cfg.architecture = arch
                for sig in sigma_values:
                    cfg.sigma = sig

                    print(cfg)
                    # global_ensemble_results,lamp_ensemble_results = ensemble_predictions(f"/nobackup/sclaam/gradient_flow_data/{cfg.dataset}/",cfg)
                    record_predictions_of_individual(f"/nobackup/sclaam/gradient_flow_data/{cfg.dataset}/",
                                                     datasets_tuple, cfg)
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(torch.cuda.memory_summary(device=None, abbreviated=False))
                    # global_ensemble_results,lamp_ensemble_results = mock_function()
                    # torch.cuda.empty_cache()
                    # For Global
                    # stage.append("Voting")
                    # accuracy.append(global_ensemble_results["voting"])
                    # sigma_list.append(sig)
                    # arch_list.append(arch)
                    # dataset_list.append(dataset)
                    # pr_list.append(pruning_rate)
                    # pruner_list.append("GMP")
                    # ## Mean
                    # stage.append("Mean")
                    # accuracy.append(global_ensemble_results["mean"])
                    # sigma_list.append(sig)
                    # arch_list.append(arch)
                    # dataset_list.append(dataset)
                    # pr_list.append(pruning_rate)
                    # pruner_list.append("GMP")
                    #
                    # # For LAMP
                    # stage.append("Voting")
                    # accuracy.append(lamp_ensemble_results["voting"])
                    # sigma_list.append(sig)
                    # arch_list.append(arch)
                    # dataset_list.append(dataset)
                    # pr_list.append(pruning_rate)
                    # pruner_list.append("LAMP")
                    # ## Mean
                    #
                    # stage.append("Mean")
                    # accuracy.append(lamp_ensemble_results["mean"])
                    # sigma_list.append(sig)
                    # arch_list.append(arch)
                    # dataset_list.append(dataset)
                    # pr_list.append(pruning_rate)
                    # pruner_list.append("LAMP")

    # ensemble_dataframe = pd.DataFrame(
    #     {
    #         "Accuracy":accuracy,
    #         "Stage" :stage,
    #         r"$\sigma$":sigma_list,
    #         "Pruner":pruner_list,
    #         "Architecture":arch_list,
    #         "Dataset" :dataset_list,
    #         "Pruning rate" :pr_list,
    #
    #     }
    # )
    # ensemble_dataframe.to_csv(f"gradientflow_stochastic_ensemble_for_all_datasets_architectures_pruning_rates.csv",header=True ,index=False)


def plot_gradientFlow_data(filepath, title=""):
    data_frame = pd.read_csv(filepath, sep=",", header=0, index_col=False)

    sns.set_theme(style="dark")

    # Plot each year's time series in its own facet
    g = sns.relplot(
        data=data_frame,
        x="Epoch", y="test_set_gradient_magnitude", col="individual", hue="individual",
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=10, height=2, aspect=1.5, legend=False,
    )

    # Iterate over each subplot to customize further
    for year, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

        # Plot every year's time series in the background
        sns.lineplot(
            data=data_frame, x="Epoch", y="val_set_gradient_magnitude", units="individual",
            estimator=None, color=".7", linewidth=1, ax=ax,
        )

    # Reduce the frequency of the x axis ticks
    ax.set_xticks(ax.get_xticks()[::2])

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("", r"$|\nabla\mathcal{L}|$")
    g.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig("le_test.png")


# def same_image_pareto_analysis(dataFrame: pd.DataFrame, file: str):
#     temp_dataframe = dataFrame[["initial_GF_valset", "initial_val_accuracy"]]
#     pareto_indexes_pre = paretoset.paretoset(costs=temp_dataframe, sense=['max', 'max'])
#     temp_dataframe2 = dataFrame[["initial_GF_valset", "final_val_accuracy"]]
#     temp_dataframe3 = dataFrame[["initial_val_accuracy", "final_val_accuracy"]]
#     temp_dataframe4 = dataFrame[["final_GF_valset", "final_val_accuracy"]]
#
#     pareto_indexes_post = paretoset.paretoset(costs=temp_dataframe2, sense=['max', 'max'])
#
#     best = temp_dataframe2['final_val_accuracy'].argmax()
#
#     plt.figure()
#     g = sns.scatterplot(data=temp_dataframe, x='initial_GF_valset', y='initial_val_accuracy', color='blue')
#
#     index_with_both = pareto_indexes_post * pareto_indexes_pre
#     data_frame_both = temp_dataframe[index_with_both]
#     print('Elements on initial accuracy pareto set: {}'.format(sum(pareto_indexes_pre)))
#     print('Elements on final accuracy pareto set: {}'.format(sum(pareto_indexes_post)))
#     print('Elements on both accuracies pareto set: {}'.format(sum(pareto_indexes_post * pareto_indexes_pre)))
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#
#     plt.scatter(x=temp_dataframe[pareto_indexes_post]['initial_GF_valset'],
#                 y=temp_dataframe[pareto_indexes_post]['initial_val_accuracy'], color='blue', edgecolors='red',
#                 # alpha=0.5,
#                 label='Pareto front for final val accuracy')
#     plt.scatter(x=data_frame_both['initial_GF_valset'],
#                 y=data_frame_both['initial_val_accuracy'], color='cyan', edgecolors='red')
#     if pareto_indexes_pre[best]:
#
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#     else:
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#     # After fine-tuning results
#     g = sns.scatterplot(data=temp_dataframe4, x="final_GF_valset", y="final_val_accuracy", color='blue', markers='+')
#     plt.scatter(x=temp_dataframe4[pareto_indexes_pre]['final_GF_valset'],
#                 y=temp_dataframe4[pareto_indexes_pre]['final_val_accuracy'], color='cyan',
#                 label='Pareto front before training'
#                 )
#
#     plt.scatter(x=temp_dataframe['final_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                 y=temp_dataframe['final_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                 color='red',
#                 marker='x',
#                 label='Single Best after fine-tuning'
#                 )
#     not_first_pareto_set = np.bitwise_not(pareto_indexes_pre)
#     new_data_after_finetuning = temp_dataframe4[not_first_pareto_set]
#     new_data = temp_dataframe[not_first_pareto_set]
#     second_pareto_front = paretoset.paretoset(costs=new_data, sense=['max', 'max'])
#     second_pareto_front_finetuning = paretoset.paretoset(costs=new_data_after_finetuning, sense=['max', 'max'])
#
#     # Before Finetunig
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#     scattter = plt.scatter(x=new_data[second_pareto_front]['initial_GF_valset'],
#                            y=new_data[second_pareto_front]['initial_val_accuracy'], color='green',
#                            label='Second Pareto front for initial val accuracy')
#     # After finetuning
#     scattter = plt.scatter(x=temp_dataframe4[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe4[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#     scattter = plt.scatter(x=new_data[second_pareto_front]['initial_GF_valset'],
#                            y=new_data[second_pareto_front]['initial_val_accuracy'], color='green',
#                            label='Second Pareto front for initial val accuracy')
#
#     plt.xlabel("Magnitude gradient")
#     plt.ylabel('Accuracy')
#
#     plt.legend()
#     plt.savefig('{}_pareto_anayisis_GF.png'.format(file), bbox_inches='tight')
#     plt.figure()
#
#     not_first_pareto_set = np.bitwise_not(pareto_indexes_pre)
#     new_data = temp_dataframe[not_first_pareto_set]
#     second_pareto_front = paretoset.paretoset(costs=new_data, sense=['max', 'max'])
#     g = sns.scatterplot(data=temp_dataframe, x='initial_GF_valset', y='initial_val_accuracy', color='blue')
#
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#     scattter = plt.scatter(x=new_data[second_pareto_front]['initial_GF_valset'],
#                            y=new_data[second_pareto_front]['initial_val_accuracy'], color='green',
#                            label='Second Pareto front for initial val accuracy')
#     # best1 = temp_dataframe[best]["initial_GF_valset"]
#     # best2 = temp_dataframe[best]["initial_val_accuracy"]
#     best1 = temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#     best2 = temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()]
#
#     temp = new_data[new_data["initial_GF_valset"] == best1]
#     temp = temp[temp["initial_val_accuracy"] == best2]
#
#     if not temp.empty:
#
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#     else:
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#
#     plt.legend()
#     plt.savefig('{}_2nd_pareto_anayisis_GF.png'.format(file), bbox_inches='tight')
#
#     index_with_both = pareto_indexes_post * pareto_indexes_pre
#     data_frame_both = temp_dataframe[index_with_both]
#     print('Elements on initial accuracy pareto set: {}'.format(sum(pareto_indexes_pre)))
#     print('Elements on final accuracy pareto set: {}'.format(sum(pareto_indexes_post)))
#     print('Elements on both accuracies pareto set: {}'.format(sum(pareto_indexes_post * pareto_indexes_pre)))
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#
#     plt.figure()
#     g = sns.scatterplot(data=temp_dataframe3, x='initial_val_accuracy', y='final_val_accuracy', color='blue')
#     plt.savefig('{}_initial_acc_vs_final_acc.png'.format(file), bbox_inches='tight')
#

# def pareto_analysis(dataFrame: pd.DataFrame, file: str):
#     # Before fine-tuning pareto set
#     temp_dataframe = dataFrame[["initial_GF_valset", "initial_val_accuracy"]]
#     pareto_indexes_pre = paretoset.paretoset(costs=temp_dataframe, sense=['max', 'max'])
#     temp_dataframe2 = dataFrame[["initial_GF_valset", "final_val_accuracy"]]
#     temp_dataframe3 = dataFrame[["initial_val_accuracy", "final_val_accuracy"]]
#     temp_dataframe4 = dataFrame[["final_GF_valset", "final_val_accuracy"]]
#
#     pareto_indexes_post = paretoset.paretoset(costs=temp_dataframe2, sense=['max', 'max'])
#
#     best = temp_dataframe2['final_val_accuracy'].argmax()
#
#     plt.figure()
#     g = sns.scatterplot(data=temp_dataframe, x='initial_GF_valset', y='initial_val_accuracy', color='blue')
#
#     index_with_both = pareto_indexes_post * pareto_indexes_pre
#     data_frame_both = temp_dataframe[index_with_both]
#     print('Elements on initial accuracy pareto set: {}'.format(sum(pareto_indexes_pre)))
#     print('Elements on final accuracy pareto set: {}'.format(sum(pareto_indexes_post)))
#     print('Elements on both accuracies pareto set: {}'.format(sum(pareto_indexes_post * pareto_indexes_pre)))
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#
#     plt.scatter(x=temp_dataframe[pareto_indexes_post]['initial_GF_valset'],
#                 y=temp_dataframe[pareto_indexes_post]['initial_val_accuracy'], color='blue', edgecolors='red',
#                 # alpha=0.5,
#                 label='Pareto front for final val accuracy')
#     plt.scatter(x=data_frame_both['initial_GF_valset'],
#                 y=data_frame_both['initial_val_accuracy'], color='cyan', edgecolors='red')
#     if pareto_indexes_pre[best]:
#
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#     else:
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#
#     plt.legend()
#     plt.savefig('{}_pareto_anayisis_GF.png'.format(file), bbox_inches='tight')
#     plt.figure()
#
#     not_first_pareto_set = np.bitwise_not(pareto_indexes_pre)
#     new_data = temp_dataframe[not_first_pareto_set]
#     second_pareto_front = paretoset.paretoset(costs=new_data, sense=['max', 'max'])
#     g = sns.scatterplot(data=temp_dataframe, x='initial_GF_valset', y='initial_val_accuracy', color='blue')
#
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#     scattter = plt.scatter(x=new_data[second_pareto_front]['initial_GF_valset'],
#                            y=new_data[second_pareto_front]['initial_val_accuracy'], color='green',
#                            label='Second Pareto front for initial val accuracy')
#     # best1 = temp_dataframe[best]["initial_GF_valset"]
#     # best2 = temp_dataframe[best]["initial_val_accuracy"]
#     best1 = temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#     best2 = temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()]
#
#     temp = new_data[new_data["initial_GF_valset"] == best1]
#     temp = temp[temp["initial_val_accuracy"] == best2]
#
#     if not temp.empty:
#
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#     else:
#         plt.scatter(x=temp_dataframe['initial_GF_valset'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     y=temp_dataframe['initial_val_accuracy'].iloc[temp_dataframe2['final_val_accuracy'].argmax()],
#                     color='red',
#                     marker='x',
#                     label='Single Best after fine-tuning'
#                     )
#
#     plt.legend()
#     plt.savefig('{}_2nd_pareto_anayisis_GF.png'.format(file), bbox_inches='tight')
#
#     index_with_both = pareto_indexes_post * pareto_indexes_pre
#     data_frame_both = temp_dataframe[index_with_both]
#     print('Elements on initial accuracy pareto set: {}'.format(sum(pareto_indexes_pre)))
#     print('Elements on final accuracy pareto set: {}'.format(sum(pareto_indexes_post)))
#     print('Elements on both accuracies pareto set: {}'.format(sum(pareto_indexes_post * pareto_indexes_pre)))
#     scattter = plt.scatter(x=temp_dataframe[pareto_indexes_pre]['initial_GF_valset'],
#                            y=temp_dataframe[pareto_indexes_pre]['initial_val_accuracy'], color='cyan',
#                            label='Pareto front for initial val accuracy')
#
#     plt.figure()
#     g = sns.scatterplot(data=temp_dataframe3, x='initial_val_accuracy', y='final_val_accuracy', color='blue')
#     plt.savefig('{}_initial_acc_vs_final_acc.png'.format(file), bbox_inches='tight')
#
#
def get_first_epoch_GF_last_epoch_accuracy(dataFrame, title, file):
    initial_val_set_GF = []
    initial_test_set_GF = []
    final_val_set_GF = []
    final_test_set_GF = []
    final_test_performance = []
    initial_test_performance = []
    final_val_performance = []
    initial_val_performance = []
    average_test_improvement_rate = []
    average_val_improvement_rate = []
    for elem in dataFrame["individual"].unique():
        temp_df = dataFrame[dataFrame["individual"] == elem]
        initial_val_set_GF.append(float(temp_df['val_set_gradient_magnitude'][temp_df["Epoch"] == -1]))
        initial_test_set_GF.append(float(temp_df['test_set_gradient_magnitude'][temp_df["Epoch"] == -1]))
        final_val_set_GF.append(float(temp_df['val_set_gradient_magnitude'][temp_df["Epoch"] == 90]))
        final_test_set_GF.append(float(temp_df['test_set_gradient_magnitude'][temp_df["Epoch"] == 90]))
        final_test_performance.append(float(temp_df["test_accuracy"][temp_df["Epoch"] == 90]))
        initial_test_performance.append(float(temp_df["test_accuracy"][temp_df["Epoch"] == -1]))
        final_val_performance.append(float(temp_df["val_accuracy"][temp_df["Epoch"] == 90]))
        initial_val_performance.append(float(temp_df["val_accuracy"][temp_df["Epoch"] == -1]))
        test_difference = temp_df["test_accuracy"][1:].diff()
        val_difference = temp_df["val_accuracy"][1:].diff()
        average_test_improvement_rate.append(float(test_difference.mean()))
        average_val_improvement_rate.append(float(val_difference.mean()))
    d = pd.DataFrame({"initial_GF_valset": initial_val_set_GF, "initial_GF_testset": initial_test_set_GF,
                      "final_GF_valset": final_val_set_GF,
                      "final_GF_testset": final_test_set_GF,
                      "final_test_accuracy": final_test_performance,
                      "test_improvement_rate": average_test_improvement_rate,
                      "val_improvement_rate": average_val_improvement_rate,
                      "initial_test_accuracy": initial_test_performance,
                      "final_val_accuracy": final_val_performance,
                      "initial_val_accuracy": initial_val_performance,
                      })
    pareto_analysis(d, file)
    pearsons_correlations = d.corr(method="pearson")
    kendal_correlations = d.corr(method="pearson")
    pearsons_correlations.to_csv(f"{file}pearsons_correlations.csv", sep=",")
    kendal_correlations.to_csv(f"{file}kendal_correlations.csv", sep=",")

    plt.figure()

    g = sns.scatterplot(data=d, x="initial_GF_valset", y="final_test_accuracy")
    # g.set_titles("")
    # g.set_axis_labels("", r"$|\nabla\mathcal{L}|$")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_initial_GF_valset_VS_final_acc_testset.png", bbox_inches="tight")
    plt.figure()
    g = sns.scatterplot(data=d, x="initial_GF_testset", y="final_test_accuracy")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_initial_GF_testset_VS_final_acc_testset.png", bbox_inches="tight")

    plt.figure()

    g = sns.scatterplot(data=d, x="initial_GF_valset", y="test_improvement_rate")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_intiial_GF_testset_VS_test_imprvement_rate.png", bbox_inches="tight")

    plt.figure()
    g = sns.scatterplot(data=d, x="initial_GF_valset", y="val_improvement_rate")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_intiial_GF_valset_VS_valset_improvement_rate.png", bbox_inches="tight")

    plt.figure()
    g = sns.scatterplot(data=d, x="initial_GF_valset", y="test_improvement_rate")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_intiial_GF_valset_VS_testset_improvement_rate.png", bbox_inches="tight")

    plt.figure()

    g = sns.scatterplot(data=d, x="initial_GF_valset", y="initial_val_accuracy")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_intial_GF_valset_VS_initial_val_accuracy.png", bbox_inches="tight")

    plt.figure()
    g = sns.scatterplot(data=d, x="initial_val_accuracy", y="final_val_accuracy")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.savefig(f"{file}_initial_val_acc_VS_final_val_accuracy.png", bbox_inches="tight")


def flops_until_thresh(args):
    df = pd.read_csv(args["file"])
    is_det = False
    if "deterministic" in args["file"]:
        is_det = True
    get_statistics_on_FLOPS_until_threshold(df, args["threshold"], is_det)


def get_statistics_on_FLOPS_until_threshold(dataFrame: pd.DataFrame, threshold: float, is_det=False):
    if not is_det:
        all_df: pd.DataFrame = None
        for sigma in dataFrame["sigma"].unique():
            sigma_temp_df = dataFrame[dataFrame["sigma"] == sigma]
            FLOPS_until_threshold = []
            sigma_list = []

            for elem in sigma_temp_df["individual"].unique():
                # One-shot data
                temp_df = sigma_temp_df[sigma_temp_df["individual"] == elem]
                flops = temp_df['sparse_flops'][temp_df['test_accuracy'] >= threshold].min()

                FLOPS_until_threshold.append(flops)
                sigma_list.append(sigma)

            d = pd.DataFrame(
                {"FLOPS_until_threshold": FLOPS_until_threshold,
                 "Sigma": sigma

                 }
            )

            if all_df is None:
                all_df = d
            else:
                all_df = pd.concat((all_df, d), ignore_index=True)

        print('Here are the statistic for sparse flops until threshold {}'.format(threshold))
        new_df = all_df.groupby('Sigma').agg(['mean', 'std', 'count'])

        ci95_hi = []
        ci95_lo = []

        for i in new_df.index:
            m, s, c = new_df.loc[i]
            if not c:
                ci95_hi.append(0)
                ci95_lo.append(0)
            else:
                ci95_hi.append(m + 1.96 * s / math.sqrt(c))
                ci95_lo.append(m - 1.96 * s / math.sqrt(c))

        new_df['ci95_lo'] = ci95_lo
        new_df['ci95_hi'] = ci95_hi
        print(new_df)
    else:

        all_df: pd.DataFrame = None
        FLOPS_until_threshold = []

        for elem in dataFrame["individual"].unique():
            # One-shot data
            temp_df = dataFrame[dataFrame["individual"] == elem]
            flops = temp_df['sparse_flops'][temp_df['test_accuracy'] >= threshold].min()

            FLOPS_until_threshold.append(flops)

        d = pd.DataFrame(
            {"FLOPS_until_threshold": FLOPS_until_threshold,

             }
        )

        if all_df is None:
            all_df = d
        else:
            all_df = pd.concat((all_df, d), ignore_index=True)

        print('Here are the statistic for sparse flops until threshold {}'.format(threshold))
        new_df = all_df.agg(['mean', 'std', 'count'])
        ci95_hi = []
        ci95_lo = []
        m = float(new_df.T["mean"])
        s = float(new_df.T["std"])
        c = float(new_df.T["count"])
        print(new_df)
        ci95_hi = m + 1.96 * s / math.sqrt(c)
        ci95_lo = m - 1.96 * s / math.sqrt(c)
        print("[{:.3E},{:.3E}]".format(ci95_lo, ci95_hi))


# def statistics_on_FLOPS_until_thresclod(args):
#     stochastic_folder = "gradient_flow_data/{}/stochastic_GLOBAL/{}/sigma{}/pr{}/"
#     deterministic_folder = "gradient_flow_data/{}/deterministic_GLOBAL/{}/sigma0.0/pr{}/"
#
#     All_stochastic_df = None
#     for name in glob.glob():


def LeMain(args):
    solution = ""
    exclude_layers = None

    if args["dataset"] == "cifar100":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar100/resnet18_cifar100_traditional_train.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "vgg19":
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar100/resnet50_cifar100.pth"
                exclude_layers = ["conv1", "linear"]
    if args["dataset"] == "cifar10":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth"
                # solution = "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth"
                # solution = "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "vgg19":
                solution = "trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar10/resnet50_cifar10.pth"
                exclude_layers = ["conv1", "linear"]
        if args["modeltype"] == "hub":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth"
                exclude_layers = ["conv1", "fc"]
    if args["dataset"] == "imagenet":

        if args["modeltype"] == "hub":

            if args["architecture"] == "resnet18":
                solution = "/nobackup/sclaam/trained_models/resnet18_imagenet.pth"
                exclude_layers = ["conv1", "fc"]
                # exclude_layers = []
            if args["architecture"] == "VGG19":
                raise NotImplementedError("Not implemented")
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "/nobackup/sclaam/trained_models/resnet50_imagenet.pth"
                exclude_layers = ["conv1", "fc"]

    cfg = omegaconf.DictConfig({
        "population": args["population"],
        "generations": 10,
        "epochs": args["epochs"],
        "short_epochs": 10,
        "architecture": args["architecture"],
        "solution": solution,
        "noise": "gaussian",
        "pruner": args["pruner"],
        "model_type": args["modeltype"],
        "fine_tune_exclude_layers": True,
        "fine_tune_non_zero_weights": True,
        "sampler": "tpe",
        "flop_limit": 0,
        "one_batch": True,
        "measure_gradient_flow": True,
        "full_fine_tune": False,
        "use_stochastic": True,
        "sigma": args["sigma"],
        "noise_after_pruning": 0,
        "amount": args["pruning_rate"],
        "dataset": args["dataset"],
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "gradient_cliping": True,
        "use_wandb": False
    })

    cfg.exclude_layers = exclude_layers

    # weights_analysis_per_weight(cfg)
    # print(omegaconf.OmegaConf.to_yaml(cfg))
    # net = get_model(cfg)
    # train, val, testloader = get_datasets(cfg)
    #
    # dense_performance = test(net, use_cuda=True, testloader=testloader, verbose=0)
    # print("Dense performance on test set = {}".format(dense_performance))
    # pruned_model = copy.deepcopy(net)
    # # cfg.amount = 0.9
    # prune_function(pruned_model, cfg)
    # remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # print("sparsity")
    #
    # # Add small noise just to get tiny variations of the deterministic case
    # det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)
    # print("Deterministic pruning outside function: {}".format(det_performance))
    # stochastic_pruning_against_deterministic_pruning(cfg,name="normal_seed3")
    # print(args)
    # cfg.solution = ""
    # truncated_network_unrestricted_training(cfg)
    # truncated_network_fine_tune_linear_layer_only(cfg)
    # explore_models_shapes()
    # record_features_cifar10_model(cfg.architecture,args["experiment"],cfg.model_type)
    # features_similarity_comparison_experiments(cfg.architecture)

    # experiment_selector(cfg, args, args["experiment"])
    # MDS_projection_plot(cfg)
    # bias_comparison_resnet18()
    # plot_histograms_predictions("normal_seed2")
    # stochastic_pruning_against_deterministic_pruning(cfg,name="normal_seed3")
    # CDF_weights_analysis_stochastic_deterministic(cfg,range=(0,0.05))
    # number_of_0_analysis_stochastic_deterministic(cfg)

    stochastic_soup_of_models(cfg, name="")


def curve_plot(filepath, filename, title: str):
    curve = np.load(filepath)
    curve = dict(curve)
    fig = plt.figure()
    fig, ax = plt.subplots()
    trans_offset1 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=0.05, y=0.10, units='inches')
    trans_offset2 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=-0.05, y=-0.10, units='inches')
    x = curve["ts"]
    y = curve["tr_loss"]
    plt.plot(x, y, label="Bezier Curve")
    plt.xlabel("$t$")
    plt.ylabel("$\mathcal{L}(w)$")
    plt.legend()

    plt.axvline(0, linewidth=1, color='r', linestyle="--")
    plt.axvline(1, linewidth=1, color='r', linestyle="--")
    plt.axhline(min(y), linewidth=1, color='k', linestyle="--")
    ax.annotate('Stochastic', (0.05, 0.9), c="r", xycoords='axes fraction')
    ax.annotate('Deterministic', (0.75, 0.9), c="r", xycoords='axes fraction')
    plt.title(title)
    plt.savefig(f"{filename}_tr_loss.pdf")

    fig = plt.figure()
    fig, ax = plt.subplots()
    trans_offset1 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=0.05, y=0.10, units='inches')
    trans_offset2 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=-0.05, y=-0.10, units='inches')
    x = curve["ts"]
    y = curve["te_loss"]
    plt.plot(x, y, label="Bezier Curve")
    plt.xlabel("$t$")
    plt.ylabel("$\mathcal{L}(w)$")
    plt.legend()

    plt.axvline(0, linewidth=1, color='r', linestyle="--")
    plt.axvline(1, linewidth=1, color='r', linestyle="--")
    plt.axhline(min(y), linewidth=1, color='k', linestyle="--")
    ax.annotate('Stochastic', (0.05, 0.9), c="r", xycoords='axes fraction')
    ax.annotate('Deterministic', (0.75, 0.9), c="r", xycoords='axes fraction')
    plt.title(title)
    plt.savefig(f"{filename}_te_loss.pdf")

    fig = plt.figure()
    fig, ax = plt.subplots()
    trans_offset1 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=0.05, y=0.10, units='inches')
    trans_offset2 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=-0.05, y=-0.10, units='inches')
    x = curve["ts"]
    y = curve["tr_nll"]
    plt.plot(x, y, label="Bezier Curve")
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$\mathcal{L}(w)$")
    plt.axvline(0, linewidth=1, color='r', linestyle="--")
    plt.axvline(1, linewidth=1, color='r', linestyle="--")
    plt.axhline(min(y), linewidth=1, color='k', linestyle="--")
    ax.annotate('Stochastic', (0.05, 0.9), c="r", xycoords='axes fraction')
    ax.annotate('Deterministic', (0.75, 0.9), c="r", xycoords='axes fraction')
    plt.title(title)
    plt.savefig(f"{filename}_tr_nll.pdf")

    fig = plt.figure()
    fig, ax = plt.subplots()
    trans_offset1 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=0.05, y=0.10, units='inches')
    trans_offset2 = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=-0.05, y=-0.10, units='inches')
    x = curve["ts"]
    y = curve["te_nll"]
    plt.plot(x, y, label="Bezier Curve")
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$\mathcal{L}(w)$")
    plt.axvline(0, linewidth=1, color='r', linestyle="--")
    plt.axvline(1, linewidth=1, color='r', linestyle="--")
    plt.axhline(min(y), linewidth=1, color='k', linestyle="--")
    ax.annotate('Stochastic', (0.05, 0.9), c="r", xycoords='axes fraction')
    ax.annotate('Deterministic', (0.75, 0.9), c="r", xycoords='axes fraction')
    # plt.text(0, y[0], ''  , transform=trans_offset1)
    # plt.text(1, y[1], '' , transform=trans_offset2)
    plt.title(title)
    plt.savefig(f"{filename}_te_nll.pdf")


def compare_architecture_distributions(arc1="", arc2=""):
    cfg1 = omegaconf.DictConfig({
        "sigma": 0.0,
        "amount": 0.9528,
        "architecture": "resnet50",
        "model_type": "alternative",
        "solution": "trained_models/cifar10/resnet50_cifar10.pth",
        "dataset": "cifar10",
        "set": "test"

    })

    cfg2 = omegaconf.DictConfig({
        "sigma": 0.0,
        "amount": 0.9,
        "architecture": "resnet18",
        "model_type": "alternative",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
        "dataset": "cifar10",
        "set": "test"

    })
    # cfg2 = omegaconf.DictConfig({
    #     "sigma":0.0,
    #     "amount":0.9,
    #     "architecture":"VGG19",
    #     "model_type": "alternative",
    #     "solution":"trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth",
    #     "dataset":"cifar10",
    #     "set":"test"
    #
    # })

    # weights_analysis_per_weight(cfg1,cfg2)


# TODO: Implement this function
def continued_fined_tuning_imagenet(cfg):
    pass


def record_model_dataset_predictions(cfg, record_dataset=False):
    train, val, test = get_datasets(cfg)
    model = get_model(cfg)
    record_predictions(model, test, "dense_{}_{}_predictions_{}".format(cfg.architecture, cfg.model_type, cfg.dataset),
                       record_dataset, file_path_x="{}_x".format(cfg.dataset), file_path_y="{}_y".format(cfg.dataset))

    det_model = copy.deepcopy(model)


def bias_comparison_resnet18():
    pred_list = []
    pred_labels = []
    with open("dense_resnet18_predictions_cifar10", "rb") as f:
        dense_predictions = pickle.load(f)
        pred_list.append(dense_predictions)
        pred_labels.append("Dense Model")
    with open("fine_tuned_resnet18_sto_lamp_0.005_predictions_cifar10", "rb") as f:
        fined_tuned_stochastic_predictions = pickle.load(f)
        pred_list.append(fined_tuned_stochastic_predictions)
        pred_labels.append(r"$\sigma$=0.005")
    with open("fine_tuned_resnet18_sto_lamp_0.001_predictions_cifar10", "rb") as f:
        fined_tuned_stochastic_predictions_01 = pickle.load(f)
        pred_list.append(fined_tuned_stochastic_predictions_01)
        pred_labels.append(r"$\sigma$=0.001")
    with open("fine_tuned_resnet18_det_lamp_predictions_cifar10", "rb") as f:
        fined_tuned_determinsitic_predictions = pickle.load(f)
        pred_list.append(fined_tuned_determinsitic_predictions)
        pred_labels.append(r"$\sigma$=0")
    data = np.concatenate((fined_tuned_determinsitic_predictions[:100, :].reshape(1, -1),
                           fined_tuned_stochastic_predictions[:100, :].reshape(1, -1),
                           fined_tuned_stochastic_predictions_01[:100, :].reshape(1, -1),
                           dense_predictions[:100, :].reshape(1, -1)), axis=0)
    from sklearn.manifold import TSNE
    # emmbeding = umap.UMAP()
    emmbeding = TSNE(n_components=2, n_jobs=2, perplexity=1)
    new_data = emmbeding.fit_transform(data)
    print(new_data)
    plt.figure()
    df = pd.DataFrame({"x": new_data[:, 0], "y": new_data[:, 1],
                       "Labels": ["Fine-Tuned Det", "Fine-Tuned Sto. 0.005", "Fine-Tuned Sto. 0.001", "Dense"]})
    g = sns.scatterplot(data=df, x="x", y="y", hue="Labels")
    plt.savefig("prediction_projections_lamp_fine_tuned.png")
    #
    # with open("one_shot_resnet18_det_prediction", "rb") as f:
    #     one_shot_determinsitic_predictions = pickle.load(f)
    #
    # with open("one_shot_resnet18_sto_prediction", "rb") as f:
    #     one_shot_stochastic_predictions = pickle.load(f)

    with open("cifar10_y", "rb") as f:
        test_labels = pickle.load(f)
    tensor_test_labels = torch.tensor(test_labels)
    biases = []
    accuracies_list = []
    for pred in pred_list:
        _, _, _, accuracies, confideces = check_correctness(torch.tensor(pred), tensor_test_labels)
        ece = calc_ece(torch.tensor(confideces), torch.tensor(accuracies))
        biases.append(ece)
        accuracies_list.append(((accuracies.sum() / accuracies.size(0)) * 100).item())
    df = pd.DataFrame({"ECE": biases, "Labels": pred_labels, "Accuracy": accuracies_list
                       })
    g = sns.catplot(kind="bar", data=df, x="Labels", y="ECE", hue="Accuracy")
    plt.savefig("bias_lamp.png", bbox_inches="tight")


def MDS_projection_plot(cfg):
    from alternate_models import ResNet18
    from sparse_ensemble_utils import project_models
    train, val, test = get_datasets(cfg)
    dense_model = get_model(cfg)
    record_predictions(dense_model, test, "dense_resnet18_predictions_cifar10")
    model_det = ResNet18()
    model_sto = ResNet18()
    list_of_models = []
    # cfg = omegaconf.DictConfig({
    #     "architecture": "resnet18",
    #     "amount":0.9,
    #     "pruner":"global",
    #     "exclude_layers":["conv1","linear"],
    #
    #
    # })
    # prune_function(model_det,cfg)
    # prune_function(model_sto,cfg)
    # lamp_det = torch.load("noisy_models/cifar10/resnet18/mask_transfer_fine_tuned_global_resnet18_cifar10.pth")
    lamp_det = torch.load("noisy_models/cifar10/resnet18/fine_tuned_det_lamp_pr0.9.pth")
    model_det.load_state_dict(lamp_det)
    record_predictions(model_det, test, "fine_tuned_resnet18_det_lamp_predictions_cifar10", record_dataset=True,
                       file_path_x="cifar10_x", file_path_y="cifar10_y")

    list_of_models.append(copy.deepcopy(model_det))
    global_det = torch.load("noisy_models/cifar10/resnet18/fine_tuned_resnet18_deterministic_global.pth")
    model_det.load_state_dict(global_det)
    record_predictions(model_det, test, "fine_tuned_resnet18_det_predictions_cifar10")

    list_of_models.append(copy.deepcopy(model_det))
    lamp_s1 = torch.load("noisy_models/cifar10/resnet18/fine_tuned_S0.001_lamp_pr0.9.pth", map_location="cpu")
    model_det.load_state_dict(lamp_s1)
    record_predictions(model_det, test, "fine_tuned_resnet18_sto_lamp_0.001_predictions_cifar10")
    list_of_models.append(copy.deepcopy(model_det))

    lamp_s5 = torch.load("noisy_models/cifar10/resnet18/fine_tuned_S0.005_lamp_pr0.9.pth", map_location="cpu")
    model_det.load_state_dict(lamp_s5)
    list_of_models.append(copy.deepcopy(model_det))
    record_predictions(model_det, test, "fine_tuned_resnet18_sto_lamp_0.005_predictions_cifar10")

    global_s1 = torch.load("noisy_models/cifar10/resnet18/fine_tuned_S0.001_pr0.9.pth", map_location="cpu")
    model_det.load_state_dict(global_s1)
    record_predictions(model_det, test, "fine_tuned_resnet18_sto_0.001_predictions_pr0.9_cifar10")
    list_of_models.append(copy.deepcopy(model_det))

    global_s5 = torch.load("noisy_models/cifar10/resnet18/fine_tuned_S0.005_pr_0.9.pth")
    model_det.load_state_dict(global_s5)
    record_predictions(model_det, test, "fine_tuned_resnet18_sto_0.005_predictions_cifar10")

    return
    list_of_models.append(copy.deepcopy(model_det))
    Pruner = ["LAMP", "GMP", "LAMP", "LAMP", "GMP", "GMP"]
    Sigma = [0, 0, 0.001, 0.005, 0.001, 0.005]
    reduced_models = project_models(list_of_models)
    x, y = reduced_models[:, 0], reduced_models[:, 1]
    df = pd.DataFrame({
        "Pruner": Pruner,
        r"$\sigma$": Sigma,
        "Dimension 1": x,
        "Dimension 2": y,
    })
    plt.figure()
    g = sns.scatterplot(data=df, x="Dimension 1", y="Dimension 2", hue=r"$\sigma$", style="Pruner", palette="deep",
                        legend="full")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)
    plt.savefig("MDS_projections_resnet18_cifar10.pdf", bbox_inches="tight")


def test_predictions_for_cfg(cfg, stochastic=False, name=""):
    train, val, testloader = get_datasets(cfg)
    model = get_model(cfg)
    pruned_model = copy.deepcopy(model)
    names, weights = zip(*get_layer_dict(pruned_model))
    sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))
    if stochastic:
        noisy_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        prune_function(noisy_model, cfg)
        remove_reparametrization(noisy_model, exclude_layer_list=cfg.exclude_layers)
        record_predictions(noisy_model, testloader,
                           "one_shot_{}_{}_sto_{}_predictions".format(cfg.architecture, cfg.dataset, name))
    else:
        prune_function(pruned_model, cfg)
        remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
        record_predictions(pruned_model, testloader,
                           "one_shot_{}_{}_det_{}_predictions".format(cfg.architecture, cfg.dataset, name))


def plot_histograms_predictions(name=""):
    with open("fine_tuned_resnet18_sto_0.005_predictions_cifar10", "rb") as f:
        fined_tuned_stochastic_predictions = pickle.load(f)
    with open("fine_tuned_resnet18_det_predictions_cifar10", "rb") as f:
        fined_tuned_determinsitic_predictions = pickle.load(f)

    with open("one_shot_resnet18_cifar10_det_seed2_predictions", "rb") as f:
        one_shot_determinsitic_predictions = pickle.load(f)

    with open("one_shot_resnet18_cifar10_sto_seed2_predictions", "rb") as f:
        one_shot_stochastic_predictions = pickle.load(f)

    with open("cifar10_y", "rb") as f:
        test_labels = pickle.load(f)
    data = np.concatenate((fined_tuned_determinsitic_predictions[:100, :].reshape(1, -1),
                           fined_tuned_stochastic_predictions[:100, :].reshape(1, -1),
                           one_shot_determinsitic_predictions[:100, :].reshape(1, -1),
                           one_shot_stochastic_predictions[:100, :].reshape(1, -1)), axis=0)
    from sklearn.manifold import TSNE
    # emmbeding = umap.UMAP()
    emmbeding = TSNE(n_components=2, n_jobs=2, perplexity=1)
    new_data = emmbeding.fit_transform(data)
    print(new_data)
    plt.figure()
    df = pd.DataFrame({"x": new_data[:, 0], "y": new_data[:, 1],
                       "Labels": ["Fine-Tuned Det", "Fine-Tuned Sto. 0.005", "One-Shot Det", "One-Shot Sto. 0.005"]})
    g = sns.scatterplot(data=df, x="x", y="y", hue="Labels")
    plt.savefig("prediction_projections.png")

    args_sort_FT_det = np.argsort(fined_tuned_determinsitic_predictions, axis=1)
    args_sort_FT_sto = np.argsort(fined_tuned_stochastic_predictions, axis=1)
    args_sort_OS_det = np.argsort(one_shot_determinsitic_predictions, axis=1)
    args_sort_OS_sto = np.argsort(one_shot_stochastic_predictions, axis=1)
    fig, axes = plt.subplots(2, 3, sharey=True)
    fig.suptitle("Resnet18 pr 0.9  CIFAR10")
    # plt.xlabel("Predicted Label")
    axes[0, 0].hist(args_sort_FT_det[:, -1], bins=10)
    axes[0, 0].set_title("Fine tuned Det.")
    axes[0, 0].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    axes[0, 1].hist(args_sort_FT_sto[:, -1], bins=10)
    axes[0, 1].set_title("Fine tuned sto.")
    axes[0, 1].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    axes[0, 2].hist(test_labels, bins=10)
    axes[0, 2].set_title("Labels")
    axes[0, 2].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    axes[1, 0].hist(args_sort_OS_det[:, -1], bins=10)
    axes[1, 0].set_title("One-Shot Det.", fontsize=6)
    axes[1, 0].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    axes[1, 1].hist(args_sort_OS_sto[:, -1], bins=10)
    axes[1, 1].set_title("One-Shot Sto. ", fontsize=6)
    axes[1, 1].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    axes[1, 2].hist(test_labels, bins=10)
    axes[1, 2].set_title("Labels", fontsize=6)
    axes[1, 2].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    plt.savefig("predictions_histograms_{}.png".format(name), bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.hist(args_sort_OS_det[:, -1], bins=10)
    plt.title("One-Shot Det.", fontsize=10)
    plt.savefig("predictions_histogram_one_shot_det_{}.png".format(name), bbox_inches="tight")


def shorcut_function(x, module):
    if module.downsample:
        return module.downsample(x)
    else:
        return x


def create_truncated_resnet18(net):
    def new_fowrard(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out = F.relu(self.layer3[0].bn1(self.layer3[0].conv1(x)))
        out = self.layer3[0].bn2(self.layer3[0].conv2(out))
        out += shorcut_function(x, self.layer3[0])
        out = F.relu(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        return out

    net.__setattr__("fc2", nn.Linear(256, 10))
    net.forward = new_fowrard.__get__(net)  # bind method


def get_features_only_until_layer(net, block=2, net_type=0):
    # ResNet block to compute receptive field for
    if net_type == 0:
        def features_only(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            if block == 0: return x
            # x = self.layer1(x)
            out = F.relu(self.layer1[0].bn1(self.layer1[0].conv1(x)))
            if block == 0.25: return out
            out = self.layer1[0].bn2(self.layer1[0].conv2(out))
            out += shorcut_function(x, self.layer1[0])
            out = F.relu(out)
            x2 = out
            if block == 0.5: return out

            out = F.relu(self.layer1[1].bn1(self.layer1[1].conv1(out)))
            if block == 0.75: return out

            out = self.layer1[1].bn2(self.layer1[1].conv2(out))
            out += shorcut_function(x2, self.layer1[1])
            out = F.relu(out)
            x = out
            if block == 1: return out
            out = F.relu(self.layer2[0].bn1(self.layer2[0].conv1(x)))
            if block == 1.25: return out
            out = self.layer2[0].bn2(self.layer2[0].conv2(out))
            out += shorcut_function(x, self.layer2[0])
            out = F.relu(out)
            x2 = out
            if block == 1.5: return out

            out = F.relu(self.layer2[1].bn1(self.layer2[1].conv1(out)))
            if block == 1.75: return out

            out = self.layer2[1].bn2(self.layer2[1].conv2(out))
            out += shorcut_function(x2, self.layer2[1])
            out = F.relu(out)
            x = out
            if block == 2: return out

            out = F.relu(self.layer3[0].bn1(self.layer3[0].conv1(x)))
            if block == 2.25: return out
            out = self.layer3[0].bn2(self.layer3[0].conv2(out))
            out += shorcut_function(x, self.layer3[0])
            if block == 2.5: return out
            out = F.relu(out)

            x2 = out
            out = F.relu(self.layer3[1].bn1(self.layer3[1].conv1(out)))
            if block == 2.75: return out

            out = self.layer3[1].bn2(self.layer3[1].conv2(out))
            out += shorcut_function(x2, self.layer3[1])
            out = F.relu(out)
            x = out

            if block == 3: return out

            out = F.relu(self.layer4[0].bn1(self.layer4[0].conv1(x)))
            if block == 3.25: return out
            out = self.layer4[0].bn2(self.layer4[0].conv2(out))
            out += shorcut_function(x, self.layer4[0])
            out = F.relu(out)
            x2 = out
            if block == 3.5: return out

            out = F.relu(self.layer4[1].bn1(self.layer4[1].conv1(out)))
            if block == 3.75: return out

            out = self.layer4[1].bn2(self.layer4[1].conv2(out))
            out += shorcut_function(x2, self.layer4[1])
            out = F.relu(out)
            x = out
            # x =  nn.AdaptiveAvgPool2d((1, 1))(x)
            # x = F.avg_pool2d(x, 1)
            return x
    else:
        def features_only(self, x):
            x = self.bn1(self.conv1(x))
            if block == 0: return x

            # x = self.layer1(x)
            out = F.relu(self.layer1[0].bn1(self.layer1[0].conv1(x)))
            if block == 0.25: return out
            out = self.layer1[0].bn2(self.layer1[0].conv2(out))
            out += self.layer1[0].shortcut(x)
            out = F.relu(out)
            x2 = out
            if block == 0.5: return out

            out = F.relu(self.layer1[1].bn1(self.layer1[1].conv1(out)))
            if block == 0.75: return out

            out = self.layer1[1].bn2(self.layer1[1].conv2(out))
            out += self.layer1[1].shortcut(x2)
            out = F.relu(out)
            x = out
            if block == 1: return out
            out = F.relu(self.layer2[0].bn1(self.layer2[0].conv1(x)))
            if block == 1.25: return out
            out = self.layer2[0].bn2(self.layer2[0].conv2(out))
            out += self.layer2[0].shortcut(x)
            out = F.relu(out)
            x2 = out
            if block == 1.5: return out

            out = F.relu(self.layer2[1].bn1(self.layer2[1].conv1(out)))
            if block == 1.75: return out

            out = self.layer2[1].bn2(self.layer2[1].conv2(out))
            out += self.layer2[1].shortcut(x2)
            out = F.relu(out)
            x = out
            if block == 2: return out

            out = F.relu(self.layer3[0].bn1(self.layer3[0].conv1(x)))
            if block == 2.25: return out
            out = self.layer3[0].bn2(self.layer3[0].conv2(out))
            out += self.layer3[0].shortcut(x)
            out = F.relu(out)
            if block == 2.5: return out

            x2 = out
            out = F.relu(self.layer3[1].bn1(self.layer3[1].conv1(out)))
            if block == 2.75: return out

            out = self.layer3[1].bn2(self.layer3[1].conv2(out))
            out += self.layer3[1].shortcut(x2)
            out = F.relu(out)
            x = out

            if block == 3: return out

            out = F.relu(self.layer4[0].bn1(self.layer4[0].conv1(x)))
            if block == 3.25: return out
            out = self.layer4[0].bn2(self.layer4[0].conv2(out))
            out += self.layer4[0].shortcut(x)
            out = F.relu(out)
            x2 = out
            if block == 3.5: return out

            out = F.relu(self.layer4[1].bn1(self.layer4[1].conv1(out)))
            if block == 3.75: return out

            out = self.layer4[1].bn2(self.layer4[1].conv2(out))
            out += self.layer4[1].shortcut(x2)
            out = F.relu(out)
            x = out
            # x = F.avg_pool2d(x, 4)
            return x

    net.forward = features_only.__get__(net)  # bind method


def get_features_only_until_block_layer(net, block=2, net_type=0):
    # ResNet block to compute receptive field for
    if net_type == 0:
        def features_only(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            if block == 0: return x

            x = self.layer1(x)
            if block == 1: return x

            x = self.layer2(x)
            if block == 2: return x

            # x = self.layer3(x)
            out = self.layer3[0].conv1(x)
            out = self.layer3[0].bn1(out)
            out = self.layer3[0].relu(out)

            out = self.layer3[0].conv2(out)
            out = self.layer3[0].bn2(out)
            out = self.layer3[0].relu(out)

            out = self.layer3[0].conv3(out)
            out = self.layer3[0].bn3(out)

            identity = self.layer3[0].downsample(x)

            out += identity
            out = self.layer3[0].relu(out)
            x = out
            out = self.layer3[1].conv1(x)
            # out = self.layer3[1].conv2(out)
            # def forward(self, x: Tensor) -> Tensor:
            #     identity = x
            #
            #     out = self.conv1(x)
            #     out = self.bn1(out)
            #     out = self.relu(out)
            #
            #     out = self.conv2(out)
            #     out = self.bn2(out)
            #     out = self.relu(out)
            #
            #     out = self.conv3(out)
            #     out = self.bn3(out)
            #
            #     if self.downsample is not None:
            #         identity = self.downsample(x)
            #
            #     out += identity
            #     out = self.relu(out)
            #
            #     return out
            if block == 3: return out

            x = self.layer4(x)

            return x
    else:
        def features_only(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            if block == 0: return x
            x = self.layer1(x)
            if block == 1: return x
            x = self.layer2(x)
            if block == 2: return x

            x = self.layer3(x)
            # out = self.layer3[0].conv1(x)
            # out = self.layer3[0].bn1(out)
            # out = F.relu(out)
            #
            # out = self.layer3[0].conv2(out)
            # out = self.layer3[0].bn2(out)
            # out = F.relu(out)
            #
            # out = self.layer3[0].conv3(out)
            # out = self.layer3[0].bn3(out)
            #
            # identity = self.layer3[0].shortcut(x)
            #
            # out += identity
            # out = F.relu(out)
            # out = self.layer3[1].conv1(x)
            # out = self.layer3[1].conv2(out)
            if block == 3: return x

            x = self.layer4(x)
            # x = self.avgpool(x)
            return x

    net.forward = features_only.__get__(net)  # bind method


def get_features_only_until_block_layer_VGG(net, block=2, net_type=0):
    # ResNet block to compute receptive field for
    if net_type == 0:
        def features_only(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            if block == 0: return x

            x = self.layer1(x)
            if block == 1: return x

            x = self.layer2(x)
            if block == 2: return x

            # x = self.layer3(x)
            out = self.layer3[0].conv1(x)
            out = self.layer3[0].bn1(out)
            out = self.layer3[0].relu(out)

            out = self.layer3[0].conv2(out)
            out = self.layer3[0].bn2(out)
            out = self.layer3[0].relu(out)

            out = self.layer3[0].conv3(out)
            out = self.layer3[0].bn3(out)

            identity = self.layer3[0].downsample(x)

            out += identity
            out = self.layer3[0].relu(out)
            x = out
            out = self.layer3[1].conv1(x)
            # out = self.layer3[1].conv2(out)
            # def forward(self, x: Tensor) -> Tensor:
            #     identity = x
            #
            #     out = self.conv1(x)
            #     out = self.bn1(out)
            #     out = self.relu(out)
            #
            #     out = self.conv2(out)
            #     out = self.bn2(out)
            #     out = self.relu(out)
            #
            #     out = self.conv3(out)
            #     out = self.bn3(out)
            #
            #     if self.downsample is not None:
            #         identity = self.downsample(x)
            #
            #     out += identity
            #     out = self.relu(out)
            #
            #     return out
            if block == 3: return out

            x = self.layer4(x)

            return x
    else:
        def features_only(self, x):
            x = self.features(x)
            # print("shape {}".format(x.size()))

            # for i, layer in enumerate(self.features):
            #
            #     x = layer(x)
            # x = F.relu(self.bn1(self.conv1(x)))
            # x=self.maxpool(x)
            # if block == 0: return x
            # x = self.layer1(x)
            # if block == 1: return x
            # x = self.layer2(x)
            # if block == 2: return x
            #
            # x = self.layer3(x)
            # # out = self.layer3[0].conv1(x)
            # # out = self.layer3[0].bn1(out)
            # # out = F.relu(out)
            # #
            # # out = self.layer3[0].conv2(out)
            # # out = self.layer3[0].bn2(out)
            # # out = F.relu(out)
            # #
            # # out = self.layer3[0].conv3(out)
            # # out = self.layer3[0].bn3(out)
            # #
            # # identity = self.layer3[0].shortcut(x)
            # #
            # # out += identity
            # # out = F.relu(out)
            # # out = self.layer3[1].conv1(x)
            # # out = self.layer3[1].conv2(out)
            # if block == 3: return x
            #
            # x = self.layer4(x)
            # x = self.avgpool(x)
            return x

    net.forward = features_only.__get__(net)  # bind method


def number_of_0_analysis_layer_two_models(pruned_model1, pruned_model_2, cfg,
                                          config_list=[], title=""):
    names_m1, weights_m1 = zip(*get_layer_dict(pruned_model1))
    # get noisy model
    names_m2, weights_m2 = zip(*get_layer_dict(pruned_model_2))

    pruning_rate_layer = lambda w: float(torch.count_nonzero(w == 0) / w.nelement())
    # check_for_layers_collapse(pruned_model_2)
    sparsities_m1 = list(map(pruning_rate_layer, weights_m1))
    sparsities_m2 = list(map(pruning_rate_layer, weights_m2))

    name_list = []
    name_list.extend(names_m1)
    name_list.extend(names_m1)

    model_name_list = ["Custom Impl."] * len(sparsities_m1)
    temp_list = ["Pytorch Impl."] * len(sparsities_m2)
    model_name_list.extend(temp_list)

    sparsities_list = []
    sparsities_list.extend(sparsities_m1)
    sparsities_list.extend(sparsities_m2)

    df = pd.DataFrame({'Pruning Rate': sparsities_list, "Implementation": model_name_list, "Layer Name": name_list})
    df.to_csv("pruning_rate_per_layer_comparison_{}_{}.csv".format(cfg.architecture, cfg.dataset), index=False, sep=";")
    # df = df.sort_values(by='count', ascending=False)
    g = sns.barplot(data=df, x="Layer Name", y="Pruning Rate", hue="Implementation")
    plt.xticks(rotation=90)
    plt.savefig('pytorch_seeds_comparisons_pruning_rate_per_layers_pr{}_{}_{}.png'.format(cfg.amount, cfg.architecture,
                                                                                          cfg.dataset),
                bbox_inches="tight")


def truncated_network_load_state_dict(cfg, filename):
    resnet18_truncated = get_model(cfg)
    create_truncated_resnet18(resnet18_truncated)
    state_dict = torch.load(filename)
    resnet18_truncated.load_state_dict(state_dict["net"])
    return resnet18_truncated


def truncated_network_fine_tune_linear_layer_only(cfg):
    from sparse_ensemble_utils import train
    cfg.num_workers = 10
    trainloader, valloader, testloader = get_datasets(cfg)
    # loaded_resnet = truncated_network_load_state_dict(cfg,
    #                                                   "trained_models/cifar10/truncated_resnet18_pytorch_test_acc_88.91.pth")
    # acc_dense = test(loaded_resnet, True, testloader, verbose=1)
    # pruning_rates = [0.5, 0.6, 0.7, 0.8,0.9]
    # for pr in pruning_rates:
    #     cfg.amount = pr
    #     pruned_model = copy.deepcopy(loaded_resnet)
    #     prune_function(pruned_model, cfg)
    #     remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    #     acc_pruned = test(pruned_model, True, testloader, verbose=0)
    #     print("Pruning rate: {} Test set accuracy: {}".format(pr, acc_pruned))
    #
    #
    # names,weights = zip(*get_layer_dict(loaded_resnet))
    # count_params = lambda x: x.numel()
    # numel = list(map(count_params,weights))
    # df = pd.DataFrame({"names":names,"number of params":numel})
    # df.to_csv("test.csv",sep=";",index= False)
    # return
    resnet18_truncated = get_model(cfg)
    name = "resnet18_pytorch"
    create_truncated_resnet18(resnet18_truncated)
    # cfg.exclude_layers = ["conv1", "fc2", "fc", "layer3.1.conv1", "layer3.1.conv2", "layer4.0.conv1", "layer4.0.conv2",
    #                       "layer4.0.downsample.0", "layer4.1.conv1", "layer4.1.conv2"]
    train_layers = ["fc2"]
    for name, param in resnet18_truncated.named_parameters():
        for layer_name in train_layers:
            if layer_name in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18_truncated.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    best_acc = 0
    for epoch in range(cfg.epochs):
        train(epoch, resnet18_truncated, trainloader, optimizer, criterion)
        acc = test(resnet18_truncated, True, testloader, verbose=1)
        scheduler.step()
        print("Test set accuracy: {}".format(acc))
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': resnet18_truncated.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if os.path.isfile('./checkpoint/truncated_{}_test_acc_{}.pth'.format(name, best_acc)):
                os.remove('./checkpoint/truncated_{}_test_acc_{}.pth'.format(name, best_acc))
            torch.save(state, './checkpoint/truncated_{}_test_acc_{}.pth'.format(name, acc))
            best_acc = acc


def truncated_network_unrestricted_training(cfg):
    from sparse_ensemble_utils import train
    cfg.num_workers = 10
    trainloader, valloader, testloader = get_datasets(cfg)
    # loaded_resnet = truncated_network_load_state_dict(cfg,
    #                                                   "trained_models/cifar10/truncated_resnet18_pytorch_test_acc_88.91.pth")
    # cfg.exclude_layers = ["conv1", "fc2", "fc", "layer3.1.conv1", "layer3.1.conv2", "layer4.0.conv1", "layer4.0.conv2",
    #                       "layer4.0.downsample.0", "layer4.1.conv1", "layer4.1.conv2"]
    # acc_dense = test(loaded_resnet, True, testloader, verbose=1)
    # pruning_rates = [0.5, 0.6, 0.7, 0.8,0.9]
    # for pr in pruning_rates:
    #     cfg.amount = pr
    #     pruned_model = copy.deepcopy(loaded_resnet)
    #     prune_function(pruned_model, cfg)
    #     remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    #     acc_pruned = test(pruned_model, True, testloader, verbose=0)
    #     print("Pruning rate: {} Test set accuracy: {}".format(pr, acc_pruned))
    #
    #
    # names,weights = zip(*get_layer_dict(loaded_resnet))
    # count_params = lambda x: x.numel()
    # numel = list(map(count_params,weights))
    # df = pd.DataFrame({"names":names,"number of params":numel})
    # df.to_csv("test.csv",sep=";",index= False)
    # return
    resnet18_truncated = get_model(cfg)
    name = "resnet18_pytorch"
    create_truncated_resnet18(resnet18_truncated)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18_truncated.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    best_acc = 0
    for epoch in range(cfg.epochs):
        train(epoch, resnet18_truncated, trainloader, optimizer, criterion)
        acc = test(resnet18_truncated, True, testloader, verbose=1)
        scheduler.step()
        print("Test set accuracy: {}".format(acc))
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': resnet18_truncated.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if os.path.isfile('./checkpoint/truncated_{}_test_acc_{}.pth'.format(name, best_acc)):
                os.remove('./checkpoint/truncated_{}_test_acc_{}.pth'.format(name, best_acc))
            torch.save(state, './checkpoint/truncated_{}_test_acc_{}.pth'.format(name, acc))
            best_acc = acc


def representation_similarity_analysis(prefix1, prefix2, number_layers, name1="", name2="", use_device="cuda"):
    from CKA_similarity.CKA import CudaCKA, CKA

    if use_device == "cuda":
        kernel = CudaCKA("cuda")
        similarity_matrix = torch.zeros((number_layers, number_layers), device=use_device)

    if use_device == "cpu":
        similarity_matrix = np.zeros((number_layers, number_layers))
        kernel = CKA()
    #### because the similiarity is a simetrical
    for i in range(number_layers):
        if use_device == "cuda":
            layer_i = torch.tensor(load_layer_features(prefix1, i, name=name1)[:1000, :])
        if use_device == "cpu":
            layer_i = load_layer_features(prefix1, i, name=name1)
        for j in range(i, number_layers):
            if use_device == "cuda":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                layer_j = torch.tensor(load_layer_features(prefix2, j, name=name2)[:1000, :])
                t1 = time.time()
                print("Time of loading both layers: {}".format(t1 - t0))
                layeri_cuda = layer_i.cuda()
                layerj_cuda = layer_j.cuda()
                layeri_cuda = layeri_cuda - torch.mean(layeri_cuda, dtype=torch.float, dim=0)
                layerj_cuda = layerj_cuda - torch.mean(layerj_cuda, dtype=torch.float, dim=0)

                t0 = time.time()
                similarity_matrix[i, j] = kernel.linear_CKA(layeri_cuda.float(), layerj_cuda.float())
                t1 = time.time()
                t0 = time.time()
                t0 = time.time()
                print("Time for linear kernel: {}".format(t1 - t0))
                del layeri_cuda
                del layerj_cuda
                torch.cuda.empty_cache()

            if use_device == "cpu":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                # layer_i = load_layer_features(prefix1, i, name=name1)[:100,:]
                layer_j = load_layer_features(prefix2, j, name=name2)[:100, :]

                layeri_cuda = layer_i - np.mean(layer_i, dtype=np.float, axis=0)
                layerj_cuda = layer_j - np.mean(layer_j, dtype=np.float, axis=0)

                similarity_matrix[i, j] = kernel.linear_CKA(layeri_cuda, layerj_cuda)
                t1 = time.time()
                print("Time of loading + linear kernel: {}".format(t1 - t0))
                del layeri_cuda
                del layeri_cuda
                del layerj_cuda

    # network1 =
    if use_device == "cuda":
        simetric_similarity = similarity_matrix.add(similarity_matrix.T)
        simetric_similarity[range(number_layers), range(number_layers)] *= 1 / 2
        return simetric_similarity.detach().cpu().numpy()
    if use_device == "cpu":
        simetric_similarity = similarity_matrix + np.transpose(similarity_matrix)
        simetric_similarity[range(number_layers), range(number_layers)] *= 1 / 2
        return simetric_similarity


##### 1 of septemeber 2023 #####################################

def features_similarity_comparison_experiments(architecture="resnet18"):
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
         # "solution": "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth",
         # explore_models_shapes()
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })

    prefix_custom_train = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_custom_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    cfg.model_type = "hub"
    prefix_pytorch_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))

    ##### -1 beacuse I dont have the linear layer here
    number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1

    #########   Pytorch vs Pytorch architectures ##################################
    # similarity_for_networks = representation_similarity_analysis(prefix_pytorch_test,prefix_pytorch_test,number_layers=number_of_layers,name1="_seed_1",name2="_seed_2")
    # filename = "similarity_experiments/{}_pytorch_V_pytorch_similarity.txt".format(cfg.architecture)
    # # with open(filename,"wb") as f :
    # np.savetxt(filename,similarity_for_networks,delimiter=",")

    ######### Custom vs Custom architectures ##################################
    #

    similarity_for_networks = representation_similarity_analysis(prefix_custom_test, prefix_custom_test,
                                                                 number_layers=number_of_layers, name1="_seed_1",
                                                                 name2="_seed_2", use_device="cuda")
    filename = "similarity_experiments/{}_custom_V_custom_similarity_cuda_1000.txt".format(cfg.architecture)
    # with open(filename,"wb") as f :
    np.savetxt(filename, similarity_for_networks, delimiter=",")
    #
    # #########   Pytorch vs Custom architectures ##################################
    #
    similarity_for_networks = representation_similarity_analysis(prefix_pytorch_test, prefix_custom_test,
                                                                 number_layers=number_of_layers, name1="_seed_1",
                                                                 name2="_seed_1", use_device="cuda")
    filename = "similarity_experiments/{}_pytorch_V_custom_similarity_cuda_100010531gg.txt".format(cfg.architecture)
    # with open(filename,"wb") as f :
    np.savetxt(filename, similarity_for_networks, delimiter=",")


def record_features_cifar10_model(architecture="resnet18", seed=1,
                                  modeltype="resnet50_normal_seed_2_tst_acc_95.65.pthalternative", solution="",
                                  seed_name="_seed_1"):
    from feature_maps_utils import save_layer_feature_maps_for_batch
    if seed == 1:
        seed_name = "_seed_1"
        if architecture == "resnet18":
            solution_normal = "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth"
            solution_pytorch = "trained_models/cifar10/resnet18_official_cifar10_seed_1_test_acc_88.5.pth"
        if architecture == "resnet50":
            solution_normal = "trained_models/cifar10/resnet50_cifar10.pth"
            solution_pytorch = "trained_models/cifar10/resnet50_official_cifar10_seed_1_test_acc_90.31.pth"
    if seed == 2:
        seed_name = "_seed_2"
        if architecture == "resnet18":
            solution_normal = "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth"
            solution_pytorch = "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth"
        if architecture == "resnet50":
            solution_normal = "trained_models/cifar10/resnet50_normal_seed_2_tst_acc_95.65.pth"
            # "/home/home01/sclaam/sparse_ensemble/trained_models/cifar10/resnet50_normal_seed_2_tst_acc_95.65.pth"
            solution_pytorch = "trained_models/cifar10/resnet50_official_cifar10_seed_2_test_acc_89.93.pth"
    if modeltype == "alternative":
        solution = solution_normal
    if modeltype == "hub":
        solution = solution_pytorch

    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype,
         # "model_type": "hub",
         "solution": solution,
         # "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
         # "solution": "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth",
         # explore_models_shapes()
         "dataset": "cifar10",
         "batch_size": 1,
         "num_workers": 4,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    train, val, testloader = get_datasets(cfg)
    ################################# dataset cifar10 ###########################################################################
    if cfg.dataset == "cifar10":
        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

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

    resnet18_normal = get_model(cfg)
    current_directory = Path().cwd()
    add_nobackup = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        add_nobackup = "/nobackup/sclaam/"

    prefix_custom_train = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_custom_test = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    #
    # cfg.model_type = "hub"
    #
    # cfg.solution = solution_pytorch
    # # cfg.solution = "trained_models/cifar10/resnet18_official_cifar10_seed_1_test_acc_88.5.pth"
    # # cfg.solution = "trained_models/cifar10/resnet50_official_cifar10_seed_1_test_acc_90.31.pth"
    # cfg.exclude_layers = ["conv1", "fc"]
    # cfg.pruner = "global"
    # # save_onnx(cfg)
    # resnet18_pytorch = get_model(cfg)
    #
    # prefix_pytorch_train = Path("{}features/{}/{}/{}/{}/".format(add_nobackup,cfg.dataset,cfg.architecture,cfg.model_type,"train"))
    # prefix_pytorch_test = Path("{}features/{}/{}/{}/{}/".format(add_nobackup,cfg.dataset,cfg.architecture,cfg.model_type,"test"))
    # prefix_pytorch_test.mkdir(parents=True,exist_ok=True)
    # prefix_pytorch_train.mkdir(parents=True,exist_ok=True)

    ###################### Get the features for the training set for both models#####################################
    # for x,y in trainloader:
    #     # First the custom implementation
    #     save_layer_feature_maps_for_batch(resnet18_normal,x,prefix_custom_train,name="_seed_1")
    #
    #     # second the custom implementation
    #     save_layer_feature_maps_for_batch(resnet18_pytorch,x,prefix_custom_train,name="_seed_1")
    # layer_features = load_layer_features(prefix_custom_test,index=0,name="_seed_1")
    # return
    maximun_samples = 2000
    resnet18_normal.cuda()
    o = 0
    for x, y in testloader:
        # First the custom implementation
        # y_hat = resnet18_normal(x)
        # print(y_hat)
        x = x.cuda()
        # return
        save_layer_feature_maps_for_batch(resnet18_normal, x, prefix_custom_test, seed_name=seed_name)
        # second the custom implementation
        # save_layer_feature_maps_for_batch(resnet18_pytorch,x,prefix_pytorch_test,seed_name=seed_name)

        print("{} batch out of {}".format(o, len(testloader)))
        if o == maximun_samples:
            break
        o += 1
    # print("before reading the layer")
    # layer_features = load_layer_features(prefix_custom_test,index=0,name="_seed_1")
    # print("Lenght of layer 0 features {}".format(len(layer_features)))
    # layer_features = load_layer_features(prefix_pytorch_test,index=0,name="_seed_1")
    # print("Lenght of layer 0 features {}".format(len(layer_features)))
    # return


#####################################
def explore_models_shapes():
    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
         # "solution": "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth",
         # explore_models_shapes()
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    # test_predictions_for_cfg(cfg,False,"seed3")
    # test_predictions_for_cfg(cfg,True,"seed3")
    # return
    train, val, testloader = get_datasets(cfg)
    resnet18_normal = get_model(cfg)
    my_impl_pruned = copy.deepcopy(resnet18_normal)
    prune_function(my_impl_pruned, cfg)
    remove_reparametrization(my_impl_pruned, exclude_layer_list=cfg.exclude_layers)
    cfg.model_type = "hub"

    # cfg.solution = "trained_models/cifar10/resnet18_official_cifar10_seed_1_test_acc_88.5.pth"
    cfg.solution = "trained_models/cifar10/resnet50_official_cifar10_seed_1_test_acc_90.31.pth"
    cfg.exclude_layers = ["conv1", "fc"]
    cfg.pruner = "global"
    # save_onnx(cfg)
    resnet18_pytorch = get_model(cfg)
    pytorch_impl_pruned = copy.deepcopy(resnet18_pytorch)
    cfg.amount = 0.99
    prune_function(pytorch_impl_pruned, cfg)
    remove_reparametrization(pytorch_impl_pruned, exclude_layer_list=cfg.exclude_layers)
    # create_truncated_resnet18(pytorch_impl_pruned)
    # det_accuracy = test(pytorch_impl_pruned, True, testloader, verbose=1)
    # number_of_0_analysis_layer_two_models(my_impl_pruned, pytorch_impl_pruned, cfg,
    #                                       title="Comparing implementations ResNet50")
    resnet18_pytorch.eval()
    resnet18_normal.eval()

    data, y = next(iter(testloader))
    images = data[:1, :, :, :]
    #######################################3

    from feature_maps_utils import get_activations_shape

    # receptive_field_dict = receptive_field(resnet18_pytorch, (3, 32, 32),device="cpu")
    # receptive_field_for_unit(receptive_field_dict, "2", (1, 1))
    from shrinkbench.metrics.flops import get_activations
    t0 = time.time()
    activations_normal = get_activations(resnet18_normal, images)
    t1 = time.time()
    activations_pytorch = get_activations(resnet18_pytorch, images)
    t2 = time.time()
    print("Time for resnet18 normal = {}".format(t1 - t0))
    print("Time for resnet18 pytorch = {}".format(t2 - t1))
    return
    input_shapes_normal, resnet18_normal_activations_shapes, module_names_normal = get_activations_shape(
        resnet18_normal, images)
    input_shapes_pytorch, resnet18_pytorch_activations_shapes, module_names_pytorch = get_activations_shape(
        resnet18_pytorch, images)

    print("{} normal".format(cfg.architecture))
    print(module_names_normal)
    print(resnet18_normal_activations_shapes)

    p1 = pd.DataFrame({
        "Module Name": module_names_normal,
        "Input Shape": input_shapes_normal,
        "Output Shape": resnet18_normal_activations_shapes
    })

    p1.to_csv("output_shapes_{}_{}_normal.csv".format(cfg.architecture, cfg.dataset), sep=";", index=False)

    shapes_fn = lambda w: w.size()

    names, weights = zip(*get_layer_dict(resnet18_normal))

    w_shapes_list = list(map(shapes_fn, weights))

    wp1 = pd.DataFrame({
        "Module Name": names,
        "Weight Shape": w_shapes_list
    })
    wp1.to_csv("weight_shapes_{}_{}_normal.csv".format(cfg.architecture, cfg.dataset), sep=";", index=False)

    print("{} pytorch".format(cfg.architecture))
    print(module_names_pytorch)
    print(resnet18_pytorch_activations_shapes)

    w_shapes_list = list(map(shapes_fn, weights))

    p2 = pd.DataFrame({
        "Module Name": module_names_pytorch,
        "Input Shape": input_shapes_pytorch,
        "Output Shape": resnet18_pytorch_activations_shapes
    })

    p2.to_csv("output_shapes_{}_{}_pytorch.csv".format(cfg.architecture, cfg.dataset), sep=";", index=False)

    shapes_fn = lambda w: w.size()

    names, weights = zip(*get_layer_dict(resnet18_pytorch))
    w_shapes_list = list(map(shapes_fn, weights))
    wp2 = pd.DataFrame({
        "Module Name": names,
        "Weight Shape": w_shapes_list
    })
    wp2.to_csv("weight_shapes_{}_{}_pytorch.csv".format(cfg.architecture, cfg.dataset), sep=";", index=False)

    #########################################
    from torch_receptive_field import receptive_field, receptive_field_for_unit
    from easy_receptive_fields_pytorch.receptivefield import receptivefield
    import pdb

    # resnet18_normal.cpu()
    # resnet18_normal.train()
    # receptive_field_dict = receptivefield(resnet18_normal, (3, 32, 32),device="cpu")
    # get_features_only_until_layer(resnet18_pytorch,block=1,net_type=0)
    # rf = receptivefield(resnet18_pytorch, (1, 3, 224, 224))
    # print(rf)
    print("Receptive field normal resnet18")

    # resnet18_normal.train()
    # get_features_only_until_layer(resnet18_normal, block=0.75, net_type=1)
    # rf = receptivefield(resnet18_normal, (1, 3, 224, 224))
    # print(rf)
    # blocks = [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    # blocks = np.linspace(0,4,17)
    blocks = [0, 1, 2, 3, 4]
    receptive_fields = []

    for i in blocks:
        get_features_only_until_block_layer(resnet18_normal, block=i, net_type=1)
        rf = receptivefield(resnet18_normal, (1, 3, 500, 500))
        # pdb.set_trace()
        print("Receptive field for block {}".format(i))
        print(rf)
        receptive_fields.append(tuple(rf.rfsize))
        # rf.show()
    rf1 = pd.DataFrame({
        "Block index": blocks,
        "Receptive field": receptive_fields
    })
    rf1.to_csv("receptive_fields_{}_{}_normal.csv".format(cfg.architecture, cfg.dataset), sep=";", index=False)
    #
    # # receptive_field_for_unit(receptive_field_dict, "2", (1, 1))
    print()
    print("Receptive field {} pytorch ".format(cfg.architecture))
    # # resnet18_pytorch.cpu()
    # # resnet18_pytorch.train()
    receptive_fields = []
    for i in blocks:
        get_features_only_until_block_layer(resnet18_pytorch, block=i, net_type=0)
        rf = receptivefield(resnet18_pytorch, (1, 3, 500, 500))
        print("Receptive field for block {}".format(i))
        print(rf)
        receptive_fields.append(tuple(rf.rfsize))
    #     # rf.show()
    rf1 = pd.DataFrame({
        "Block index": blocks,
        "Receptive field": receptive_fields
    })
    rf1.to_csv("receptive_fields_{}_{}_pytorch.csv".format(cfg.architecture, cfg.dataset), sep=";", index=False)


def stochastic_soup_of_models(cfg: omegaconf.DictConfig, eval_set: str = "test", name: str = "", version="dense"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    number_of_samples = cfg.epochs
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []

    t0 = time.time()
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)
    t1 = time.time()
    print("Time for test: {}".format(t1 - t0))
    pruned_original = copy.deepcopy(net)
    names, weights = zip(*get_layer_dict(net))

    number_of_layers = len(names)

    sigma_per_layer = dict(zip(names, [cfg.sigma] * number_of_layers))

    if cfg.pruner == "global":

        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")

    else:

        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    remove_reparametrization(pruned_original, exclude_layer_list=cfg.exclude_layers)

    print("pruned_performance of pruned original")

    t0 = time.time()

    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)

    print("Det. performance in function: {}".format(pruned_original_performance))

    t1 = time.time()

    print("Time for test: {}".format(t1 - t0))

    soup1_list = []
    soup2_list = []
    dense_soup1_list = []
    new_pruning_rates_list = []
    determinsitc_model_performance_list = [pruned_original_performance] * number_of_samples
    original_with_new_pruning_rates_list = []
    dense_soup_then_prune_delta = []
    dense_original_list =[original_performance]*number_of_samples
    del pruned_original

    labels = []

    ##########  first soup and then prune ############

    for i in range(number_of_samples):

        number_of_models = 0
        sum_vector = None

        for n in range(N):

            current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
            if sum_vector is None:

                sum_vector = parameters_to_vector(current_model.parameters())

                number_of_models += 1

            else:

                sum_vector += parameters_to_vector(current_model.parameters())

                number_of_models += 1

        soup1_model = copy.deepcopy(net)

        vector_to_parameters(sum_vector / number_of_models, soup1_model.parameters())

        soup_dense_performance = test(soup1_model, use_cuda, evaluation_set, verbose=0)

        dense_soup1_list.append(soup_dense_performance)

        if cfg.pruner == "global":

            prune_with_rate(soup1_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")

        else:

            prune_with_rate(soup1_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)

        # # Here is where I transfer the mask from the pruned stochastic model to the
        # # original weights and put it in the ranking

        # # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(soup1_model, exclude_layer_list=cfg.exclude_layers)
        soup_performance = test(soup1_model, use_cuda, evaluation_set, verbose=0)

        soup1_list.append(soup_performance)


    ##########  first prune and then soup ############

    for i in range(number_of_samples):
        pruned_vector = None
        number_of_models = 0
        for n in range(N):

            current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)

            if cfg.pruner == "global":

                prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")

            else:

                prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                                pruner=cfg.pruner)

            # # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)

            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

            if pruned_vector is None:

                pruned_vector = parameters_to_vector(current_model.parameters())

                number_of_models += 1

            else:

                pruned_vector += parameters_to_vector(current_model.parameters())

                number_of_models += 1

        soup2_model = copy.deepcopy(net)

        vector_to_parameters(pruned_vector/number_of_models, soup2_model.parameters())

        soup2_performance = test(soup2_model, use_cuda, evaluation_set, verbose=0)
        soup2_list.append(soup2_performance)

        total = count_parameters(soup2_model)

        zero = count_zero_parameters(soup2_model)

        new_pruning_rate = float(zero / total)

        new_pruning_rates_list.append(new_pruning_rate)

        deterministic_with_new_pruning_rate = copy.deepcopy(net)

        if cfg.pruner == "global":

            prune_with_rate(deterministic_with_new_pruning_rate, new_pruning_rate, exclude_layers=cfg.exclude_layers,
                            type="global")

        else:

            prune_with_rate(deterministic_with_new_pruning_rate, new_pruning_rate, exclude_layers=cfg.exclude_layers,
                            type="layer-wise",
                            pruner=cfg.pruner)

        # # Here is where I transfer the mask from the pruned stochastic model to the
        # # original weights and put it in the ranking

        # # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)

        remove_reparametrization(deterministic_with_new_pruning_rate, exclude_layer_list=cfg.exclude_layers)

        deterministic_performance_with_new_pruning_rate = test(deterministic_with_new_pruning_rate, use_cuda,
                                                               evaluation_set, verbose=0)

        original_with_new_pruning_rates_list.append(deterministic_performance_with_new_pruning_rate)

    df = pd.DataFrame({"Original Pruning Rate": [cfg.amount] * number_of_samples,
                       "Deterministic performance": determinsitc_model_performance_list,
                       "Soup then Prune accuracy": soup1_list,
                       "Prune then Soup accuracy": soup2_list,
                       "Original with new pruning rate": original_with_new_pruning_rates_list,
                       "New pruning rate": new_pruning_rates_list,
                       "Dense soup then prune":dense_soup1_list,
                       "Dense original":dense_original_list
                       })

    df.to_csv(
        "soup_ideas_results/soup_{}_{}_{}_{}_sigma_{}_results.csv".format(cfg.architecture, cfg.dataset, cfg.amount,
                                                                          name,
                                                                          cfg.sigma),
        index=False)

    #
    # # torch.cuda.empty_cache()
    #
    # # print("Stocastic pruning performance")
    #
    # # stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
    #
    # # print("Time for test: {}".format(t1 - t0))
    #
    #
    # # pruned_performance.append(stochastic_pruned_performance)
    # # stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
    # # del current_model
    # # torch.cuda.empty_cache()
    #
    # # len(pruned performance)-1 because the first one is the pruned original
    #
    # labels.extend(["stochastic pruned"] * (len(pruned_performance)))
    #
    # # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # # index 0 of pruned_performance.
    #
    # assert len(labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
    #                                                f"{len(labels)}!={len(pruned_performance)}"
    #
    # ranked_index = np.flip(np.argsort(pruned_performance))
    # index_of_pruned_original = list(ranked_index).index(0)
    # all_index = np.ones(len(ranked_index), dtype=bool)
    # all_index[index_of_pruned_original] = False
    # ranked_index = ranked_index[all_index]
    # pruned_performance = np.array(pruned_performance)
    # stochastic_dense_performances = np.array(stochastic_dense_performances)
    # result = time.localtime(time.time())
    #
    # del pop
    #
    # cutoff = original_performance - 2

    ################################# Plotting The Comparison #########################################################
    #
    # fig, ax = plt.subplots(figsize=fig_size,layout="compressed")
    #
    # original_line = ax.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")
    #
    # deterministic_pruning_line = ax.axhline(y=pruned_original_performance, c="purple", label="Deterministic Pruning")
    # plt.xlabel("Ranking Index", fontsize=fs)
    # plt.ylabel("Accuracy", fontsize=fs)
    # stochastic_models_points_dense = []
    # stochastic_models_points_pruned = []
    # transfer_mask_models_points = []
    # stochastic_with_deterministic_mask_models_points = []
    # for i, element in enumerate(pruned_performance[ranked_index]):
    #     if labels[ranked_index[i]] == "sto mask transfer":
    #         sto_transfer_point = ax.scatter(i, element, c="tab:orange", marker="P")
    #         transfer_mask_models_points.append(sto_transfer_point)
    #     elif labels[ranked_index[i]] == "det mask transfer":
    #         det_transfer_point = ax.scatter(i, element, c="tab:olive", marker="X")
    #         stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
    #     else:
    #         pruned_point = ax.scatter(i, element, c="steelblue", marker="x")
    #         stochastic_models_points_pruned.append(pruned_point)
    # for i, element in enumerate(stochastic_dense_performances[ranked_index]):
    #     if i == index_of_pruned_original or element == 1:
    #         continue
    #         # ax.scatter(i, element, c="y", marker="o", label="original model performance")
    #     else:
    #         dense_point = ax.scatter(i, element, c="c", marker="1")
    #         stochastic_models_points_dense.append(dense_point)
    #
    # plt.legend([original_line, tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense),
    #             deterministic_pruning_line],
    #            ['Original Performance', 'Pruned Stochastic', 'Dense Stochastic', "Deterministic Pruning"],
    #            scatterpoints=1,
    #            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    # plt.grid(ls='--', alpha=0.5)
    # plt.savefig(
    #     f"data/figures/_{cfg.dataset}_{cfg.pruner}_{cfg.architecture}_stochastic_deterministic_{cfg.noise}_sigma_"
    #     f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
    #     f"_{cfg.population}_{eval_set}_{name}.pdf")
    # plt.savefig(
    #     f"data/figures/_{cfg.dataset}_{cfg.pruner}_{cfg.architecture}_stochastic_deterministic_{cfg.noise}_sigma_"
    #     f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
    #     f"_{cfg.population}_{eval_set}_{name}.pgf")


if __name__ == '__main__':
    # MDS_projection_plot()

    # cfg_training = omegaconf.DictConfig({
    #     "architecture": "resnet18",
    #     "batch_size": 512,
    #     "lr": 0.001,
    #     "momentum": 0.9,
    #     "weight_decay": 1e-4,\
    #     "cyclic_lr": True,
    #     "lr_peak_epoch": 5,
    #     "optim": "adam",
    #     "solution": "trained_models/traditional_trained_val_accuracy=91,86.pt",
    #     "num_workers": 1,
    #     "cosine_schedule": False,
    #     "epochs": 24
    # })
    # run_traditional_training(cfg_training)

    # stochastic_pruning_against_deterministic_pruning(cfg)
    # stochastic_pruning_global_against_LAMP_deterministic_pruning(cfg)

    # save_onnx(cfg)

    ############################## Epsilon experiments for the boxplots ################################################

    # identifier = f"{time.time():14.5f}".replace(" ", "")
    # population_sweeps_transfer_mask_rank_experiments(cfg,identifier=identifier)

    ########### All epsilon stochastic pruning #######################
    # fp = "data/epsilon_experiments_t_1-33_full.csv" # -> The name of this must be the result of  the previews function and be consistent with the cfg.
    #
    # fp = "epsilon_experiments_lamp_full.csv"
    # fp = "data/epsilon_experiments_cifar100_resnet18_global_1680113970.02633_full.csv"
    # fp = "data/epsilon_experiments_mnist_resnet18_global_1680114120.50534_full.csv"
    # fp = "data/epsilon_experiments_cifar100_VGG19_global_1680266419.94637_full.csv"
    # fp = "data/epsilon_experiments_cifar10_VGG19_global_1680561544.77210_full.csv"
    # [0.72, 0.88, 0.94]
    # cfg = omegaconf.DictConfig({
    #     "architecture": "resnet18",
    #     "dataset": "cifar10",
    #     "exclude_layers": ["conv1", "linear"],
    #     "model_type":"alternative",
    #     "pruner": "global",
    #     "amount": 0.9,
    #     "batch_size": 512,
    #     "lr": 0.001,
    #     "momentum": 0.9,
    #     "weight_decay": 1e-4,\
    #     "cyclic_lr": True,
    #     "lr_peak_epoch": 5,
    #     "optim": "adam",
    #     "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
    #     "num_workers": 1,
    #     "cosine_schedule": False,
    #     "epochs": 24
    # })
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.003],
    #                                           specific_pruning_rates=[0.9])
    #
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.003],
    #                                           specific_pruning_rates=[0.8])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.003],
    #                                           specific_pruning_rates=[0.5])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.001],
    #                                           specific_pruning_rates=[0.9])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.001],
    #                                           specific_pruning_rates=[0.8])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.001],
    #                                           specific_pruning_rates=[0.5])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.005],
    #                                           specific_pruning_rates=[0.9])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.005],
    #                                           specific_pruning_rates=[0.8])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.005],
    #                                           specific_pruning_rates=[0.5])
    #
    # # ##############################################################################
    # #  Stochastic/deterministic prunig with meausrement of gradient flow (use task array runs for concurrent runs)
    # # ##############################################################################

    #
    parser = argparse.ArgumentParser(description='Stochastic pruning experiments')
    parser.add_argument('-exp', '--experiment', type=int, default=15, help='Experiment number', required=True)
    parser.add_argument('-pop', '--population', type=int, default=1, help='Population', required=False)
    parser.add_argument('-gen', '--generation', type=int, default=10, help='Generations', required=False)
    # parser.add_argument('-mod', '--model_type',type=str,default=alternative, help = 'Type of model to use', required=False)
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Epochs for fine tuning', required=False)
    parser.add_argument('-sig', '--sigma', type=float, default=0.005, help='Noise amplitude', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch size', required=True)
    parser.add_argument('-pr', '--pruner', type=str, default="global", help='Type of prune', required=True)
    parser.add_argument('-dt', '--dataset', type=str, default="cifar10", help='Dataset for experiments', required=True)
    parser.add_argument('-ar', '--architecture', type=str, default="resnet18", help='Type of architecture',
                        required=True)
    # parser.add_argument('-so', '--solution',type=str,default="", help='Path to the pretrained solution, it must be consistent with all the other parameters', required=True)
    parser.add_argument('-mt', '--modeltype', type=str, default="alternative",
                        help='The type of model (which model definition/declaration) to use in the architecture',
                        required=True)
    parser.add_argument('-pru', '--pruning_rate', type=float, default=0.9, help='percentage of weights to prune',
                        required=False)
    ############# this is for pr and sigma optim ###############################
    parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers', required=False)
    parser.add_argument('-ob', '--one_batch', type=bool, default=False, help='One batch in sigma pr optim',
                        required=False)

    parser.add_argument('-sa', '--sampler', type=str, default="tpe", help='Sampler for pr sigma optim', required=False)
    parser.add_argument('-ls', '--log_sigma', type=bool, default=False,
                        help='Use log scale for sigma in pr,sigma optim', required=False)
    parser.add_argument('-tr', '--trials', type=int, default=300, help='Number of trials for sigma,pr optim',
                        required=False)
    parser.add_argument('-fnc', '--functions', type=int, default=1,
                        help='Type of functions for MOO optim of sigma and pr', required=False)

    args = vars(parser.parse_args())

    LeMain(args)

    #
    # MDS_projection_plot()

    #
    #     ###  Stochastic pruning with pruner, dataset, sigma, and pruning rate present on cfg
    #     #  Also if measure GF is True in cfg then it measures the gradientflow, flops and other measurements  for the particlar
    #
    # experiment_selector(cfg,number_experiment=6)
    #
    #     ################# Unify sigma experiments results for gradient flow measurement ####################################
    #

    #
    # sigma_values = [0.001,0.0021,0.0032,0.0043,0.005,0.0065,0.0076,0.0087,0.0098,0.011]
    # sigma_values = [0.001,0.003,0.005]
    # pruning_rate_values = [0.8,0.85,0.9,0.95]
    # architecture_values = ["VGG19","resnet50","resnet18"]
    # dataset_values = ["cifar10","cifar100"]
    #
    #
    # sigma_values = [0.005]
    # dataset_values = ["cifar10"]
    # pruning_rate_values = [0.9]
    # architecture_values = ["resnet18"]
    # cfg = omegaconf.DictConfig({
    #     "sigma":0.005,
    #     "amount":0.9,
    #     "architecture":"resnet18",
    #     "model_type": "alternative",
    #     "dataset": "cifar10",
    #     "set":"test"
    # })
    #
    #
    # for dataset in dataset_values:
    #     cfg.dataset = dataset
    #     for pruning_rate in pruning_rate_values:
    #         cfg.amount = pruning_rate
    #         for arch in architecture_values:
    #             cfg.architecture = arch
    #             for sig in sigma_values:
    # gradient_flow_especific_combination_dataframe_generation_stochastic_only("gradient_flow_data/mask_transfer_det_sto/cifar10/",cfg,2,surname="mask_transfer_")
    # unify_sigma_datasets(sigmas=sigma_values,cfg=cfg)

    ################################################# Ensemble predictions ############################################
    # sigma_values = [0.001,0.003,0.005]
    # pruning_rate_values = [0.8,0.85,0.9,0.95]
    # architecture_values = ["resnet18","resnet50","VGG19"]
    # dataset_values = ["cifar10","cifar100"]
    #
    # cfg = omegaconf.DictConfig({
    #     "sigma":0.001,
    #     "amount":0.9,
    #     "architecture":"resnet18",
    #     "model_type": "alternative",
    #     "dataset": "imagenet",
    #     "set":"test",
    #     "solution":"",
    #     "batch_size": 512,
    #     # "batch_size": 128,
    #     "num_workers": 0,
    # })
    # create_ensemble_dataframe(cfg,sigma_values, architecture_values, pruning_rate_values, dataset_values)
    #

    #     accuracy = []
    #     stage = []
    #     id = []
    #     sigma_list = []
    #     gradient_flow= []
    #     pruner_list = []
    #     arch_list = []
    #     dataset_list = []
    #     pr_list = []
    #

    #     # unify_all_variables_datasets(sigmas=sigma_values,architectures=architecture_values,pruning_rates=pruning_rate_values,datasets=dataset_values)
    #     #
    #
    #
    # #
    # #     ########################## Scatter plots for the accuracy vs GF ##################################################
    # #
    # #     # gradient_flow_correlation_analysis("gradient_flow_data/",cfg)
    # #     #
    # #
#     cfg = omegaconf.DictConfig({
#         "sigma":0.001,
#         "amount":0.9,
#         "architecture":"resnet18",
#         "model_type": "alternative",
#         "dataset": "cifar10",
#         "set":"test",
#         "solution":"",
#         "batch_size": 512,
#         # "batch_size": 128,
#         "num_workers": 0,
#     })
#     # df = pd.read_csv(f"gradientflow_stochastic_lamp_mask_transfer_resnet18_cifar10_sigma_0.005_pr0.9.csv",sep = ",",header = 0, index_col = False)
#     # df2 = pd.read_csv(f"gradientflow_stochastic_global_mask_transfer_resnet18_cifar10_sigma_0.005_pr0.9.csv",sep = ",",header = 0, index_col = False)
#     df = pd.read_csv(f"gradientflow_stochastic_lamp_all_sigmas_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv",sep = ",",header = 0, index_col = False)
#     df2 = pd.read_csv(f"gradientflow_stochastic_global_all_sigmas_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv",sep = ",",header = 0, index_col = False)
#     deterministic_lamp_df = pd.read_csv(f"gradientflow_deterministic_lamp_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv",sep = ",",header = 0, index_col = False)
#     deterministic_glbal_df = pd.read_csv(f"gradientflow_deterministic_global_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv",sep = ",",header = 0, index_col = False)
# # # #
# #     sigmas = [0.005]
#     sigmas = [0.001,0.0021,0.005,0.0065,0.0076,0.011]
#
#     directory = "gradient_flow_results_test_set/"
# # # #
#     scatter_plot_sigmas(df,None, deterministic_dataframe1=deterministic_lamp_df,
#                         deterministic_dataframe2=deterministic_glbal_df, det_label1='Deter. LAMP',
#                         det_label2='Deter. GMP', file=f"{directory}lamp_scatter_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}_{cfg.set}.pdf", use_set=cfg.set,sigmas_to_show=sigmas)
#     scatter_plot_sigmas(df2,None, deterministic_dataframe1=deterministic_lamp_df,
#                         deterministic_dataframe2=deterministic_glbal_df, det_label1='Deter. LAMP',
#                         det_label2='Deter. GMP', file=f"{directory}global_scatter_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}_{cfg.set}.pdf",use_set=cfg.set, sigmas_to_show=sigmas)
#
#
#     # ################################## Barplot with all results #######################################################################
#     #
#     # df = pd.read_csv(f"gradientflow_stochastic_all_sigmas_architectures_datasets_pr.csv", sep=",", header=0,
#     #                  index_col=False)
#     # df2 = pd.read_csv(f"gradientflow_stochastic_ensemble_for_all_datasets_architectures_pruning_rates.csv", sep=",",
#     #                   header=0, index_col=False)
#     # bar_plot_every_experiment(df, df2, use_set="test", sigmas_to_show=[0, 0.001, 0.003, 0.005])
#
#     # ################################## CURVE PLOTS #######################################################################
#     #
#     # curve_plot("dnn_mode_connectivity/evaluate_curve/cifar100/resnet18/global/fine_tuned/curve.npz","deter_vs_sto_GLOBAL_resnet18_Sig_0.001_cifar100_fine_tuned","CIFAR100")
#     #
#     # ########### Both functions down grab a csv file with names that reflect the pruning rate, dataset, architecture, pruner and  type of
#     #     # pruning done. They plot Gradient flow VS accuracy in the validation set of the given dataset
#     #
#     #
#     #     ##########################  Last table  Flops count for LAMP  ####################################################
#     #     print(f"FLOPS results for {cfg.architecture} on {cfg.dataset} with pruning rate {cfg.amount}")
#     #     print("Lamp stochastic")
#     #     # fp = "gradientflow_stochastic_lamp_all_sigmas_pr0.9.csv"
#     #     fp = f"gradientflow_stochastic_lamp_all_sigmas_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv"
#     #     df = pd.read_csv(fp,sep = ",",header = 0, index_col = False)
#     #
#         # get_statistics_on_FLOPS_until_threshold(df,92)
#     #     # fp = "gradientflow_deterministic_lamp_pr0.9.csv"
#     #     fp = f"gradientflow_deterministic_lamp_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv"
#     #     df = pd.read_csv(fp,sep = ",",header = 0, index_col = False)
#     #
#     #     print("Now lamp deterministic")
#     #
#     # get_statistics_on_FLOPS_until_threshold(df,92,is_det=True)
#     #
#     #     #########################  Last table  Flops count for GMP #######################################################
#     #
#     #     print("Global stochastic")
#     #     # fp = "gradientflow_stochastic_global_all_sigmas_pr0.9.csv"
#     #     fp = f"gradientflow_stochastic_global_all_sigmas_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv"
#     #     df = pd.read_csv(fp ,sep = ",",header = 0, index_col = False)
#     #
#     #     get_statistics_on_FLOPS_until_threshold(df,92)
#     #     # fp = "gradientflow_deterministic_lamp_pr0.9.csv"
#     #     fp = f"gradientflow_deterministic_global_{cfg.architecture}_{cfg.dataset}_pr{cfg.amount}.csv"
#     #     df = pd.read_csv(fp,sep = ",",header = 0, index_col = False)
#     #
#     #     print("Now global deterministic")
#     #     get_statistics_on_FLOPS_until_threshold(df,92,is_det=True)
#     #
