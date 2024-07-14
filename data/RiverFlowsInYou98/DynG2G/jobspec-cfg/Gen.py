import argparse
import yaml
import os
import time
import torch
import torch.nn
import numpy as np
import copy
import pickle

from utils import *
from models import *
from data import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "-f",
    "--config_file",
    default="configs/config-1.yaml",
    help="Configuration file to load.",
)
ARGS = parser.parse_args(args=[])

with open(ARGS.config_file, "r") as f:
    config = yaml.safe_load(f)
print("Loaded configuration file", ARGS.config_file)
print(config)

K = config["K"]
n_hidden = config["n_hidden"]
tolerance = config["tolerance"]
L_list = config["L_list"]
save_MRR_MAP = config["save_MRR_MAP"]
save_sigma_mu = config["save_sigma_mu"]

p_val = config["p_val"]
p_test = config["p_test"]
p_train = 1 - p_val - p_test

scale = False
verbose = True

data_name = "AS"
name = "Results/" + data_name + "_" + os.environ["SLURM_ARRAY_JOB_ID"] + "/"
logger = init_logging_handler(name)
logger.debug(str(config))

device = check_if_gpu()
logger.debug("The code will be running on {}.".format(device))

print("Dataset: " + str(data_name))
if data_name == "SBM":
    data = Dataset_SBM("datasets/sbm_50t_1000n_adj.csv")
    L = 64
    num_epochs = 700
    patience_init = 10
elif data_name == "UCI":
    data = Dataset_UCI(
        (
            "datasets/download.tsv.opsahl-ucsocial.tar.bz2",
            "opsahl-ucsocial/out.opsahl-ucsocial",
        )
    )
    L = 256
    num_epochs = 100
    patience_init = 3
elif data_name == "AS":
    # data = Dataset_AS("datasets/as_data")
    with open("datasets/as_data/ASdata_class", "rb") as f:
        data = pickle.load(f)
    L = 64
    num_epochs = 700
    patience_init = 10
elif data_name == "BitCoin":
    data = Dataset_BitCoin("datasets/soc-sign-bitcoinotc.csv")
    L = 256
    num_epochs = 700
    patience_init = 10
elif data_name == "RM":
    data = Dataset_RM(("datasets/download.tsv.mit.tar.bz2", "mit/out.mit"))
    L = 64
    num_epochs = 700
    patience_init = 10

# data.A_list = [data.A_list[0]] * len(data)
# data.X_list = [data.X_list[0]] * len(data)


learning_rate = 1e-3
train_time_list = []
mu_list = []
sigma_list = []
theta = 0.25
resetting_counts = 0


for t in range(len(data)):
    logger.debug("==============================================")
    logger.debug("timestamp {}".format(t))
    A, X = data[t]
    N, D = X.shape

    num_ones = min(1000, A.nnz)
    num_zeros = min(1000, A.nnz)
    val_ones = sample_ones(A, num_ones)
    val_zeros = sample_zeros(A, num_zeros)
    val_edges = np.row_stack((val_ones, val_zeros))
    val_label = A[val_edges[:, 0], val_edges[:, 1]].A1

    logger.debug("# vertices: %d; # nonzeros: %d" % (N, A.nnz))
    logger.debug(
        "# edges for validation: positive: %d; negative: %d" % (num_ones, num_zeros)
    )
    hops = get_hops(A, K)
    scale_terms = {}
    for h in hops:
        if h != -1:
            scale_terms[h] = hops[h].sum(1).A1
        else:
            scale_terms[max(hops.keys()) + 1] = hops[1].shape[0] - hops[h].sum(1).A1

    start = time.time()

    G2G = Graph2Gauss(n_hidden, L, D)
    # if t == 0:
    #     G2G = Graph2Gauss(n_hidden, L, D)
    # elif t == 1:
    #     G2G = copy.deepcopy(G2G)
    #     G2G_prev = copy.deepcopy(G2G)
    #     if G2G.layers[0].weight.data.shape[0] < D:
    #         dummy_input = InputLinear(G2G.layers[0].weight.data.shape[0])
    #         dummy_output, G2G.layers[0] = wider(dummy_input, G2G.layers[0], D)
    # else:
    #     G2G = copy.deepcopy(G2G)
    #     G2G_prev2 = copy.deepcopy(G2G_prev)
    #     G2G_prev = copy.deepcopy(G2G)

    #     add_net1 = copy.deepcopy(G2G_prev)
    #     add_net2 = copy.deepcopy(G2G_prev2)
    #     if G2G.layers[0].weight.data.shape[0] < D:
    #         dummy_input = InputLinear(G2G.layers[0].weight.data.shape[0])
    #         dummy_output, G2G.layers[0] = wider(dummy_input, G2G.layers[0], D)
    #     if add_net1.layers[0].weight.data.shape[0] < D:
    #         dummy_input = InputLinear(add_net1.layers[0].weight.data.shape[0])
    #         dummy_output, add_net1.layers[0] = wider(dummy_input, add_net1.layers[0], D)
    #     if add_net2.layers[0].weight.data.shape[0] < D:
    #         dummy_input = InputLinear(add_net2.layers[0].weight.data.shape[0])
    #         dummy_output, add_net2.layers[0] = wider(dummy_input, add_net2.layers[0], D)
    #     for param1, param2, param3 in zip(
    #         G2G.parameters(), add_net1.parameters(), add_net2.parameters()
    #     ):
    #         param1.data = theta * param2.data + (1 - theta) * param3.data

    G2G = G2G.to(device)
    optimizer = torch.optim.Adam(G2G.parameters(), lr=learning_rate)
    patience = patience_init
    best_score = 0
    best_epoch = 0
    if not os.path.exists(name + "/models"):
        os.makedirs(name + "/models")

    logger.debug("Training")
    for epoch in range(0, num_epochs + 1):
        if verbose and (epoch == 1 or epoch % 10 == 0):
            logger.debug("----------------------------------------------")
            logger.debug("time stamp %d, L: %d, epoch: %3d" % (t, L, epoch))
        G2G.train()
        optimizer.zero_grad()
        X = X.to(device)
        _, mu, sigma = G2G(X)
        triplets, triplet_scale_terms = to_triplets(sample_all_hops(hops), scale_terms)
        loss_s = build_loss(triplets, triplet_scale_terms, mu, sigma, L, scale=scale)
        if loss_s > 1e4:
            logger.debug("Loss overflow, resetting G2G model")
            resetting_counts = resetting_counts + 1
            G2G.reset_parameters()
            _, mu, sigma = G2G(X)
            triplets, triplet_scale_terms = to_triplets(
                sample_all_hops(hops), scale_terms
            )
            loss_s = build_loss(
                triplets, triplet_scale_terms, mu, sigma, L, scale=scale
            )
        if verbose and (epoch == 1 or epoch % 10 == 0):
            G2G.eval()
            patience -= 1
            neg_val_energy = -Energy_KL(mu, sigma, val_edges, L).cpu().detach().numpy()
            val_auc, val_ap = score_link_prediction(val_label, neg_val_energy)
            if val_auc + val_ap > best_score:
                model_name = (
                    name + "models/time" + str(t) + "epoch" + str(epoch) + ".pt"
                )
                torch.save(G2G.state_dict(), model_name)
                best_score = val_auc + val_ap
                best_epoch = epoch
                patience = patience_init
            logger.debug(
                "loss: %.4f, val_auc: %.4f, val_ap: %.4f"
                % (loss_s.item(), val_auc, val_ap)
            )
            # logger.debug(
            #     "patience: %d, best_epoch: %d, best_score: %.4f"
            #     % (patience, best_epoch, best_score)
            # )
            if patience == 0 or abs(val_auc + val_ap - 2.0) < 1e-4:
                logger.debug("L: {}, epoch: {:3d}: Early Stopping".format(L, epoch))
                logger.debug("----------------------------------------------")
                break
        loss_s.backward()
        optimizer.step()
    end = time.time()
    logger.debug("Training G2G at time stamp %d costs %.2fs." % (t, end - start))
    train_time_list.append(end - start)

    logger.debug("loading the best model from epoch %d..." % (best_epoch))
    G2G.load_state_dict(
        torch.load(name + "models/time" + str(t) + "epoch" + str(best_epoch) + ".pt")
    )
    G2G.eval()
    _, mu, sigma = G2G(X)
    neg_val_energy = -Energy_KL(mu, sigma, val_edges, L).cpu().detach().numpy()
    val_auc, val_ap = score_link_prediction(val_label, neg_val_energy)
    logger.debug("val_auc {:.4f}, val_ap: {:.4f}".format(val_auc, val_ap))
    for root, dirs, files in os.walk(name + "models"):
        for file in files:
            os.remove(os.path.join(name + "models", file))
    mu_list.append(mu.cpu().detach().numpy())
    sigma_list.append(sigma.cpu().detach().numpy())
    logger.debug("----------------------------------------------")

print("Training finished!")
print(
    "G2G model resets %d times in %d time stamps during training."
    % (resetting_counts, len(data))
)
if save_sigma_mu == True:
    if not os.path.exists(name + "/saved_embed"):
        os.makedirs(name + "/saved_embed")
    with open(name + "/saved_embed/mu" + str(L), "wb") as f:
        pickle.dump(mu_list, f)
    with open(name + "/saved_embed/sigma" + str(L), "wb") as f:
        pickle.dump(sigma_list, f)
