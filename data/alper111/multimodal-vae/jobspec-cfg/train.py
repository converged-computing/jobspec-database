"""Train multimodal VAE."""
import os
import time
import argparse

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

import models
import data
import utils


parser = argparse.ArgumentParser("Train multimodal VAE.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
dev = opts["device"]
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

logdir = os.path.join(opts["save"], "log")
writer = SummaryWriter(logdir)

idx = torch.randperm(opts["traj_count"])[:opts["traj_count"]].tolist()

trainset = data.MyDataset(opts["data"], modality=opts["modality"], action=opts["action"], mode="train", traj_list=idx)
valset = data.MyDataset(opts["data"], modality=opts["modality"], action=opts["action"], mode="val")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=10000, shuffle=False)
val_sample = iter(valloader).next()
x_val = trainset.normalize(val_sample)
x_val = [x.to(dev) for x in x_val]

model = models.MultiVAE(
    in_blocks=opts["in_blocks"],
    in_shared=opts["in_shared"],
    out_shared=opts["out_shared"],
    out_blocks=opts["out_blocks"],
    init=opts["init_method"])
model.to(dev)
optimizer = torch.optim.Adam(lr=opts["lr"], params=model.parameters(), amsgrad=True)
print(model)
print("Parameter count:", utils.get_parameter_count(model))
best_error = 1e5
BETA = opts["beta"]
ban_list = [0 for _ in opts["modality"]]

for e in range(opts["epoch"]):
    running_avg = 0.0
    for i, x_t in enumerate(trainloader):
        optimizer.zero_grad()
        x_t = [x.to(dev) for x in x_t]
        x_t = trainset.normalize(x_t)

        x_noised = utils.noise_input(x_t, banned_modality=ban_list, prob=[0.5, 0.5, 0.5],
                                     direction="both", modality_noise=True)

        loss = model.loss(x_noised, x_t, lambd=opts["lambda"], beta=BETA, reduce=opts["reduce"], mse=opts["mse"])
        loss.backward()
        optimizer.step()
        running_avg += loss.item()
        del x_t[:], x_noised[:]

    running_avg /= (i+1)
    BETA = BETA * opts["beta_decay"]
    with torch.no_grad():
        x_val_plain = utils.noise_input(x_val, banned_modality=ban_list, prob=[1.0, 0.0])
        x_val_noised = utils.noise_input(x_val, banned_modality=ban_list, prob=[0.0, 0.5, 0.5],
                                         direction="both", modality_noise=True)
        mse_val = model.loss(x_val_plain, x_val, lambd=1.0, beta=0.0, sample=False, reduce=True, mse=True)
        mse_val_noised = model.loss(x_val_noised, x_val, lambd=1.0, beta=0.0, sample=False, reduce=True, mse=True)

    del x_val_plain[:], x_val_noised[:]

    writer.add_scalar("Epoch loss", running_avg, e)
    writer.add_scalar("MSE val",  mse_val, e)
    writer.add_scalar("MSE val noised",  mse_val_noised, e)
    print("Epoch %d loss: %.5f, MSE val: %.5f, Noised: %.5f" % (e+1, running_avg, mse_val, mse_val_noised))

    if mse_val_noised < best_error:
        best_error = mse_val_noised.item()
        model.save(opts["save"], "multivae_best")

    model.save(opts["save"], "multivae_last")
