"""Test multimodal VAE for UR10 data."""
import os
import argparse

import yaml
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import models
import data
import utils

parser = argparse.ArgumentParser("Test multimodal VAE.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-banned", help="banned modalities", nargs="+", type=int, required=True)
parser.add_argument("-prefix", help="output prefix", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
print(yaml.dump(opts))

trainset = data.MyDataset(opts["data"], modality=opts["modality"], action=opts["action"], mode="train")
testset = data.MyDataset(opts["data"], modality=opts["modality"], action=opts["action"], mode="test")

model = models.MultiVAE(
    in_blocks=opts["in_blocks"],
    in_shared=opts["in_shared"],
    out_shared=opts["out_shared"],
    out_blocks=opts["out_blocks"],
    init=opts["init_method"])
model.to(opts["device"])
model.load(opts["save"], "multivae_best")
model.cpu().eval()
print(model)

out_folder = os.path.join(opts["save"], "outs")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
outpath = os.path.join(out_folder, args.prefix+"-result.txt")
if os.path.exists(outpath):
    os.remove(outpath)
outfile = open(outpath, "a")


N = 10

k_step = [[[], [], [], [], [], [], [], [], [], [], []] for _ in trainset.modality]
condition_idx = [68, 30, 60, 35, 72, 45, 61, 35, 43, 48]  # nvm about this

for exp in range(N):
    exp_folder = os.path.join(out_folder, str(exp))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    x_test = testset.get_trajectory(exp)
    x_test = trainset.normalize(x_test)
    L = x_test[0].shape[0]
    # condition on the half-point on the trajectory
    # start_idx = L // 2
    start_idx = condition_idx[exp]
    forward_t = L - start_idx
    backward_t = start_idx
    x_noised = utils.noise_input(x_test, args.banned, prob=[0., 1.], direction="forward", modality_noise=False)
    x_condition = [x[start_idx:(start_idx+1)] for x in x_noised]
    # x[t+1] <- x[t]
    for i, dim in enumerate(trainset.dims):
        x_condition[i][:, dim:] = x_condition[i][:, :dim]

    with torch.no_grad():
        # one-step forward prediction
        _, _, y_onestep, _ = model(x_noised, sample=False)

        # forecasting
        y_forecast = model.forecast(x_condition, forward_t, backward_t, banned_modality=args.banned)

        # clamp to limits
        for y_i in y_onestep:
            y_i.clamp_(-1., 1.)
        for y_i in y_forecast:
            y_i.clamp_(-1., 1.)

        x_test = trainset.denormalize(x_test)
        y_onestep = trainset.denormalize(y_onestep)
        y_forecast = trainset.denormalize(y_forecast)

        for i, dim in enumerate(trainset.dims):
            # error at k-step prediction instead of one-step.
            # As k increases, models prediction performance drops dramatically
            # due to cascading error. This is not the case with CNMP since
            # we query each point independently, so the error does not accumulate.
            for k in range(min(11, L-start_idx)):
                diff = (x_test[i][start_idx+k, :dim] - y_forecast[i][start_idx+k, :dim]).abs()
                # for the img modality, average out pixels
                if i == 0:
                    diff = diff.mean()
                k_step[i][k].append(diff)

            # plot the error for each dimension of each modality (other than img)
            if i != 0:
                for j in range(dim):
                    plt.plot(x_test[i][:, j+dim], c="k", label="Truth")
                    plt.plot(y_onestep[i][:, j+dim], c="b", label="One-step")
                    plt.plot(y_forecast[i][:, j+dim], c="m", label="Forecast")
                    plt.scatter(start_idx-1, x_test[i][start_idx-1, j+dim], c="r", marker="x", label="Condition point")
                    plt.ylabel("$q_%d$" % (j))
                    plt.xlabel("$t$")
                    plt.legend()
                    # save to pdf
                    temp_path = os.path.join(exp_folder, args.prefix+("-%s-%d.pdf" % (trainset.modality[i], j)))
                    pp = PdfPages(temp_path)
                    pp.savefig()
                    pp.close()
                    plt.close()

            # calculate mean abs. error for each dimension of each modality (other than img)
            err_onestep = (x_test[i][:, dim:] - y_onestep[i][:, dim:]).abs().mean(dim=0)
            err_forecast = (x_test[i][:, dim:] - y_forecast[i][:, dim:]).abs().mean(dim=0)
            # avg over channels, height, and width if it's image
            print("%s modality error" % trainset.modality[i], file=outfile)
            if i == 0:
                err_onestep = err_onestep.mean()
                err_forecast = err_forecast.mean()
                print("onestep error: %.4f" % err_onestep, file=outfile)
                print("forecast error: %.4f" % err_forecast, file=outfile)
            else:
                formatted = ", ".join(["%.4f" for _ in range(dim)])
                print("onestep error: " + (formatted % tuple(err_onestep)), file=outfile)
                print("forecast error: " + (formatted % tuple(err_onestep)), file=outfile)

            print("k-step error", file=outfile)
            for k in range(11):
                print("k=%d" % k, file=outfile)
                err = k_step[i][k][0]
                if i == 0:
                    print("%.4f" % err, file=outfile)
                else:
                    formatted = ", ".join(["%.4f" for _ in range(dim)])
                    print(formatted % tuple(err), file=outfile)

        # record image trajectory as a video
        # concat real and predicted video in the width dimension
        # and permute to [time, height, width, channel] to save the video
        x_cat = torch.cat([x_test[0][:, 3:], y_onestep[0][:, 3:]], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, args.prefix+"-onestep.mp4"), x_cat.byte(), fps=30)
        x_cat = torch.cat([x_test[0][:, 3:], y_forecast[0][:, 3:]], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, args.prefix+"-forecast.mp4"), x_cat.byte(), fps=30)
        # difference video. This should be all black if the model predicts perfectly.
        x_diff = (x_test[0][:, 3:] - y_forecast[0][:, 3:]).abs().permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, args.prefix+"-diff.mp4"), x_diff.byte(), fps=30)
