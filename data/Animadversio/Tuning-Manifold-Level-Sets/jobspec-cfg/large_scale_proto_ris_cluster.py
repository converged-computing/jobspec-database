import re
import os
import sys
import torch
from lpips import LPIPS
import numpy as np
from os.path import join
import matplotlib.pylab as plt
from easydict import EasyDict
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from core.utils.GAN_utils import upconvGAN
from core.utils.CNN_scorers import TorchScorer
from core.utils.plot_utils import show_imgrid, save_imgrid, save_imgrid_by_row
from core.proto_analysis_lib import sweep_folder, visualize_proto_by_level, visualize_score_imdist, \
        calc_proto_diversity_per_bin, visualize_diversity_by_bin, filter_visualize_codes
from core.latent_explore_lib import latent_explore_batch, latent_diversity_explore, \
    latent_diversity_explore_wRF_fixval, search_peak_evol, search_peak_gradient, calc_rfmap
from argparse import ArgumentParser

sys.path.append("/home/binxu.w/Tuning-Manifold-Level-Sets")

parser = ArgumentParser()
parser.add_argument('--units', nargs='+', type=str, required=True)
parser.add_argument('--chan_rng', nargs=2, type=int, default=[0, 10])
parser.add_argument('--repn', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=40)
args = parser.parse_args()

netname = args.units[0]
layername = args.units[1]
chan_rng = args.chan_rng
repn = args.repn
batch_size = args.batch_size

if len(args.units) == 5:
    centpos = (int(args.units[3]), int(args.units[4]))
    unit_tup = (netname, layername, int(args.units[2]), int(args.units[3]), int(args.units[4]))
elif len(args.units) == 3:
    centpos = None
    unit_tup = (netname, layername, int(args.units[2]))
else:
    raise ValueError("args.units should be a 3 element or 5 element tuple!")

# repn = 1
# batch_size = 40

Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)


for channel in range(chan_rng[0], chan_rng[1]):
    if len(unit_tup) == 5:
        unit_tup_new = (netname, layername, channel, unit_tup[3], unit_tup[4])
    else:
        unit_tup_new = (netname, layername, channel)
    scorer = TorchScorer(netname)
    scorer.select_unit(unit_tup_new, allow_grad=True)
    unitlabel = "%s-%d" % (scorer.layer.replace(".Bottleneck", "-Btn").strip("."), scorer.chan)
    outroot = join(r"/scratch1/fs1/holy/insilico_exp/proto_diversity", netname)
    outrf_dir = join(outroot, unitlabel+"_rf")
    os.makedirs(outrf_dir, exist_ok=True)
    # Compute RF map for the unit.
    rfmaptsr, rfmapnp, fitdict = calc_rfmap(scorer, outrf_dir, label=unitlabel, use_fit=True, )
    #%%
    # Perform evolution with CMA and gradient. Save the code and image
    z_evol, img_evol, resp_evol, resp_all, z_all = search_peak_evol(G, scorer, nstep=100)
    z_base, img_base, resp_base = search_peak_gradient(G, scorer, z_evol, resp_evol, nstep=100)
    resp_base = torch.tensor(resp_base).float().cuda()
    save_imgrid(img_base, join(outrf_dir, "proto_peak.png"))
    save_imgrid(img_base*rfmaptsr, join(outrf_dir, "proto_peak_rf.png"))
    torch.save(dict(z_base=z_base, img_base=img_base, resp_base=resp_base,
                    z_evol=z_evol, img_evol=img_evol, resp_evol=resp_evol,
                    unit_tuple=unit_tup, unitlabel=unitlabel),
               join(outrf_dir, "proto_optim.pt"))
    #%%
    for ratio in np.arange(0.0, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_max"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(imgdist_obj="max", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    dz_sigma=1.5, noise_std=0.3, steps=75, batch_size=batch_size)
        S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * 0.1)

    for ratio in np.arange(0.0, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_none"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(imgdist_obj="none", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    dz_sigma=1.5, noise_std=0.3, steps=75, batch_size=batch_size)
        S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * 0.1) #ratio * 0.2

    for ratio in np.arange(0.0, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_min"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(imgdist_obj="min", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    dz_sigma=1.5, noise_std=0.3, steps=75, batch_size=batch_size)
        S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * 0.1) #ratio * 0.2

    for ratio in np.arange(0.0, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_max_abinit"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(imgdist_obj="max", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    dz_sigma=3, noise_std=0.3, steps=75, batch_size=batch_size)
        S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=True,
                                       err=resp_base.item() * 0.1)

    for ratio in np.arange(0.0, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_none_abinit"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(imgdist_obj="none", alpha_img=0.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    dz_sigma=3, noise_std=0.3, steps=75, batch_size=batch_size)
        S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=True,
                                       err=resp_base.item() * 0.1)

    for suffix in ["min", "max", "none", "max_abinit", "none_abinit"]:
        sumdict, sumdir = sweep_folder(outrf_dir, dirnm_pattern=f"fix.*_{suffix}$",
                                       sum_sfx=f"summary_{suffix}")
        visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25, )
        visualize_score_imdist(sumdict, sumdir, )
        df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir,
                                                                   bin_width=0.10, distsampleN=40)
        visualize_diversity_by_bin(df, sumdir)
