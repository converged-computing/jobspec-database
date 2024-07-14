import argparse
import os
import sys
import warnings

parser = argparse.ArgumentParser(description = "")
parser.add_argument("--adata", metavar = "adata", type = str)
parser.add_argument("--cellsets", metavar = "cellsets", type = str)
parser.add_argument("--t_initial", metavar = "t_initial", type = float, default = 8)
parser.add_argument("--t_final", metavar = "t_final", type = float, default = 14.5)
parser.add_argument("--numcells", metavar = "numcells", type = int, default = 250)
parser.add_argument("--pcadim", metavar = "pcadim", type = int, default = 10)
parser.add_argument("--lamda", metavar = "lamda", type = float, default = 2.5e-3)
parser.add_argument("--eps_df", metavar = "eps_df", type = float, default = 0.025)
parser.add_argument("--eps_eff", metavar = "eps_eff", type = float, default = 0.1)
parser.add_argument("--dt0", metavar = "dt0", type = float, default = 0.5)
parser.add_argument("--steps_ng", metavar = "steps_ng", type = int, default = 25)
parser.add_argument("--steps_g", metavar = "steps_g", type = int, default = 25)
parser.add_argument("--outfile", metavar = "outfile", type = str, default = "reprog.out")
parser.add_argument("--srand", metavar = "srand", type = int, default = 0)

args = parser.parse_args()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wot
import pegasus as pg
from math import *
import ot
import pykeops

import torch
from torch.autograd import grad, Variable
import copy
import scipy as sp
from scipy import stats
import sklearn
import dill
import anndata
import gwot
from gwot import models, util, ts, anndata_utils, altsolver

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

np.random.seed(args.srand)
adata = anndata.read_h5ad(args.adata)
cell_sets = wot.io.read_sets(args.cellsets)
obs_celltype = pd.DataFrame(cell_sets.X, dtype = bool, columns = cell_sets.var.index, index = cell_sets.obs.index)
idx_batch = pd.Index.intersection(cell_sets.obs.index, adata.obs.index)
adata_batch = adata[idx_batch, ]
adata_batch.obs = pd.concat([adata_batch.obs, obs_celltype.loc[idx_batch, ]], axis = 1)

adata_batch = adata_batch[adata_batch.obs.day > 4, ]
del adata_batch.uns

adata_batch.obs = adata_batch.obs.drop(columns = ["MET", "OPC", "Astrocyte", "Neuron", "SpongioTropho", "ProgenitorTropho", "SpiralArteryTrophoGiant", "RadialGlia"])
adata_batch.obs.loc[:, "None"] = (adata_batch.obs.iloc[:, 3:].sum(1) == 0)
adata_batch = adata_batch[adata_batch.obs.iloc[:, 3:].sum(1) == 1, :]

adata_ts = adata_batch[~np.isnan(adata_batch.obs.day), :] 
pg.pca(adata_ts, n_components = args.pcadim, features = None)
t_map = np.array(adata_ts.obs.day.unique())
t_map = t_map[t_map >= args.t_initial]
t_map = t_map[t_map < args.t_final]
t_map.sort()

props = [adata_batch[adata_batch.obs.day == i, :].obs.iloc[:, 3:].sum(0) for i in t_map]
props = [p/p.sum() for p in props]

props_perturb = [np.random.dirichlet(5*p + 1e-3) for p in props]
# def normalise(x):
#     return x/x.sum()
# props_perturb = [normalise(p * np.random.uniform(size = len(p))) for p in props]

celltypes = adata_ts.obs.iloc[:, 3:].columns
adata_subsamp = []
for i in range(len(t_map)):
    p = props_perturb[i]/props[i]
    p[np.isnan(p)] = 0
    p[np.isinf(p)] = 0
    p = p/p.sum()
    adata_ = adata_ts[adata_ts.obs.day == t_map[i]]
    q = np.zeros(adata_.shape[0])
    for c in celltypes:
        q[adata_.obs.loc[:, c]] = p.loc[c]
    q = q/q.sum()
    adata_subsamp += [adata_[np.random.choice(adata_.shape[0], size = args.numcells, p = q), :], ]

adata_s = adata_subsamp[0].concatenate(adata_subsamp[1:])
adata_s.obsm["X_pca_orig"] = adata_s.obsm["X_pca"]
days = adata_s.obs.day.unique()
days_tot = adata_s.obs.day.unique().shape[0]

props_subsamp = [adata_s[adata_s.obs.day == i, :].obs.iloc[:, 3:].sum(0) for i in t_map]
props_subsamp = [p/p.sum() for p in props_subsamp]

# recompute PCA for subsampled data
pg.pca(adata_s, n_components = adata.obsm["X_pca"].shape[1], features = None)
pg.neighbors(adata_s)
pg.diffmap(adata_s)
adata_s.obsm['X_fle'] = np.array(adata_s.obsm['X_fle'])

c_means = np.array([gwot.anndata_utils.get_C_mean(adata_s, t_map[i], t_next = t_map[i+1], mode = "tr") for i in range(0, len(t_map[:-1]))])
c_means_self = np.array([gwot.anndata_utils.get_C_mean(adata_s, t, mode = "self") for t in t_map])

dt = np.array([t_map[i+1] - t_map[i] for i in range(0, len(t_map)-1)])

idx = [np.where(adata_s.obs.day == t)[0] for t in t_map]
t_idx = np.zeros(adata_s.shape[0], dtype = np.int64)
for i in range(0, len(idx)):
    t_idx[idx[i]] = i

dt0 = args.dt0/dt.sum()
tsdata = gwot.ts.TimeSeries(x = np.array(adata_s.obsm["X_pca"], dtype = np.float64), 
                dt = dt/dt.sum(), 
                t_idx = t_idx, 
                D = args.eps_eff/(2*dt0))

# solve without growth
model_ng = gwot.models.OTModel(tsdata, lamda = args.lamda,
        eps_df = args.eps_df*torch.ones(tsdata.T).cuda(), 
        lamda_i = torch.ones(tsdata.T).cuda(), 
        m_i = torch.ones(tsdata.T).cuda(),
        g_i = torch.ones(tsdata.T, tsdata.x.shape[0]).cuda(),
        kappa = torch.from_numpy(10/tsdata.dt).cuda(), 
        c_scale = torch.from_numpy(c_means).cuda(),
        c_scale_df = torch.from_numpy(c_means_self).cuda(),
        growth_constraint="KL",
        pi_0 = "uniform", 
        device = device,
        use_keops = True)

model_ng.solve_lbfgs(steps = args.steps_g, max_iter = 25, lr = 1, history_size = 50, line_search_fn = 'strong_wolfe', factor = 2, tol = 1e-5, retry_max = 0)

with torch.no_grad():
    P_ng = model_ng.get_P()
    R_ng = model_ng.get_R()
p_ng = (P_ng.T/P_ng.sum(dim = 1)).T
r_ng = (R_ng.T/R_ng.sum(dim = 1)).T

g = torch.from_numpy(np.array(adata_s.obs.cell_growth_rate)).cuda()
g_i = torch.stack([(torch.from_numpy(np.array(adata_s.obs.cell_growth_rate)).cuda())**(t_map[i+1]-t_map[i]) for i in range(0, tsdata.T-1)]).cuda()
r = torch.stack([(p_ng[i, :]*(g**(t_map[i+1]-t_map[i]))).sum() for i in range(0, model_ng.ts.T-1)])
m_i = torch.cumprod(torch.cat([torch.tensor([1., ]).cuda(), r]), dim = 0)

model_g = gwot.models.OTModel(tsdata, lamda = args.lamda,
        eps_df = args.eps_df*torch.ones(tsdata.T).cuda(), 
        lamda_i = torch.ones(tsdata.T).cuda(), 
        m_i = m_i,
        g_i = g_i.cuda(),
        kappa = torch.from_numpy(10/tsdata.dt).cuda(), 
        c_scale = torch.from_numpy(c_means).cuda(),
        c_scale_df = torch.from_numpy(c_means_self).cuda(),
        growth_constraint="KL",
        pi_0 = "uniform", 
        device = device,
        use_keops = True)

for _ in range(4):
    model_g.solve_lbfgs(steps = args.steps_g, max_iter = 25, lr = 1, history_size = 10, line_search_fn = 'strong_wolfe', factor = 2, tol = 1e-5, retry_max = 0)

with torch.no_grad():
    P = model_g.get_P()
    R = model_g.get_R()
p = (P.T/P.sum(dim = 1)).T
r = (R.T/R.sum(dim = 1)).T

props_gwot = [np.array([p[i, adata_s.obs.loc[:, c]].sum().item() for c in celltypes]) for i in range(len(t_map))]
for x in props_gwot:
    x = x/x.sum()

print("Done, writing results...")
with open(args.outfile, "wb") as f:
    dill.dump({"props" : props, "props_perturb" : props_perturb, "props_subsamp" : props_subsamp, "props_gwot" : props_gwot}, f)

