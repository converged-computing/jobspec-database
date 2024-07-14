#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import numpy.random as npr
from scipy.stats import gamma, norm, invgamma, t


data_file = "CombineDS.csv"
header_file = "CombineDS_header.csv"
logZ_file = "logz.txt"


def pdf_normal_inverse_gamma_v(x, μ, v, α, β):
    σ = np.sqrt(β/α/v)
    return t.pdf(x, 2.0*α, μ, σ)


def row(data):
    idx = npr.choice(len(data), 1, p=data['W'])
    return data.loc[idx]


def sample(row, dist, parameter_map):
    kwargs = {k: row[v] for k, v in parameter_map.items()}
    if dist == "normal_inverse_gamma_v":
        kwargs = {
            'df': 2.0*kwargs['α'],
            'loc': kwargs['μ'],
            'scale': np.sqrt(kwargs['β']/kwargs['α']/kwargs['v'])
        }
        dist = t
    return dist.rvs(**kwargs)


columns = pd.read_csv(header_file, delimiter=', ', engine='python').columns
data = pd.read_csv(data_file, header=None, delimiter=', ', engine='python')
logZ = pd.read_csv(logZ_file, header=None)
data.columns = columns
data['logZ'] = logZ[0].repeat(len(data) / len(logZ)).values
data['W'] = np.exp(data['logZ'] - data['logZ'].max())
data['W'] /= data['W'].sum()

r = row(data)
l = sample(r, gamma, {'a': 'lambda_0.k', 'scale': 'lambda_0.theta'})
m = sample(r, gamma, {'a': 'mu_0.k', 'scale': 'mu_0.theta'})
print(json.dumps(    {
    'lambda': l,
    'mu': m,
    'epsilon': m/l,
    'nu': sample(r, gamma, {'a': 'nu_0.k', 'scale': 'nu_0.theta'}),
    'log_α_gbm': sample(r, "normal_inverse_gamma_v", {'μ': 'alphaSigma_gbm.m0', 'v': 'alphaSigma_gbm.v', 'α': 'alphaSigma_gbm.a', 'β': 'alphaSigma_gbm.b'}),
    'sigma2_gbm': sample(r, invgamma, {'a': 'alphaSigma_gbm.a', 'scale': 'alphaSigma_gbm.b'}),
    'log_α': sample(r, "normal_inverse_gamma_v", {'μ': 'alphaSigma.m0', 'v': 'alphaSigma.v', 'α': 'alphaSigma.a', 'β': 'alphaSigma.b'}),
    'sigma2': sample(r, invgamma, {'a': 'alphaSigma.a', 'scale': 'alphaSigma.b'})
   
}, ensure_ascii=False))
