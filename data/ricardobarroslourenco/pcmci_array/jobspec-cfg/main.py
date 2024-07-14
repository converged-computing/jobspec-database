# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:04:58 2023

@author: user
"""

import os
from osgeo import gdal
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import time
from sklearn.linear_model import LinearRegression
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.causal_effects import CausalEffects
from functools import partial
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed
import pandas as pd
import argparse
import time


def path_effect(graph, dataframe, X, Y, mediation=None, mask_type=None):
    causal_effects = CausalEffects(graph, graph_type='stationary_dag',
                                   X=X, Y=Y,
                                   S=None,
                                   verbosity=0)
    dox_vals = np.linspace(0., 1., 2)

    # Fit causal effect model from observational data
    causal_effects.fit_wright_effect(
        dataframe=dataframe,
        mask_type=mask_type,
        mediation=mediation,  # 'direct',
        method='parents',
        data_transform=None,  # sklearn.preprocessing.StandardScaler(),
    )

    # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    intervention_data = np.repeat(dox_vals.reshape(len(dox_vals), 1), axis=1, repeats=len(X))
    pred_Y = causal_effects.predict_wright_effect(
        intervention_data=intervention_data)
    return pred_Y[1] - pred_Y[0]


def total_effect(dgraph, dataframe, X, Y, mask_type=None):
    causal_effects = CausalEffects(dgraph, graph_type='stationary_dag',
                                   X=X, Y=Y,
                                   S=None,  # S could be a modulating variable to estimate conditional causal effects
                                   verbosity=0)
    # Optional data transform
    data_transform = StandardScaler()

    # Confidence interval range
    conf_lev = 0.9

    # Fit causal effect model from observational data
    causal_effects.fit_total_effect(
        dataframe=dataframe,
        mask_type=mask_type,
        estimator=LinearRegression(),
        data_transform=data_transform,
    )

    # Fit bootstrap causal effect model
    causal_effects.fit_bootstrap_of(
        method='fit_total_effect',
        method_args={'dataframe': dataframe,
                     'mask_type': 'y',
                     'estimator': LinearRegression(),
                     'data_transform': data_transform,
                     },
        seed=4
    )

    # Define interventions
    dox_vals = np.linspace(0., 1., 2)

    # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    intervention_data = np.repeat(dox_vals.reshape(len(dox_vals), 1), axis=1, repeats=len(X))
    pred_Y = causal_effects.predict_total_effect(
        intervention_data=intervention_data)

    # Bootstrap: Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    intervention_data = np.repeat(dox_vals.reshape(len(dox_vals), 1), axis=1, repeats=len(X))
    conf = causal_effects.predict_bootstrap_of(
        method='predict_total_effect',
        method_args={'intervention_data': intervention_data},
        conf_lev=conf_lev)

    return pred_Y[1] - pred_Y[0], conf[0, 1] - conf[0, 0], conf[1, 1] - conf[1, 0]


def causal_diseff(index, row, col, ti_path, evi_path, var_names0,
                  var_names1):  # (args.sm,args.evi,args.var_names0,args.car_names1,row,col)
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    # row, col = hl[0], hl[1]
    datatime = np.linspace(2019 - 8. / 12, 2022 - 1. / 12., 44)

    ti_data = np.load(ti_path)
    evidata = np.load(evi_path)
    data = np.array([ti_data[:, row, col], evidata[:, row, col]]).T
    cr = np.where(data == -9999)

    if cr[0].shape[0] > 19:
        # continue
        s_lag, s_ce, sconf_0, sconf_1 = -9999, -9999, -9999, -9999
    else:
        # residual_here is dataframe containing both image stacks
        residuals_here = pp.DataFrame(np.copy(data), var_names=[var_names0, var_names1], datatime=datatime,
                                      missing_flag=-9999.)
        ##################################causal discovery################################
        T, N = residuals_here.values[0].shape  # T:44, N:2
        tau_max = 12
        pc_alpha = 0.05
        # set link_assumptions
        link_assumptions = {j: {(i, -tau): 'o?o' for i in range(N) for tau in range(tau_max + 1)
                                if (i, -tau) != (j, 0)}
                            for j in range(N)}
        link_assumptions[0] = {(0, -1): '-->'}
        link_assumptions[1] = {(i, -tau): 'o?o' for i in range(N) for tau in range(tau_max + 1)
                               if (i != 1) or (i, -tau) == (1, -1)}
        parcorr = ParCorr(mask_type=None, significance='analytic')
        pcmci = PCMCI(
            dataframe=residuals_here,
            # dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0)
        # pcmci.verbosity = 1
        try:

            results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha,
                                      alpha_level=0.05,
                                      link_assumptions=link_assumptions)
            # deal with the matrix and graph in order to make a proper graph
            dmatrix = np.copy(results['val_matrix'])
            dgraph = np.where(results['graph'] != '', '', results['graph'])
            dmatrix = np.where(results['graph'] == 'o-o', 0, dmatrix)
            for i in range(2):
                if i == 0:
                    ind = np.argmax(abs(dmatrix[i, 0, :]))
                    dgraph[i, 0, ind] = '-->'
                    ind1 = np.argmax(abs(dmatrix[i, 1, :]))
                    dgraph[i, 1, ind1] = '-->'
                    if ind1 == 0:
                        dgraph[1, i, ind1] = '<--'
                        dmatrix[1, i, ind1] = 0
                    s_lag = ind1
                if i == 1:
                    ind = np.argmax(abs(dmatrix[i, 1, :]))
                    dgraph[i, 1, ind] = '-->'

            ##################################causal effect###################################

            try:

                ce = total_effect(dgraph, residuals_here,
                                  X=[(0, -s_lag)], Y=[(1, 0)], mask_type=None)
                #             s_ce1 = path_effect(dgraph, residuals_here,
                #                                 X=[(0,-s_lag)], Y=[(1,0)], mask_type='y')
                s_ce, sconf_0, sconf_1 = ce[0], ce[1], ce[2]
            except ValueError:
                s_ce, sconf_0, sconf_1 = -9999, -9999, -9999
        except ValueError:
            s_lag, s_ce, sconf_0, sconf_1 = -9999, -9999, -9999, -9999
    return index, row, col, s_lag, s_ce, sconf_0, sconf_1


def loop_across_parameters(begin_idx, end_index):
    df = pd.read_csv('hl.csv')

    columns = ['index', 'row', 'col', 's_lag', 's_ce', 'sconf_0', 'sconf_1']
    output_df = pd.DataFrame(columns=columns)

    for line in range(begin_idx + 1, end_index):
        buffer_line = df.loc[df['index'] == line]
        results = causal_diseff(*buffer_line.values.tolist()[0])
        output_df = output_df.append(pd.DataFrame(results, columns=columns))
    return output_df


if __name__ == '__main__':
    start_time = time.time()
    # Set parser for arguments submitted to main.py (with data types enforced)
    parser = argparse.ArgumentParser(description='Process input parsed from hl.csv')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--begin_idx', type=int, required=True)
    parser.add_argument('--end_idx', type=int, required=True)

    args = parser.parse_args()
    # main(args.index, args.temperature, args.category)
    # output = pd.DataFrame(causal_diseff(args.index, args.row,args.col,args.sm,args.evi,args.var_names0,args.var_names1))
    output = loop_across_parameters(begin_idx=args.begin_idx, end_index=args.end_idx)
    output.to_csv(str('/home/xinran22/scratch/SIF_SM/' + 'index_' + args.index + '.csv'))
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
