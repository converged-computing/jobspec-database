#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import concurrent.futures
from itertools import permutations
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from networkx.convert_matrix import from_pandas_adjacency

# local imports
import BLEval as ev 

def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(
        description='Run pathway reconstruction pipeline.')

    parser.add_argument('-c','--config', default='config.yaml',
        help="Configuration file containing list of datasets "
              "algorithms and output specifications.\n")

    parser.add_argument('--dataset_names', default=None,
        help='Comma delimited list of datasets from the config file to run.')

    parser.add_argument('--algorithm_names', default=None,
        help='Comma delimited list of algorithms from the config file to run; \
              overrides the should_run parameter in the config file for the algorithms.')

    parser.add_argument('--output_prefix', default=None,
        help='File name prefix for the evaluation output; \
              overrides the output_prefix parameter in the config file.')

    parser.add_argument('--output_dir', default=None,
        help='Output directory prefix for the evaluation output; \
              overrides the output_dir parameter in the config file.')
    
    parser.add_argument('-a', '--auc', action="store_true", default=False,
        help="Compute median of areas under Precision-Recall and ROC curves;\
              possible edges are pairs of nodes in the ground truth network;\
              original BEELINE implementation.\n")

    parser.add_argument('--auc2', action="store_true", default=False,
        help="Compute median of areas under Precision-Recall and ROC curves;\
              possible edges are pairs of nodes in the ground truth network;\
              optimized implementation with dictionary.\n")

    parser.add_argument('--auc3', action="store_true", default=False,
        help="Compute median of areas under Precision-Recall and ROC curves;\
              possible edges are pairs of nodes in the ground truth network;\
              new implemenation with numpy arrays.\n")

    parser.add_argument('--auc4', action="store_true", default=False,
        help="Compute median of areas under Precision-Recall and ROC curves;\
              possible edges are pairs of nodes in the expression data;\
              new implemenation with numpy arrays.\n")
    
    parser.add_argument('--pauc4', action="store_true", default=False,
        help="Compute partial areas under Precision-Recall and ROC curves;\
              possible edges are pairs of nodes in the expression data;\
              new implemenation with numpy arrays.\n")
    
    parser.add_argument('-j', '--jaccard', action="store_true", default=False,
      help="Compute median Jaccard index of predicted top-k networks "
      "for each algorithm for a given set of datasets generated "
      "from the same ground truth network.\n")

    parser.add_argument('-r', '--spearman', action="store_true", default=False,
      help="Compute median Spearman Corr. of predicted edges "
      "for each algorithm  for a given set of datasets generated "
      " from the same ground truth network.\n")

    parser.add_argument('-t', '--time', action="store_true", default=False,
      help="Analyze time taken by each algorithm for a.\n")
    
    parser.add_argument('-e', '--epr', action="store_true", default=False,
      help="Compute median early precision.")
    
    parser.add_argument('-s','--sepr', action="store_true", default=False,
      help="Analyze median (signed) early precision for activation and inhibitory edges.")

    parser.add_argument('-m','--motifs', action="store_true", default=False,
      help="Compute network motifs in the predicted top-k networks.")
    
    parser.add_argument('-p','--paths', action="store_true", default=False,
      help="Compute path length statistics on the predicted top-k networks.")

    parser.add_argument('-b','--borda', action="store_true", default=False,
      help="Compute edge ranked list using the various Borda aggregatio methods.")
        
    return parser

def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()
    
    return opts

def main():
    opts = parse_arguments()
    config_file = opts.config

    evalConfig = None

    with open(config_file, 'r') as conf:
        #evalConfig = ev.ConfigParser.parse(conf,opts)
        evalSummarizer = ev.ConfigParser.parse(conf,opts)

    print('\nPost-run evaluation started...')
    #evalSummarizer = ev.BLEval(evalConfig.input_settings, evalConfig.output_settings)
    print(evalSummarizer.input_settings.algorithms)
    print(evalSummarizer.input_settings.datasets)
    
    outDir = str(evalSummarizer.output_settings.base_dir) + \
            str(evalSummarizer.input_settings.datadir).split("inputs")[1] + "/"+\
            str(evalSummarizer.output_settings.output_prefix) + "-"
    print(outDir)
    
    # Compute and plot ROC, PRC and report median AUROC, AUPRC
    # possible edges are between nodes in the ground truth
    if (opts.auc):
        print('\n\nComputing areas under ROC and PR curves...')

        AUPRC, AUROC = evalSummarizer.computeAUC()
        AUPRC.to_csv(outDir+'AUPRC.csv')
        AUROC.to_csv(outDir+'AUROC.csv')

    # Compute and plot ROC, PRC and report median AUROC, AUPRC
    # possible edges are between nodes in the ground truth
    if (opts.auc2):
        AUPRC2, AUROC2 = evalSummarizer.computeAUC2()
        AUPRC2.to_csv(outDir+'AUPRC2.csv')
        AUROC2.to_csv(outDir+'AUROC2.csv')

    # Compute and plot ROC, PRC and report median AUROC, AUPRC
    # possible edges are between nodes in the ground truth
    if (opts.auc3):
        AUPRC3, AUROC3, AveP3 = evalSummarizer.computeAUC3()
        AUPRC3.to_csv(outDir+'AUPRC3.csv')
        AUROC3.to_csv(outDir+'AUROC3.csv')
        AveP3.to_csv(outDir+'AveP3.csv')

    # Compute and plot ROC, PRC and report median AUROC, AUPRC
    # possible edges are between nodes in the expression data
    if (opts.auc4):
        AUPRC4, AUROC4, AveP4 = evalSummarizer.computeAUC4()
        AUPRC4.to_csv(outDir+'AUPRC4.csv')
        AUROC4.to_csv(outDir+'AUROC4.csv')
        AveP4.to_csv(outDir+'AveP4.csv')

    # Compute partial AUROC, AUPRC
    # possible edges are between nodes in the expression data
    if (opts.pauc4):
        PAUC4 = evalSummarizer.computePAUC4()
        for metric_name,metric_df in PAUC4.items():
            metric_df.to_csv(outDir+metric_name+'4.csv')
        
    # Compute Jaccard index    
    if (opts.jaccard):
        print('\n\nComputing Jaccard index...')

        jaccDict = evalSummarizer.computeJaccard()
        jaccDict.to_csv(outDir + "Jaccard.csv")
        
    # Compute Spearman correlation scores
    if (opts.spearman):
        print('\n\nComputing Spearman\'s correlation...')

        corrDict = evalSummarizer.computeSpearman()
        corrDict.to_csv(outDir + "Spearman.csv")
        
    # Compute median time taken
    if (opts.time):
        print('\n\nComputing time taken...')

        TimeDict = evalSummarizer.parseTime()
        pd.DataFrame(TimeDict).to_csv(outDir+'Times.csv')
    
    # Compute early precision
    if (opts.epr):
        print('\n\nComputing early precision values...')
        ePRDF = evalSummarizer.computeEarlyPrec()
        ePRDF.to_csv(outDir + "EPr.csv")
        ePRDF = evalSummarizer.computeEarlyPrecTFOnly()
        ePRDF.to_csv(outDir + "EPr-TF.csv")
        
    # Compute early precision for activation and inhibitory edges
    if (opts.sepr):
        print('\n\nComputing early precision values for activation and inhibitory edges...')
        
        actDF, inhDF = evalSummarizer.computeSignedEPrec()
        actDF.to_csv(outDir + "EPr-Activation.csv")
        inhDF.to_csv(outDir + "EPr-Inhibitory.csv")

    # Compute median time taken
    if (opts.motifs):
        print('\n\nComputing network motifs...')

        FBL, FFL, MI = evalSummarizer.computeNetMotifs()
        FBL.to_csv(outDir+'NetworkMotifs-FBL.csv')
        FFL.to_csv(outDir+'NetworkMotifs-FFL.csv')
        MI.to_csv(outDir+'NetworkMotifs-MI.csv')
        
    # Compute path statistics such as number of TP, FP, 
    # and path lengths among TP in the top-k networks.
    if (opts.paths):
        print('\n\nComputing path length statistics on predicted networks...')
        evalSummarizer.computePaths()
        
   # Compute edge ranked list using the borda method
    if (opts.borda):
        print('\n\nComputing edge ranked list using the borda method')
        evalSummarizer.computeBorda()


    print('\n\nEvaluation complete...\n')


if __name__ == '__main__':
  main()
