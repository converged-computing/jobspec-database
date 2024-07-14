import numpy as np
import pandas as pd
import scipy as sp
import math as math
import time as time
import argparse
import textwrap
import os
#from pandas_plink import read_plink
from sklearn import linear_model
import sklearn as skl
from pysnptools.snpreader import Bed
 
def runLasso(genPATH,trait,indexVAR,outDIR):

    # constants
    nstep = 200
    # origininally lamratio = 0.04
    # lamratio of 0.01 gets hgt predictors to ~25k snps
    # setting it to 0.04 again...
    lamratio = 0.01
    if trait == 'bioMarkers2.19':
        lamratio = 0.004

    print(lamratio)
        
    # inputs
    index = indexVAR
    gwasPATH = outDIR+"gwas."+str(trait)+"."+str(index)+".csv"
    phenPATH = outDIR+str(trait)+".pruned.txt"
    trainPATH = outDIR+"TrainSet."+str(index)+".txt"

    # outputs
    lamPATH = outDIR+"lasso.lambdas."+str(trait)+"."+str(index)+".txt"
    betaPATH = outDIR+"lasso.betas."+str(trait)+"."+str(index)+".txt"
    
    #    (bim,fam,G) = read_plink(genPATH)
    #    fam = pd.DataFrame(fam.values)
    #    bim = pd.DataFrame(bim.values)
    G = Bed(genPATH,count_A1=False)
    fam = pd.read_csv(genPATH+".fam",header=None,sep=' ')
    bim = pd.read_csv(genPATH+".bim",header=None,sep='\t')

    phen = pd.read_csv(phenPATH,header=0,sep=' ')
    # blank,chr, snp, bp, a1, fa, fu, a2 ,x2, P, OR, blank
    gwas = pd.read_csv(gwasPATH)
    top = 50000
 
    # sort gwas into top N snps
    # excluding the sex chromosomes (and MT)
    sexchr = bim[0].astype(int).ge(23)
    best = gwas[~sexchr].sort_values(by='P',ascending=True)['SNP'][0:top]
    subsetP = bim[1].isin(best)
    subsetP = np.stack(pd.DataFrame(list(range(bim.shape[0])))[subsetP].values,axis=1)[0]

    subsetN = pd.read_csv(trainPATH,sep=' ',header=None)
    subsetN = subsetN[0].values

    # this will subset the bed matrix
    # and then actually load it into memory (.compute())
    t = time.time()
    print("Original shape of G:",flush=True)
    print(G.shape)
    subG = np.asfortranarray(G[subsetN,subsetP].read().val)
    print("Next shape of G:")
    print(subG.shape)
    elapsed = time.time() - t
    print(elapsed)
    print("Final shape of SubG:",flush=True)
    print(subG.shape)
                    
    print("Calc means")
    # calculate column means with no missing values
    # nanmean calculates mean skipping nan
    center = np.zeros(subG.shape[1])
    spread = np.zeros(subG.shape[1])
    for col in range(0,subG.shape[1]):
        center[col] = np.nanmean(subG[:,col])
        spread[col] = np.nanstd(subG[:,col])

    print("NA repl")     
    # na replacement
    missing = np.argwhere(np.isnan(subG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        subG[ind1,ind2] = center[ind2]

    print("Standardize")    
    # standardize the columns
    for col in range(0,subG.shape[1]):
        val = spread[col]
        if spread[col] == 0.0:
            val = 1.0
        subG[:,col] = (subG[:,col] - center[col])/val

    # standardize the phenotype
    y = phen.iloc[subsetN,3]
    ymu = np.mean(y)
    ysig = np.std(y)
    y = (y-ymu)/ysig

    # do the lasso
    print("Begin LASSO.",flush=True)    
    t = time.time()
    path = skl.linear_model.lasso_path(subG,y,n_alphas=nstep,eps=lamratio)
    elapsed = time.time() - t
    print("LASSO time:",flush=True)
    print(elapsed)
    
    betas = path[1]
    lamb = path[0]

    metadat = bim.iloc[subsetP,:]
    metadat = metadat.reset_index(drop=True)
    betas = pd.DataFrame(np.transpose(np.transpose(betas)*np.transpose(ysig/spread)))
    lamb = pd.DataFrame(lamb)

    out = pd.concat([metadat,pd.DataFrame(center),betas],ignore_index=True,axis=1)
    out.to_csv(r''+betaPATH,sep = ' ',index=False,header=False)
    lamb.to_csv(r''+lamPATH,sep=' ',index=False,header=False)

    return 0


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     prog='lasso',
                                     usage='%(lasso)s',
                                     description='''Runs the lasso path algo.''')

    # essential arguments
    required_named = parser.add_argument_group('Required named arguments')
    required_named.add_argument('--geno-path',
                                type=str,
                                required=True,
                                help='path to genotypes')

    required_named.add_argument('--trait',
                                type=str,
                                required=True,
                                help='name of trait')

    required_named.add_argument('--index-var',
                                type=str,
                                required=True,
                                help='index variable, 1-5')

    # file to
    required_named.add_argument('--output-directory',
                                type=str,
                                required=True,
                                help='Where all the output goes')

    args = parser.parse_args()

    runLasso(args.geno_path,args.trait,args.index_var,args.output_directory)


exit(main())

