'''
To generate phase diagram, need to run:
    1. generateparams.py
    2. jobarray.py
    3. plotPD.py
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import multiprocessing 
import numpy as np
import time
import warnings
from itertools import product
import sys
import pandas as pd
import h5py
import testphonons
from wffiles.updatedjastrow import UpdatedJastrow

def ConstructPhaseDiagram(argin,datadir):
    r_s,nconfig,seed,elec,ph,coul,growth_est,nstep,Nk_cut,tproj,arrstep,popstep,savestep,l,eta,initcond,tau = argin

    wf = UpdatedJastrow(r_s,nconfig=nconfig,diffusion=coul)
    print(wf.L)

    pos = testphonons.InitPos(wf,initcond,d=1) 

    filename = "DMC_{9}_diffusion_{10}_el{8}_ph{7}_rs_{0}_popsize_{1}_seed_{2}_N_{3}_eta_{4:.2f}_l_{5:.2f}_nstep_{6}_popstep{11}_arrstep{12}".format(r_s,nconfig,seed,Nk_cut,eta,l,nstep,ph,elec,initcond,coul,popstep,arrstep)
    print(filename)
   
    np.random.seed(seed)
    tic = time.perf_counter()
    filename = os.path.join(datadir,filename + '_tau_' + str(tau))
    h5name = filename + ".h5"
    print(h5name)
    csvname = filename + ".csv"
        
    df = testphonons.simple_dmc(
        wf,
        pos= pos,
        tau=tau,
        popstep=popstep,
        N=Nk_cut,
        nstep=nstep,
        tproj=tproj,
        l=l,
        eta=eta,
        elec=elec,
        phonon=ph,
        gth=growth_est,
        h5name = h5name,
        arrstep = arrstep,
        savestep = savestep,
        resumeh5 = '',
    )
    df.to_csv(csvname, index=False)
       
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-id', type=str,default='')
    parser.add_argument('-outdir', type=str,default='./test')
    parser.add_argument('-startnum', type=int,default=0)
    parser.add_argument('-endnum', type=int,default=2)
    parser.add_argument('-paramfile', type=str,required=True)

    args = parser.parse_args()
    ID = args.id
    savedir = args.outdir

    # Grab the arguments that are passed in
    # This is the task id and number of tasks that can be used
    # to determine which indices this process/task is assigned
    start_num = args.startnum
    end_num = args.endnum

    df = pd.read_csv(args.paramfile)
    #quantities=['D/g','th_c/th0','oscratio','D','g','theta_c','theta0','B','mu','A','K0','initcond','initphi','inittheta','L','resolution','tmax','dt','wtfrac']    
    #if len(ID) == 0: ID = 'data'

    if start_num > len(df.index): sys.exit('too many jobs initialized!!')
    if end_num > len(df.index): end_num = len(df.index)

    #csvname = os.path.join(savedir,'%s_tasks%d-%d.txt' %(ID,start_num,end_num))
    #print(csvname)

    tic = time.perf_counter()
    for j in range(start_num,end_num):
        row = df.iloc[[j]] #assume taskID starts at j=0
        argin = tuple(row.values[0])
        result = ConstructPhaseDiagram(argin,savedir)
    #with open(csvname,'w') as txtfile:
    #    txtfile.write(','.join(quantities) + '\n')
    #        txtfile.write(','.join(map(str,result)))
    #        txtfile.write('\n')
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    
