import os
import numpy as np
import time
from itertools import product
import sys
import pandas as pd

def GenerateParams(infodict,savedir,ID='',xvar='eta',xb=[0,.1,5],yvar='l',yb=[0,50,10],opt='lin'):
    el = infodict['elec']
    ph = infodict['ph']
    coul = infodict['coul']
    if opt == 'log':
        xs = np.logspace(xb[0],xb[1],int(xb[2]))
        ys = np.logspace(yb[0],yb[1],int(yb[2]))
    else:
        opt = 'lin'
        xs = np.linspace(xb[0],xb[1],int(xb[2]))
        ys = np.linspace(yb[0],yb[1],int(yb[2]))
    fname = '%s_%s_params_%s_el%d_ph%d_coul%d.csv' %(xvar,yvar,opt,el,ph,coul)
    if len(ID) > 0:
        fname = os.path.join(savedir,'%s_%s' %(ID,fname))
    else:
        fname = os.path.join(savedir,fname)
    infodict = {k:[v] for k,v in infodict.items()}
    df = pd.DataFrame.from_dict(infodict)
    df = df.loc[df.index.repeat(len(xs)*len(ys))].reset_index(drop=True)
    job_args = [(x,y) for x,y in product(xs,ys)]
    df[xvar] = [item[0] for item in job_args]
    df[yvar] = [item[1] for item in job_args]
    print(fname)
    print(df.head())
    df.to_csv(fname,index=False)

if __name__ == '__main__':
    from argparse import ArgumentParser

    # sample command line command
    # python generateparams.py --rs 30 --outdir test
    # python generateparams.py --rs 30 --outdir test --xb 0 0.1 2 --outdir test --Nstep 50
    # python generateparams.py --rs 30 --outdir data_ph --xb 0 0.08 10 --Nstep 50000 --elec 1 --ph 1
    # python generateparams.py --rs 30 --outdir data_ph --xb 0 0.08 10 --Nstep 50000 --elec 1 --ph 0 
    # python generateparams.py --rs 30 --outdir data_jell --xb 0 0.08 10 --Nstep 20000 --elec 1 --ph 0
    # python generateparams.py --rs 30 --outdir data_ph_30k --xb 0 0.08 10 --Nstep 30000 --elec 1 --ph 1
    # python generateparams.py --rs 30 --outdir data_pol_30k_try2 --xb 0 0.08 10 --Nstep 30000 --elec 1 --ph 1 --coul 0

    parser = ArgumentParser()
    # set variable params
    parser.add_argument('--xvar', type=str, default='eta')
    parser.add_argument('--yvar', type=str, default='l') 
    parser.add_argument('--xb', nargs='+', default=[0,0.1,5]) #min, max, Nbins
    parser.add_argument('--yb', nargs='+', default=[1E-2,25,10]) #min, max, Nbins
    parser.add_argument('--opt', type=str, default='lin') #generate params on linear or log scale

    # set constant params
    parser.add_argument('--rs', type=int, default=30)
    parser.add_argument('--nconf', type=int, default=512)
    parser.add_argument('--seed',type=int,default=0) #random seed
    parser.add_argument('--elec', type=int, default=1) #on/off switch for electrons
    parser.add_argument('--ph', type=int, default=1) #on/off switch for phonons
    parser.add_argument('--Ncut',type=int,default=15) # defines spherical momentum cutoff k_cut = 2pi*N/L
    parser.add_argument('--Nstep',type=int, default=80000) # number of steps in simulation
    parser.add_argument('--tproj',type=int,default=128) # projection time = tau * nsteps
    parser.add_argument('--arrstep',type=int,default=50) # how frequently to save phonon info + interparticle distances
    parser.add_argument('--popstep',type=int,default=200) # how frequently to branch --> comb through existing walkers
    parser.add_argument('--savestep',type=int,default=5) # how frequently to save energy information
    parser.add_argument('--l',type=np.float64,default=5) #plays the role of the electron coupling strength U: l = U/2 
    parser.add_argument('--eta',type=np.float64,default=0.001) 
    parser.add_argument('--gth',type=int,default=0) #on/off switch for growth estimator
    parser.add_argument('--outdir',type=str,default='test') 
    parser.add_argument('--id',type=str,default='data') 
    parser.add_argument('--init',type=str,default='bind') 
    parser.add_argument('--tau',type=np.float64, default = None) 
    parser.add_argument('--coul',type=int,default=1) # on/off switch for diffusion (jellium, no Coulomb). Allowed to have diffusion (i.e. 0 Coulomb) + phonons to try and define a baseline for binding energy calc

    args = parser.parse_args()
    xvar = args.xvar
    yvar = args.yvar
    opt = args.opt
    xb = [float(x) for x in args.xb]
    yb = [float(y) for y in args.yb]
    print(xvar,yvar,xb,yb)
    ID = args.id
    r_s = args.rs  # inter-electron spacing, controls density
    tau = args.tau
    if tau is None: tau = r_s/40 
    print(tau)
    Nstep=args.Nstep
    popstep = args.popstep
    arrstep = args.arrstep
    savestep = args.savestep
    nconfig = args.nconf #default is 5000
    seed = args.seed
    elec_bool = args.elec > 0
    gth_bool = args.gth > 0
    coul = args.coul > 0
    if Nstep is None:
        tproj = args.tproj #projection time = tau * nsteps
        Nstep = int(tproj/tau)
    else:
    # if num of sim steps is specified, this determines the sim projection time
        if Nstep % arrstep != 0:
            Nstep = int(arrstep*np.ceil(Nstep/arrstep))
            print('big arrays (position, phonon amplitudes) are only saved every %d steps, rounding to nearest save point: Nstep = %d' %(arrstep, Nstep))
        tproj = Nstep*tau

    ph_bool = args.ph > 0
    Ncut = args.Ncut
    init = args.init
    l = args.l
    eta = args.eta
    savedir = args.outdir

    if os.path.isdir(savedir) == 0:
        print('Making data directory...')
        os.mkdir(savedir)

    infodict = {
        'r_s':r_s,
        'nconfig':nconfig,
        'seed': seed,
        'elec': elec_bool,
        'ph': ph_bool,
        'coul': coul, #diffusion = 1 / coul = 0 --> no Coulomb
        'growth_est': gth_bool,
        'nstep': Nstep,
        'Nk_cut': Ncut,
        'tproj': tproj,
        'arrstep': arrstep,
        'popstep': popstep,
        'savestep': savestep,
        'l': l,
        'eta': eta,
        'initcond': init,
        'tau': tau,
    }
    GenerateParams(infodict,savedir,ID,xvar,xb,yvar,yb,opt)
