from __future__ import print_function

import os
import stat
import json
import csv
import itertools
import logging as log
from argparse import ArgumentParser
from datetime import datetime
import neuron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from  toolbox.Util_H5io3 import write3_data_hdf5
import ruamel.yaml as yaml
import sys
import ast

#import yaml as yaml
# import yaml as yaml

from stimulus import stims, add_stims
import models

templates_dir = '/global/cfs/cdirs/m2043/hoc_templates/hoc_templates'

try:
    from mpi4py import MPI
    mpi = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_tasks = comm.Get_size()
except:
    mpi = False
    comm = None
    rank = 0
    n_tasks = 1
    
from neuron import h, gui

VOLTS_SCALE = 1

MODELS_BY_NAME = models.MODELS_BY_NAME
stim_mul_range = [1,0.3]
stim_offset_range = [0,0.1]


def _rangeify_linear(data, _range):
    return data * (_range[1] - _range[0]) + _range[0]

def _rangeify_exponential(data, _range):
    if tuple(_range) == (0, 0):
        return 0

    # return np.float_power(_range[1], data) * np.float_power(_range[0], 1-data)

    return np.exp(
        data * (np.log(_range[1]) - np.log(_range[0])) + np.log(_range[0])
    )

def get_model(model, log, m_type=None, e_type=None, cell_i=0, init_cell=False,*params):
    if model == 'BBP':
        if m_type is None or e_type is None:
            raise ValueError('Must specify --m-type and --e-type when using BBP')
        
        if e_type == 'cADpyr':
            model = models.BBPExcV2(m_type, e_type, cell_i, *params, log=log)
        else:
            model = models.BBPInh(m_type, e_type, cell_i, *params, log=log)
            
        # if(init_cell):
        #     model.create_cell_multi() # THE Change has been overwritten :|
        model.create_cell()
        return model
    elif model =='newBBP':
        model = models.newExcBBP(m_type, e_type, cell_i, *params, log=log)
        model.create_cell()
    elif model =='newM1':
        mod_path="/global/homes/k/ktub1999/mainDL4/DL4neurons2/newM1hocs"
        model = models.NewM1_TTPC_NA_HH(mod_path,m_type, e_type, cell_i, *params, log=log)
        model.create_cell()
        return model
    elif model == 'M1_TTPC_NA_HH':
        mod_path="/global/homes/k/ktub1999/mainDL4/DL4neurons2/Neuron_Model_HH"
        model = models.M1_TTPC_NA_HH(mod_path,m_type, e_type, cell_i, *params)
        model.create_cell()
        return model
    else:
        return MODELS_BY_NAME[model](*params, log=log)
    
        
        

        

def clean_params(args, model):
    """convert to float, use defaults where requested

    Return: args.params, but with 'rand' replaced by float('inf') and
    'def' replaced by the default value of that param (from the
    model). Also return a list of booleans indicating whether 'def'
    was passed (this is needed to special case its value in the output
    to 1.1
    """
    defaults = model.DEFAULT_PARAMS
    if args.params:
        assert len(args.params) == len(defaults)
        defs = [param == 'def' for param in args.params]
        return [float(x if x != 'rand' else 'inf') if x != 'def' else default
#                 for (x, default) in zip(args.params, defaults)]
                for (x, default) in zip(args.params, defaults)], [False if param != 'def' else True for param in args.params]
    else:
        return [float('inf')] * len(defaults), [False] * len(defaults)


def report_random_params(args, params, model):
    param_names = model.PARAM_NAMES
    for param, name in zip(params, param_names):
        if param == float('inf'):
            log.debug("Using random values for '{}'".format(name))

def get_included(args,model):
    cell_count=0
    # if(args.cell_count):
    #     cell_count=args.cell_count
    # if(args.model == "BBP"):
    #     base_params = pd.read_csv("./sensitivity_analysis/NewBase2/MeanParams"+str(int(cell_count))+".csv")
    # elif(args.model =="M1_TTPC_NA_HH"):
    #     base_params = pd.read_csv("./sensitivity_analysis/NewBase2/M1params"+str(int(cell_count))+".csv")
    # elif(args.model =="newBBP"):
        # base_params = pd.read_csv("./sensitivity_analysis/NewBase2/NewBBPparams"+str(int(cell_count))+".csv")
    # elif(args.model =="newM1"):
        # base_params = pd.read_csv("./sensitivity_analysis/NewBase2/newM1params"+str(int(cell_count))+".csv")
    params=model.PARAM_NAMES
    # params=list(base_params["Parameters"])
    included_index=[]
    for i in range(len(params)):
        if(params[i] not in args.exclude):
            included_index.append(i)
    return included_index


def get_ranges(args,model):
    return model.UNIT_RANGES


def get_ranges_old(args):
    cell_count=0
    if(args.cell_count):
        cell_count=args.cell_count
    res=[]
    default_params_wide= pd.read_csv("/pscratch/sd/k/ktub1999/main/DL4neurons2/sensitivity_analysis/NewBase2/NewBase"+str(int(cell_count))+".csv")
    default_params_nrow= pd.read_csv("/pscratch/sd/k/ktub1999/main/DL4neurons2/sensitivity_analysis/NewBase2/MeanParams"+str(int(cell_count))+".csv")
    # default_params= pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase"+str(int(cell_count))+".csv")
    params=list(default_params_wide["Parameters"])
    # params = model.PARAM_NAMES
    # model_default_params = model.DEFAULT_PARAMS
    for i in range(len(params)):
        param=params[i]
        # Base = model_default_params[i]
        if(param in args.wide):
            Base = default_params_wide["Values"].iloc[i]
        else:
            Base = default_params_nrow["Values"].iloc[i]
        if(param=="e_pas_all"):
            if(param in args.wide):
                a_value=20
            else:
                a_value=10
            
        elif(param=="cm_somatic" or param=="cm_axonal"):
            if(param in args.wide):
                a_value=1.45
            else:
                a_value=0.875
                Base=1.125
        else:
            if(param in args.wide):
            # a_value=1.0 #NRow
               a_value=1.5  
            else:
               a_value=1.0 
        res.append([Base,a_value])
    return res


def get_random_params(args,model,n=1):
    ndim = len(model.DEFAULT_PARAMS)
    rand = np.random.uniform(-1,1,(n,ndim))
    set_unit_params=False
    count_cell=0
    if(args.cell_count):
        count_cell=args.cell_count
    phy_res=[]

    if(args.model == "BBP"):
        base_params = pd.read_csv("./sensitivity_analysis/NewBase2/MeanParams"+str(int(count_cell))+".csv")
    elif(args.model =="M1_TTPC_NA_HH"):
        #SAVE as CSV xander 4
        base_params = pd.read_csv("./sensitivity_analysis/NewBase2/M1params"+str(int(count_cell))+".csv")
    elif(args.model=="newBBP"):
        base_params = pd.read_csv("./sensitivity_analysis/NewBase2/NewBBPparams"+str(int(count_cell))+".csv")
    elif(args.model =="newM1"):
        base_params = pd.read_csv("./sensitivity_analysis/NewBase2/newM1params"+str(int(count_cell))+".csv")
    # base_params = pd.read_csv("./sensitivity_analysis/NewBase2/MeanParams"+str(int(count_cell))+".csv")
    
    #for each sample

    
    for i in range(n):
        curr_phy_res=[]
        #for each parameter
        if(args.default_base):
            base_dict={}
            base_dict['Parameters']=list(model.PARAM_NAMES)
            base_dict['Values']=list(model.DEFAULT_PARAMS)
            base_params=pd.DataFrame.from_dict(base_dict)
        for j in range(ndim):
            
            u = rand[i][j]
            [uLb, uUb] = model.UNIT_RANGES[j]
            pram = base_params['Parameters'].iloc[j]

                
            if(pram=='e_pas_all' or pram=='cm_somatic' or pram =='cm_axonal' or pram=='cm_all' or pram =='sh_na12' or pram=='sh_na16'):


                b_value = (uLb+uUb)/2
                a_value = (uUb-uLb)/2
                if(pram in args.exclude or args.def_params):
                    curr_phy_res.append(base_params['Values'].iloc[j])
                    rand[i][j]=0
                    continue
                ## ADD a check if A_value  is 0
                P = b_value + a_value * u
            else:
                new_base=base_params['Values'].iloc[j]*10**((uLb+uUb)/2)
                b_value = 0
                a_value = (uUb - uLb)/2
                if(pram in args.exclude or args.def_params):
                    curr_phy_res.append(base_params['Values'].iloc[j])
                    rand[i][j]=0
                    continue
                P = new_base*10**(b_value+a_value*u)
            curr_phy_res.append(P)
            if(not set_unit_params):
                model.UNIT_PARAMS[j]=[b_value,a_value]
            #For meta Data
        set_unit_params=True
        phy_res.append(curr_phy_res)
       
    return phy_res,rand


def get_random_params_old(args,model,n=1):
    ndim = len(model.DEFAULT_PARAMS)
    rand = np.random.uniform(-1,1,(n,ndim))
    if(args.def_params):
        rand = np.zeros([n,ndim],float)
        # np.random.uniform(-1,1,(n,ndim))
    phy_res=[]
    count_cell=0
    if(args.cell_count):
        count_cell=args.cell_count
    
    default_params_wide= pd.read_csv("./sensitivity_analysis/NewBase2/NewBase"+str(int(count_cell))+".csv")
    default_params_nrow= pd.read_csv("./sensitivity_analysis/NewBase2/MeanParams"+str(int(count_cell))+".csv")
    # default_params= pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase"+str(int(count_cell))+".csv")
    # model_default_params=model.DEFAULT_PARAMS
    # model_param_names = model.PARAM_NAMES
    for i in range(n):
        curr_phy_res=[]
        for j in range(ndim):
            u = rand[i][j]
            # B = model_default_params[j]
            # pram = model_param_names[j]

            
            pram = default_params_wide["Parameters"].iloc[j]
            if(pram in args.wide):
                B=default_params_wide["Values"].iloc[j]
            else:
                B=default_params_nrow["Values"].iloc[j]

            if(args.def_params):
                curr_phy_res.append(B)
            elif(pram in args.exclude):
                curr_phy_res.append(B)
                rand[i][j]=0
            elif(pram=="e_pas_all"):
                #P = Base*(A+B*u) because Linear Param, Ranges = -125, -25
                # a_value=10 For Narrow
                if(pram in args.wide):
                    a_value=20
                else:
                    a_value=10
                b_value=(-2/3)
                curr_phy_res.append(B+a_value*u)
                # curr_phy_res.append(B*(a_value+b_value*u))
            elif(pram=="cm_somatic" or pram=="cm_axonal"):
                # a_value = 0.875 For Narrow
                # B=1.125
                if(pram in args.wide):
                    a_value=1.45 # For Wide
                else:
                    a_value = 0.875
                    B=1.125
                # b_value=1.45
                curr_phy_res.append(B+a_value*u)
                # curr_phy_res.append(B*(a_value+b_value*u))
            else:
                # a_value=1.0 #For Narrow
                # a_value=1.5 # For Wide
                if(pram in args.wide):
                    a_value=1.5
                else:
                    a_value=1.0
                b_value=1.5                
                curr_phy_res.append(B*np.exp((u*a_value)*np.log(10)))
        phy_res.append(curr_phy_res)
    return phy_res,rand

def get_random_params2(args,model,n=1):
    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    ranges = model.PARAM_RANGES
    Default_paramsdf =pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv") 
    Default_params = Default_paramsdf["Values"].tolist()
    params = list(Default_paramsdf["Parameters"])
    ranges=[]
    # Bounds = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/Bounds.csv")
    pindex=0
    for params_single in Default_params:
        # bounds = Bounds.loc[Bounds['Parameter']==params[pindex]]
        # LB = bounds['LB']
        # UB = bounds['UB']
        LB =-1
        UB=2
        # TO CALCULATE THE VLAUE :Value = Base* e^(x*a*ln10)
        #a =(Upper Bound - LowerBound)/2
        a_value = 0
        b_value=1.5
        if(params[pindex]=="e_pas_all"):#LINEAR parameter
            a_value =1
            ranges.append((-25,-125))
        elif(params[pindex]=="cm_somatic" or params[pindex]=="cm_axonal"):
            ranges.append((0.1,3))
        # elif(params[pindex]=="gNaTa_tbar_NaTa_t_axonal"):
        #     ranges.append((0.3402369775,100))
        else:
            # a_value = float(a_value.iloc[0])
            ranges.append((params_single*np.exp((int(-1)*b_value+a_value)*np.log(10)),params_single*np.exp((int(+1)*b_value+a_value)*np.log(10))))
            
        pindex+=1
    ranges = tuple(ranges)
    # ranges = model.PARAM_RANGES
    ndim = len(ranges)
    rand = np.random.rand(n, ndim)
    phys_rand = np.zeros(shape=rand.shape)
    params, defaulteds = clean_params(args, model)
    rangeify = _rangeify_linear if args.linear else _rangeify_exponential
    report_random_params(args, params, model)
    for i, (_range, param, varied, defaulted) in enumerate(zip(ranges, params, model.get_varied_params(), defaulteds)):
        # Default params swapped in by clean_params()

        # If that parameter is not variable (either not present, or equals zero)
        if not varied:
            rand[:, i] = 0.5 # so it gets set to 0 when writen to disk (see upar in save_h5())

        # If that parameter is in general allowed to vary (is present
        # and nonzero in the BBP model), but we asked for it to be
        # fixed by passing 'def' on the command line:
        if varied and defaulted:
            rand[:, i] = (-1.1 + 1) / 2.0 # so it gets set to -1.1

        # Put either the fixed or random params into phys_rand
        if param == float('inf'):
            #print(f'{i}-{model.PARAM_NAMES[i]}')
            if i in args.linear_params_inds:
                #print(f'{i}-{model.PARAM_NAMES[i]} is linear')
                phys_rand[:, i] = _rangeify_linear(rand[:, i], _range)
            else:
                phys_rand[:, i] = rangeify(rand[:, i], _range)
        else:
            phys_rand[:, i] = np.array([param] * n)
    phys_rand[np.isnan(phys_rand)] = 9e9
        
            
        
    return phys_rand, rand
def get_stim(args,idx):
    
    stim_fn = f'./{args.stim_file[idx]}'
    stim = np.genfromtxt(stim_fn, dtype=np.float32)
    u_offset=0
    u_mul=0
    
    if (args.stim_multiplier!=None):
        stim_mul = args.stim_multiplier
        # print("Taking Sim from ARGS for MUL",args.stim_multiplier)    
    else:
        u_mul=np.random.uniform(-1,1,1)[0]
        b_value=stim_mul_range[0]
        a_value=stim_mul_range[1]
        stim_mul=b_value+a_value*u_mul
        # stim_mul = np.random.uniform(stim_mul_range[0],stim_mul_range[1])
    if (args.stim_dc_offset!=None):
        stim_offset = args.stim_dc_offset
        # print("Taking Sim from ARGS for OFFSET",args.stim_dc_offset)
    else:
        # print(args.stim_dc_offset,"Offset")
        
        u_offset=np.random.uniform(-1,1,1)[0]
        b_value=stim_offset_range[0]
        a_value=stim_offset_range[1]
        stim_offset=b_value+a_value*u_offset
        # stim_offset = np.random.uniform(stim_offset_range[0],stim_offset_range[1])
    stim = stim*stim_mul+stim_offset
    return stim,stim_mul,stim_offset,u_mul,u_offset
        
def get_mpi_idx(args, nsamples):
    if args.trivial_parallel:
        return 0, args.num
    elif args.node_parallel:
        params_per_task = (nsamples // 64) + 1
        task_i = rank % 64
    else:
        params_per_task = (nsamples // n_tasks) + 1
        task_i = rank
    start = params_per_task * task_i
    stop = min(params_per_task * (task_i + 1), nsamples)
    if args.num:
        stop = min(stop, args.num)
    log.debug("There are {} ranks, so each rank gets {} param sets".format(n_tasks, params_per_task))
    log.debug("This rank is processing param sets {} through {}".format(start, stop))

    return start, stop





def create_h5(args, nsamples,model):
    # print(f'in crate h5 we have {nsamples} nsamples ')
    #log.info("Creating h5 file {}".format(args.outfile))
    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    with h5py.File(args.outfile, 'w') as f:
        # write params
        ndim = len(model.PARAM_NAMES)
        f.create_dataset('phys_par', shape=(nsamples, ndim), dtype=np.float32)
        f.create_dataset('norm_par', shape=(nsamples, ndim), dtype=np.float32)
        #f.create_dataset('varParL', data=np.string_(model.PARAM_NAMES))
        #if args.model == 'BBP':
            #f.create_dataset('probeName', data=np.string_(model.get_probe_names())) ## MISSING CERTAIN PROBES



        # create stim, qa, and voltage datasets
        stim = get_stim(args,0)
        ntimepts = len(stim)
        if args.model == 'BBP':
            f.create_dataset('voltages', shape=(nsamples, ntimepts, model._n_rec_pts(),len(args.stim_file)), dtype=np.int16)
        else:
            f.create_dataset('voltages', shape=(nsamples, ntimepts), dtype=np.int16)
        f.create_dataset('stim_par', shape=(nsamples,2,len(args.stim_file)), dtype=np.int32)
        #f.create_dataset('stim', data=stim)
        #f.create_dataset('binQA', shape=(nsamples,), dtype=np.int32)
        
    #log.info("Done.")


def _normalize(args, data,model ,minmax=1):
    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    nsamples = data.shape[0]
    mins = np.array([tup[0] for tup in model.PARAM_RANGES])
    mins = np.tile(mins, (nsamples, 1)) # stacked to same shape as input
    ranges = np.array([_max - _min for (_min, _max) in model.PARAM_RANGES])
    ranges = np.tile(ranges, (nsamples, 1))

    return 2*minmax * ( (data - mins)/ranges ) - minmax

    
def save_h5(args, buf_vs, buf_stims, params, start, stop,force_serial=False, upar=None,stim_params = [],stim=None):
    #log.info("saving into h5 file {}".format(args.outfile))
    #print(f'printing soma? {model.get_probe_names()}')
    if (comm and n_tasks > 1) and not force_serial:
        log.debug("using parallel")
        kwargs = {'driver': 'mpio', 'comm': comm}
    else:
        log.debug("using serial")
        kwargs = {}

    model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    if not os.path.exists(args.outfile):
        create_h5(args, stop-start,model)
    
    with h5py.File(args.outfile, 'a', **kwargs) as f:
        log.debug("opened h5")
        log.debug(str(params))
        #f['voltages'][start:stop, ...] = (buf*VOLTS_SCALE).clip(-32767,32767).astype(np.int16)
        #f['binQA'][start:stop] = qa
        f['stim_par'][start:stop] = stim_params
        if not args.blind:
            f['phys_par'][start:stop, :] = params
            f['norm_par'][start:stop, :] = (upar*2 - 1) if upar is not None else _normalize(args, params,model)
        #log.info("saved h5")
    #log.info("closed h5")
    os.chmod(args.outfile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

def plot(args, data, stim):
    if args.plot is not None:
        ntimepts = len(stim)
        t_axis = np.linspace(0, ntimepts*h.dt, ntimepts)
        plt.figure(figsize=(10, 5))
        plt.xlabel('Time (ms)')
        if args.plot == [] or 'v' in args.plot:
            plt.plot(t_axis, data['v'][:ntimepts], label='V_m')
        if args.plot == [] or 'v_dend' in args.plot:
            plt.plot(t_axis, data['v_dend'][:ntimepts], label='v_dend')
        if args.plot == [] or 'stim' in args.plot:
            plt.plot(t_axis, stim[:ntimepts], label='stim')
        if args.plot == [] or 'ina' in args.plot:
            plt.plot(t_axis, data['ina'][:ntimepts] * 100, label='i_na*100')
        if args.plot == [] or 'ik' in args.plot:
            plt.plot(t_axis, data['ik'][:ntimepts] * 100, label='i_k*100')
        if args.plot == [] or 'ica' in args.plot:
            plt.plot(t_axis, data['ica'][:ntimepts] * 100, label='i_ca*100')
        if args.plot == [] or 'i_cap' in args.plot:
            plt.plot(t_axis, data['i_cap'][:ntimepts] * 100, label='i_cap*100')
        if args.plot == [] or 'i_leak' in args.plot:
            plt.plot(t_axis, data['i_leak'][:ntimepts] * 100, label='i_leak*100')

        if not args.no_legend:
            plt.legend()

        plt.show()


def lock_params(args, paramsets,model):
    # DEPRECATED. Create/use Latched model sublcasses (see HHBallStick7ParamLatched)
    assert len(args.locked_params) % 2 == 0

    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    paramnames = model.PARAM_NAMES
    nsets = len(args.locked_params)//2
    
    targets = [args.locked_params[i*2] for i in range(nsets)]
    sources = [args.locked_params[i*2+1] for i in range(nsets)]

    for source, target in zip(sources, targets):
        source_i = paramnames.index(source)
        target_i = paramnames.index(target)
        paramsets[:, target_i] = paramsets[:, source_i]

def template_present(cellName,i_cell=0):
    
    i_cell = str(int(i_cell)+1)
    # template_cell = templates_dir+"/"+cellName
    cell_clones =  os.listdir(templates_dir)
    cell_clones =[x for x in cell_clones if cellName in x]
    cell_is=[]
    # print(i_cell)
    for x in cell_clones:
        cell_is.append(x.split('_')[-1])
    if(i_cell not in cell_is):
        print("Template Doesnt Exist{}, Skipping".format(cellName))
        # print(template_cell)
        return False
    return True

def get_init_volts(args,model,simTime,dt):
    # init_counter=args.init_counter
    # fig = plt.figure()
    stim = np.zeros(int(simTime/dt))
    data = model.simulate(stim, dt)
    Data = data[list(data.keys())[0]]
    # plt.plot(Data)
    # plt.savefig("init"+str(init_counter)+".png")
    # init_counter+=1
    v_init= np.median(Data[150:])
    # f = open("Vinits.txt", "a")
    # f.write(args.m_type+args.e_type+str(model.cell_i)+":"+str(v_init)+"\n")
    # f.close()
    return v_init

    


def main(args):
    
    
    tot_time = datetime.now()
    # print(args)
    # args.wide = set(args.wide)
    if args.trivial_parallel and args.outfile and '{NODEID}' in args.outfile:
        args.outfile = args.outfile.replace('{NODEID}', os.environ['SLURM_PROCID'])

    if (not args.outfile) and (not args.force) and (args.plot is None) and (not args.create_params):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")
    if(args.m_type and args.e_type):
        if(not args.cell_i):
            args.cell_i=0
        bbp_name=args.m_type+"_"+args.e_type+"_"+str(args.cell_i)
        
    if args.cori_csv:
        # if(args.generate_all):
        #     cori_i = int(os.environ.get('SLURM_PROCID')) /5
        #     args.cell_i = int(os.environ.get('SLURM_PROCID')) %5
        #     args.cell_i=str(args.cell_i)
        # else :
        cori_i = args.cori_start + int(os.environ.get('SLURM_PROCID')) % (args.cori_end - args.cori_start)
        if cori_i == 9:
                return
        with open(args.cori_csv, 'r') as infile:
            allcells = csv.reader(infile, delimiter=',')
            for i, row in enumerate(allcells):
                if i == cori_i:
                    log.debug(row)
                    bbp_name = row[0]
                    args.m_type = row[1]
                    args.m_type = str(args.m_type)
                    args.e_type = row[2]
                    args.e_type = str(args.e_type)
                    log.info("from rank {} running cell {}".format(rank, bbp_name))
                    print("from rank {} running cell {}".format(rank, bbp_name))
                    
                    break

         
        # Get param string for holding some params fixed
        #paramuse = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1] \
        paramuse = [1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1] if args.e_type == 'cADpyr' else [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        args.params = [('inf' if use else 'def') for use in paramuse]

        if args.e_type in ('bIR', 'bAC'):
            paramuse[20] = 0
            log.info('Not varying negative parameters for e-type {}'.format(args.e_type))
    # print(type(args.m_type), type(args.e_type))
    cellName = str(args.m_type)+"_"+str(args.e_type)
    # template_cell = templates_dir+"/"+cellName
    if(not template_present(cellName,args.cell_i)):
        return

    if args.outfile and '{BBP_NAME}' in args.outfile:
        args.outfile = args.outfile.replace('{BBP_NAME}', bbp_name)
        #args.metadata_file = args.metadata_file.replace('{BBP_NAME}', bbp_name)
    f = open(os.devnull, 'w')
    # sys.stdout = f
    model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i,args.init_cell)
    sys.stdout = sys.__stdout__

    if args.create:
        if not args.num:
            raise ValueError("Must pass --num when creating h5 file")
        create_h5(args, args.num,model)
        exit()
    rand =[]
    if args.create_params:
        paramsets,rand= get_random_params(args,model,n=args.num)
        np.savetxt(args.param_file,paramsets)
        # exit()

    if args.blind and not args.param_file:
        raise ValueError("Must pass --param-file with --blind")

    
    if args.metadata_only:
        write_metadata(args, model)
        exit()
    
    if(args.unit_param_upper!=None and args.unit_param_lower!=None):
        # print(len(args.unit_param_upper),len(args.unit_param_lower),"SIZES")
        for param_no in range(len(args.unit_param_upper)):
            #param_no has ['param_name','uLb','uUb']    
            model.UNIT_RANGES[param_no]=[float(args.unit_param_lower[param_no]),float(args.unit_param_upper[param_no])]
            # print(model.UNIT_RANGES[param_no])
    
    if(args.unit_params_csv!=None):
        unit_params_lb = pd.read_csv(args.unit_params_csv)['LB']
        unit_params_ub = pd.read_csv(args.unit_params_csv)['UB']
        assert len(model.UNIT_RANGES)==len(unit_params_ub),"Unit Prams csv length is wrong"
        for param_no in range(len(unit_params_ub)):
            #param_no has ['param_name','uLb','uUb']    
            model.UNIT_RANGES[param_no]=[float(unit_params_lb[param_no]),float(unit_params_ub[param_no])]
            # print(model.UNIT_RANGES[param_no])



    if args.param_file:
        all_paramsets = np.genfromtxt(args.param_file, dtype=np.float32)
        upar = None # TODO: save or generate unnormalized params when using --param-file
        start, stop = get_mpi_idx(args, len(all_paramsets))
        # start, stop = 0, 1
        # print("Reading from param_file")
        print(start,stop)
        if args.num and start > args.num:
            return
        if(args.num==1 or all_paramsets.shape[0]==1):
            paramsets=[]
            paramsets.append(all_paramsets)
        else:
            paramsets = all_paramsets[start:stop, :]
        paramsets = np.atleast_2d(paramsets)
        print("Shape of parameters:",paramsets.shape)
        # print("Param Size",paramsets.size)
        # print("Reading from param_file")
        # print(paramsets.shape)

    elif args.num:
        start, stop = get_mpi_idx(args, args.num)
        paramsets, rand = get_random_params(args, model,n=stop-start)
        paramsets = np.atleast_2d(paramsets)
    elif args.params not in (None, [None]):
        #paramsets = np.atleast_2d(np.array(args.params))
        #NEED TO CHANGE THIS
        paramsets = np.atleast_2d(model.DEFAULT_PARAMS)
        upar = None
        start, stop = 0, 1
    
    else:
        log.info("Cell parameters not specified, running with default parameters")
        paramsets = np.atleast_2d(model.DEFAULT_PARAMS)
        upar = None
        start, stop = 0, 1
    stimL = []
    # MAIN LOOP   
    lock_params(args, paramsets,model)
    stim,stim_mul,stim_offset,u_mul,u_offset = get_stim(args,0)#only for gettign the length to create the buffer
    #Skipping the first 1000 zeros in stim.
    buf_vs = np.zeros(shape=(stop-start, len(stim[1000:]), model._n_rec_pts(),len(args.stim_file)), dtype=np.float32)
    buf_stims = np.zeros(shape=(stop-start, 2,len(args.stim_file)), dtype=np.float32)
    buf_stims_unit = np.zeros(shape=(stop-start, 2,len(args.stim_file)), dtype=np.float32)
    # print('dd',type(paramsets),paramsets.shape)
    # h.hoc_stdout("Temp"+str(os.environ['SLURM_PROCID']))#Changing the output to temp file
    start_stim_time = datetime.now()
    print("Time to generate Data",(start_stim_time-tot_time).total_seconds())
    f = open(os.devnull, 'w')
    # sys.stdout = f
    model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i,args.init_cell)
    model.set_attachments(stim,len(stim),args.dt)
    counter_params =0
    


    # args.init_counter=0
    for iSamp, params in enumerate(paramsets):
        # print(type(os.environ['SLURM_PROCID']),"ProcID")
        if(os.environ['SLURM_PROCID']=="0" and iSamp%100==0):
            sys.stdout = sys.__stdout__
            # h.hoc_stdout()
            print("Executing for ",iSamp)
            # h.hoc_stdout("Temp")
            # f = open(os.devnull, 'w')
            # sys.stdout = f

        
            if args.print_every and iSamp % args.print_every == 0:
                log.info("Processed {} samples".format(iSamp))
        log.debug("About to run with params = {}".format(params))
        
        model._set_self_params(*params)
        model.init_parameters()
        
        v_init = get_init_volts(args,model,500,2)
        # args.init_counter+=1
        # v_init=-74
        #print(f'printing buf {buf}, printing shape{buf.shape}')
        for stim_idx in range(len(args.stim_file)):
            stim,stim_mul,stim_offset,u_mul,u_offset = get_stim(args,stim_idx)
            #stimL.append(stim)
            buf_stims[iSamp,:,stim_idx]=np.array([stim_mul,stim_offset])
            buf_stims_unit[iSamp,:,stim_idx]=np.array([u_mul,u_offset])
            #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)   
            #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)   
            # print("Counter Param",counter_params)
            counter_params+=1 
            
            data = model.simulate(stim, args.dt,v_init)
            Data = data[list(data.keys())[0]]
            # plt.plot(Data)
            # print('aaa',buf_vs[0,:,:,stim_idx].shape,len(data.values()))
            for iProb,k in enumerate(data):
                wave=data[k][1000:-1] ## IGNORING the first 1000 0 values to reduce the size of the HDF5 files.
                # print('bbb',iProb,k,wave.shape)
                buf_vs[iSamp,:,iProb,stim_idx]=wave#change here!!!!
        curr_time = datetime.now()
        min28=28*60
        min10=60*60*11
        if((curr_time-tot_time).total_seconds()>=min10):
            sys.stdout = sys.__stdout__
            print("TIMELIMIT,BREAKING after",iSamp)
            # sys.stdout = open(os.devnull, 'w')
            break
    # plt.savefig("/global/homes/k/ktub1999/mainDL4/DL4neurons2/NewBasePlots/TESTING.png")
    # plot(args, data, stim)
    h.hoc_stdout()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    sys.stdout = sys.__stdout__
    print("COMPLETED ALL SIMULATIONS",dt_string,flush=True)
    log.info("COMPLETED ALL SIMULATIONS {}".format(dt_string))
    
    

    # Save to disk
    if args.outfile:
        if(len(rand)==0):
            myUpar= _normalize(args, paramsets,model).astype(np.float32)
            myUpar=myUpar*2 -1
        buf_vs=(buf_vs*VOLTS_SCALE).clip(-32767,32767).astype(np.float16)
        if(np.isnan(buf_vs).any()):
            print("Assertion Error at END")
            log.info("Assertion Error at END {}".format(dt_string))
        # assert not np.isnan(buf_vs).any(),np.count_nonzero(np.isnan(buf_vs))
        assert not np.isnan(buf_stims).any()
        assert not np.isnan(buf_stims_unit).any()
        if args.model == 'M1_TTPC_NA_HH':
            bbp_name = 'M1_TTPC_NA_HH'
        else:
            bbp_name = model.cell_kwargs['model_directory']
        if(len(rand)>0):
            outD={'volts':buf_vs[:iSamp+1],'phys_stim_adjust':buf_stims[:iSamp+1],'unit_stim_adjust':buf_stims_unit[:iSamp+1],'phys_par':paramsets[:iSamp+1].astype(np.float32),'unit_par':rand.astype(np.float32)}
        else:
            outD={'volts':buf_vs[:iSamp+1],'phys_stim_adjust':buf_stims[:iSamp+1],'unit_stim_adjust':buf_stims_unit[:iSamp+1],'phys_par':paramsets[:iSamp+1].astype(np.float32),'unit_par':myUpar}
        outF=args.outfile
        # if(args.generate_all):
        #     outF+str(args.cell_i)+".h5"
        # if(args.thread_number):
        #     outF+str(os.environ.get('SLURM_PROCID'))+".h5"
        stim_names=[]
        for stim in args.stim_file:
            ind =stim.rfind('/')
            stim_names.append(stim[ind+1:])

        metadata = {
        'timeAxis': {'step': args.dt, 'unit': "(ms)"},
        'probeName': list(model.get_probe_names()),
        'bbpName': bbp_name,
        'parName': model.PARAM_NAMES,
        'linearParIdx': args.linear_params_inds,
        'stimParRange':[stim_mul_range,stim_offset_range],
        'stimParName': ['stim_mult','stim_offset'],
        'physParRange':get_ranges(args,model),
        'jobId':os.environ['SLURM_ARRAY_JOB_ID'],
        'stimName': stim_names, 
        'neuronSimVer': neuron.__version__,
        'include': get_included(args,model)
    }
        write3_data_hdf5(outD,outF,metaD=metadata)

        
if __name__ == '__main__':
    parser = ArgumentParser()

    with open('cells.json') as infile:
        cells = json.load(infile)
        ALL_MTYPES = cells.keys()
        ALL_ETYPES = list(set(itertools.chain.from_iterable(mtype.keys() for mtype in cells.values())))

    parser.add_argument('--model', choices=MODELS_BY_NAME.keys(),
                        default='hh_ball_stick_7param')
    parser.add_argument('--m-type', choices=ALL_MTYPES, required=False, default=None)
    parser.add_argument('--e-type', choices=ALL_ETYPES, required=False, default=None)
    parser.add_argument('--cell-i', type=int, required=False, default=0)
    parser.add_argument('--cori-start', type=int, required=False, default=None, help='start cell')
    parser.add_argument('--cori-end', type=int, required=False, default=None, help='end cell')
    parser.add_argument('--cori-csv', type=str, required=False, default=None,
                        help='When running BBP on cori, use SLURM_PROCID to compute m-type and e-type from the given cells csv')
    parser.add_argument('--celsius', type=float, default=34)
    parser.add_argument('--dt', type=float, default=.025)

    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help='nwb file to save to')
    parser.add_argument('--metadata-file', type=str, required=False, default=None,
                        help='for BBP only')
    parser.add_argument('--metadata-only', action='store_true', default=False,
                        help='create metadata file then exit')
    parser.add_argument('--create', action='store_true', default=False,
                        help="create the file, store all stimuli, and then exit " \
                        + "(useful for writing to the file from multiple ranks)"
    )
    parser.add_argument(
        '--create-params', action='store_true', default=False,
        help="create the params file (--param-file) and exit. Must use with --num"
    )
    parser.add_argument(
        '--generate-all',action='store_true', default=False,
        help='Generate Data for all Cells including clones, There is no need to pass the i_cell, Should be used with a CSV and num'
    )
    parser.add_argument(
        '--thread-number',action='store_true', default=False,
        help='Adds the thread number at the end of the hdf5 file'
    )

    parser.add_argument(
        '--def-params',action='store_true', default=False,
        help='To be used with create_params, should be used while using default values'
    )
    
    parser.add_argument(
        '--init-cell',action='store_true', default=False,
        help='To be used when performing simulations of two different cells in a same run'
    )
    

    parser.add_argument(
        '--plot', nargs='*',
        choices=['v', 'stim', 'ina', 'ica', 'ik', 'i_leak', 'i_cap', 'v_dend'],
        default=None,
        help="--plot w/ no arguments: plot everything. --plot [args]: print the given variables"
    )
    parser.add_argument('--no-legend', action='store_true', default=False,
                        help="do not display the legend on the plot")
    
    parser.add_argument(
        '--force', action='store_true', default=False,
        help="make the script run even if you don't plot or save anything"
    )

    # CHOOSE PARAMETERS
    parser.add_argument(
        '--num', type=int, default=None,
        help="number of param values to choose. Will choose randomly. " + \
        "See --params. When multithreaded, this is the total number over all ranks, " + \
        "except when using --trivial-parallel"
    )
    parser.add_argument(
        '--trivial-parallel', action='store_true', default=False, required=False,
        help='each process runs all --num samples, with each rank writing output to a ' + \
        'separate file.'
    )
    parser.add_argument(
        '--node-parallel', action='store_true', default=False, required=False,
        help='each node runs --num samples over 64 processes. One output file per node'
    )
    parser.add_argument(
        '--params', type=str, nargs='+', default=None,
        help='When used with --num, fixes the value of some params. To indicate ' + \
        'that a param should not be held fixed, set it to "rand". ' + \
        'to use the default value, use "def"' + \
        'eg to use the default 1st param, random 2nd param, ' + \
        'and specific values 3.0 and 4.0 for the 3rd and 4th params, use "def inf 3.0 4.0"'
    )
    parser.add_argument('--param-file', '--params-file', type=str, default=None)
    parser.add_argument(
        '--blind', action='store_true', default=False,
        help='do not save parameter values in the output nwb. ' + \
        'You better have saved them using --param-file'
    )
    parser.add_argument(
        '--linear', action='store_true', default=False,
        help='when selecting random params, distribute them uniformly' + \
        'throughout the range, rather than exponentially'
    )
    parser.add_argument(
        '--linear-params-inds', type=int, nargs='*', default=[],required=False,
        help='When used with --num, indicates which parameters should be randomized linearly all the other would be randomized exponentially or linearly if --linear is true'
    )
    #CHOOSE PROBES
    parser.add_argument(
        '--axon-probes', type=float,nargs='*',
        help="specify the distances of axonal probe/s from the soma")
    
    parser.add_argument(
        '--dend-probes', type=float,nargs='*',
        help="specify the distances of axonaldend probe/s from the soma")
    

    # CHOOSE STIMULUS
    parser.add_argument(
        '--stim-file', type=str,nargs='+', default=os.path.join('stims', 'chaotic_2.csv'),
        help="csv to use as the stimulus")
    
    parser.add_argument(
        '--stim-dc-offset', type=float,
        help="apply a DC offset to the stimulus (shift it). Happens after --stim-multiplier"
    )
    parser.add_argument(
        '--stim-multiplier',  type=float,
        help="scale the stimulus amplitude by this factor. Happens before --stim-dc-offset"
    )
    parser.add_argument(
        '--stim-noise', action='store_true', default=False,
        help="add random multiplier and dc-offset to the stim in the range hard coded in this script this will override --stim-multiplier and stim-dc-offest"
    )
    parser.add_argument(
        '--tstart', type=int, default=None, required=False,
        help='when to start the recording'
    )
    parser.add_argument(
        '--tstop', type=int, default=None, required=False,
        help='when to stop the recording'
    )

    parser.add_argument('--print-every', type=int, default=1000)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--locked-params', '--lock-params', type=str, nargs='+', default=[])
    
    parser.add_argument(
        '--cell-count', type=int, default=None, required=False,
        help='count of the cells already issued. Used for paralleised reading from csvs'
    )
    
    parser.add_argument(
        '--exclude',nargs='+' ,type=str, default="", required=False,
        help='count of the cells already issued. Used for paralleised reading from csvs'
    )
    
    parser.add_argument(
        '--wide',nargs='+' ,type=str, default="", required=False,
        help='Specify which parameters are to be generated in Wide range, Parameters are by default set to Narrow range'
    )

    parser.add_argument(
        '--unit-param-upper',nargs='+',type=str, default=None, required=False,
        help='Specify the range for sampling of each params - Upper Bound'
    )
    parser.add_argument(
        '--unit-param-lower',nargs='+',type=str, default=None, required=False,
        help='Specify the range for sampling of each params - Lower Bound'
    )

    parser.add_argument('--unit-params-csv', type=str, required=False, default=None,
                        help='CSV to get unit parameter ranges for parameter variation')
    
    
    parser.add_argument(
        '--default-base',action='store_true', default=False,
        help='To be used with create_params, should be used while using default values'
    )


    args = parser.parse_args()

    if args.tstart or args.tstop:
        raise ValueError('--tstart and --tstop not yet implemented')

    log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG if args.debug else log.INFO)

    main(args)
