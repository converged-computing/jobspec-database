# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:14:41 2020

@author: daniel
"""

from mpi4py import MPI 
import d1msn as msn
import plasticity_experiment as pe
import parameters as pp
import numpy as np
import pickle
import json
import scipy.signal as ss
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
print("Number of processes = %d" % nprocs)
print("Before opening model sets")

with open('D1_71bestFit_updRheob.pkl', 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")
print("Before creating a cell")

cell_index = 34
eta = np.exp((int(sys.argv[1]) -30)*0.1)

variables = model_sets[cell_index]['variables']
cell = msn.MSN(variables = variables)
independent_dends = pp.independent_dends
cell.increase_dend_res(independent_dends, 5)
cell.insert_spines(independent_dends, pp.cluster_start_pos, pp.cluster_end_pos, num_spines = pp.plateau_cluster_size_max)
print("Created a cell")

on_spine = 1
add_spine = 0
vs_amps = []
vs_durs = []
vd_amps = []
vd_durs = []
vs_list = []
pp.eta = eta
if rank == 0:
    # 2. Create tasks for the queue of tasks for parallel execution
    tasks = []
#    alphas = (np.arange(0.054,0.096,0.004)).tolist()
    num_syns = (np.arange(1,21,1)).tolist()
    trials = 50
    for input_size in num_syns:
        for trial in range(1, trials + 1):        
            for dend in independent_dends:
#                for alpha in alphas:
#                  tasks.append([input_size, trial, dend, alpha])
                  tasks.append([input_size, trial, dend])
                  
    div, res = divmod(len(tasks), nprocs)
    counts = [div + 1 if p < res else div for p in range(nprocs)]
   
    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]
    tasks = [tasks[starts[p]:ends[p]] for p in range(nprocs)]
else:
    tasks = None

tasks = comm.scatter(tasks, root=0)
for t in tasks:
    cluster_size = t[0]; tr = t[1]; dend = t[2];
    print("Running trial %d for eta = %.3f on dendrite %d with %d clustered synapses." % (tr, eta, dend, cluster_size))

    vs = []
    ex = pe.Plasticity_Experiment('record_ca', cell)
    ex.insert_synapses('noise_SPN')
    ex.insert_synapses('my_spillover', [dend], deterministic = 0, 
                       num_syns = cluster_size, add_spine = add_spine, on_spine = on_spine)
    ex.set_up_recording([dend])
    ex.simulate()
    tv = ex.tv.to_python()
    t = ex.t.to_python()
    vs.append(ex.vs.to_python())
    vs_list.append(ex.vs.to_python()[int(pp.skip_first_x_ms*pp.nrn_dots_per_1ms):])
    
    cell.esyn = []
    ex.estim = []
    ex.enc = []

    cell.isyn = []
    ex.istim = []
    ex.inc = []

    ex.exglusec = []
    ex.exglu = []
    ex.exnc = []
    for s in cell.spines:
        s.syn_on = 0

#
    vs_indices = []; vs_widths = []
    vd_indices = []; vd_widths = []
    for v in vs:
        vs_indices.append(ss.find_peaks(v))
        vs_widths.append(ss.peak_widths(v, (vs_indices[-1])[0], rel_height = 0.15))
        
    vs_durs.append([v[0][0] for v in vs_widths])
    vs_amps.append([v[1][0] for v in vs_widths])

vs_durs = comm.gather(vs_durs, root = 0)
vs_amps = comm.gather(vs_amps, root = 0)
vs_list = comm.gather(vs_list, root = 0)      
# 5. Calculate and plot results
if rank == 0:
    res1 = []; res2 = []; 
    for vsa, vsd in zip(vs_amps, vs_durs):
        res1.extend(vsa)
        res2.extend(vsd)
    vs_amps = res1;  vs_durs = res2; 
#    vs_amps = np.reshape(vs_amps, (len(num_syns), trials, len(independent_dends), len(alphas)) )    
#    vs_durs = np.reshape(vs_durs, (len(num_syns), trials, len(independent_dends), len(alphas)) )    
    
    res = []
    for v in vs_list:
        res.extend(v)
    vs_list = res
    
    res_dict = {'num_syns': num_syns,
                'trials': trials,
                'independent_dends': independent_dends,
                'vs_widths': vs_durs,
                'vs_amps': vs_amps,
                'vs': vs_list}
    to_save = json.dumps(res_dict)
    with open('./results/eta/spillover_eta_%.3f.dat' % eta, 'w', encoding = 'utf-8') as f:
        json.dump(to_save, f)
