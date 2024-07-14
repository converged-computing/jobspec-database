#!/usr/bin/env python
# coding: utf-8
## Credits: http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html

#from mpi4py import MPI
import os
import hostlist  #pip install python-hostlist

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
# get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
local_nodeID = int(os.environ["SLURM_NODEID"])

# get node list from slurm
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    
try:
    # slurm reserves one port per job
    resv_ports = os.environ['SLURM_STEP_RESV_PORTS'].split("-")
    os.environ['MASTER_PORT'] = str(resv_ports[-1])
    os.environ['MASTER_ADDR'] = hostnames[0]
except KeyError:
    print(f'WARNING from slurmTorch.py: Use Fallback MASTER_ADDR and MASTER_PORT')
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
