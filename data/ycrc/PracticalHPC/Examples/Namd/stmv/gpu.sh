#!/bin/bash

#SBATCH -p pi_cryoem,scavenge -c 8 -t 24:00 --gres=gpu:4 --gres-flags=enforce-binding

module load NAMD/2.13-multicore-CUDA

# When running CUDA NAMD always add +idlepoll to the command line. This is needed to poll the GPU for results rather than sleeping while idle.

namd2 +idlepoll +ppn $SLURM_CPUS_ON_NODE stmv.namd 

