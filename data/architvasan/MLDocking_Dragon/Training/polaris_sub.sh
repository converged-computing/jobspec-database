#!/bin/bash
#PBS -N st_rec0
#PBS -l select=4
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=grand
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

module load conda/2023-10-04
conda activate /lus/eagle/projects/datascience/avasan/envs/st_env_10_04

#cd /grand/datascience/avasan/Benchmarks_ST_Publication/ST_Revised_Train_multiReceptors/3CLPro_7BQY_A_1_F

NP=16
PPN=4
OUT=test_mod_token_embedding.log
let NDEPTH=64/$NP
let NTHREADS=$NDEPTH

TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true

mpiexec --np 4 -ppn 4 --cpu-bind verbose,list:0,1,2,3,4,5,6,7 -env NCCL_COLLNET_ENABLE=1 -env NCCL_NET_GDR_LEVEL=PHB python smiles_regress_transformer_run.py > $OUT
#,9,10,11,12,13,14,15,16
