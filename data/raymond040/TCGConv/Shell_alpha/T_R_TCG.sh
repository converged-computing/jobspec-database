#!/bin/bash
#PBS -P Project
#PBS -j oe
#PBS -q volta_gpu
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=24:00:00

cd #PBS_o_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);

image="/home/svu/e0407728/SIF/edge-hpc_v0.1.sif"
singularity exec -e $image bash << EOF > $PBS_JOBID.$PBS_JOBNAME.out 2> $PBS_JOBID.$PBS_JOBNAME.err

python3 "/home/svu/e0407728/My_FYP/TCGConv/tune_alpha.py" --dataset_name "R" --percentage 0.3 --type_ED "sub" --num_groups 30 --root "/home/svu/e0407728/My_FYP/TCGConv/" --num_version $PBS_JOBID --model_type "TCGConv" --n_run 5
EOF
