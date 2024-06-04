#!/bin/bash
#PBS -l walltime=20:30:0,nodes=1:xk:ppn=16,gres=shifter

cd ${PBS_O_WORKDIR}

export PYTHONPATH=${PYTHONPATH}:${PBS_O_WORKDIR}/src

export OMP_NUM_THREADS=16

module load shifter
shifterimg pull luntlab/cs547_project:latest

WANDB_ENTITY="uiuc-cs547-2021sp-group36"
WANDB_PROJECT="image_similarity"

#RUNS="20mi8lwz 2vvezipo 1lqinpcj 37sw0tda 3gvpqxyk 3qk56buy"
RUNS="17vm39k0 qg5hjev2 13hdhxd0"

for one_run in ${RUNS}
do
    mkdir -p results/${one_run}
    pushd results/${one_run}
    
    aprun -b -n 1 -d 16  -- shifter --image=docker:luntlab/cs547_project:latest \
        --module=mpich,gpu -- wandb pull -e ${WANDB_ENTITY} -p ${WANDB_PROJECT} ${one_run}
    
    aprun -b -n 1 -d 16  -- shifter --image=docker:luntlab/cs547_project:latest \
        --module=mpich,gpu -- python ${PBS_O_WORKDIR}/src/eval_neighbors.py \
            --config ./config.yaml --weight_file model_state.pt > results.txt
    
    popd

done
