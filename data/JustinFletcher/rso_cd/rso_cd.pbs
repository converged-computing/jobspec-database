#!/bin/bash
## Walltime in hours:minutes:seconds
#PBS -l walltime=100:00:00
## -o specifies output file
#PBS -o ~/log/queue_exhaustion.out
## -e specifies error file
#PBS -e ~/log/queue_exhaustion.error
## Nodes, Processors, CPUs (processors and CPUs should always match)
#PBS -l select=1:mpiprocs=20:ncpus=20
## Enter the proper queue
#PBS -q standard
#PBS -A MHPCC96650DE1

module load anaconda3/5.2.0
module load tensorflow/1.11.0
cd ~/projects/rso_cd/

python train_rso_change_detection.py --dataset_path="/gpfs/home/fletch/data/rso_cd/" --log_path="/gpfs/home/fletch/logs/rso_cd/" --gpu_list="0, 1, 2, 3" --num_training_epochs=100000 --validation_callback_frequency=128 --batch_size=4096