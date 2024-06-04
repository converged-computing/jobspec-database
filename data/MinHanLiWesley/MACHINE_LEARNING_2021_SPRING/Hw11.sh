#!/usr/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1
#PBS -q gpu
#PBS -j oe
source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wesley/anaconda3/lib/
cd /home/wesley/ML2021/HW11_Domain_Adaptation
conda activate pytorch
echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "=========================================================="

START_TIME=$SECONDS

# python -u -m $MODEL --no_gene --no_predict --name $NAME --data $DATA | tee /home/wesley/EDC/FPC/ML/results/$NAME/$NAME.log
python -u   hw11_domain_adaptation.py | tee predict.log
# python  -u -m $MODEL --no_train --no_gene --name $NAME   --data $DATA |tee /home/wesley/EDC/FPC/ML/results/$NAME/predict2.log

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
echo "Job Ended at $(date)"
echo '======================================================='

conda deactivate
