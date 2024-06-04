#!/bin/bash
#SBATCH --account=hpcrcf   ### change this to your actual account for charging
#SBATCH --partition=preempt       ### queue to submit to
#SBATCH --job-name=associator    ### job name
#SBATCH --output=hostname.out   ### file in which to store job stdout
#SBATCH --error=hostname.err    ### file in which to store job stderr
#SBATCH --nodes=1               ### number of nodes to use
#SBATCH --ntasks-per-node=1     ### number of tasks to launch per node
#SBATCH --cpus-per-task=1       ### number of cores for each task
#SBATCH --gres=gpu:1          ### General REServation of gpu:number of gpus
#SBATCH --constraint=volta
#SBATCH --mem=10G

module purge
module load tensorflow2
cd /home/jsearcy/cascadia_data_mining
python train_associator_v2.py >> associator_log
 
