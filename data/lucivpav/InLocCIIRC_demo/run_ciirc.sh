#!/bin/bash
#SBATCH --job-name=InLocCIIRC_demo
#SBATCH --nodes=1
#SBATCH --partition gpu
#SBATCH --output=InLocCIIRC_demo.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --time=1-00:00:00
module load MATLAB/2018a
module load SuiteSparse/5.1.2-foss-2018b-METIS-5.1.0
nvidia-smi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lucivpav/gflags-2.2.2/build/lib:/home/lucivpav/InLoc_demo/functions/vlfeat/toolbox/mex/mexa64
cat startup.m inloc_demo.m | matlab -nodesktop
