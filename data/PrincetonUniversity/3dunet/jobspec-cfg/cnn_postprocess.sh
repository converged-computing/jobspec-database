#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o /scratch/gpfs/zmd/logs/cnn_postprocess_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/gpfs/zmd/logs/cnn_postprocess_%j.err        # STDERR #add _%a to see each array job

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anaconda3/5.3.1
. activate 3dunet

echo "Experiment name:" "$1"
echo "Storage directory:" "$2"

#generate memmap array of reconstructed cnn output
OUT0=$(sbatch slurm_scripts/cnn_step21.sh "$1" "$2") 
echo $OUT0

#populate reconstructed array
OUT1=$(sbatch --dependency=afterany:${OUT0##* } --array=0-130 slurm_scripts/cnn_step2.sh "$1" "$2") 
echo $OUT1

#generate cell measures
OUT2=$(sbatch --dependency=afterany:${OUT1##* } --array=0-30 slurm_scripts/cnn_step3.sh "$1" "$2") 
echo $OUT2

#export csv dictionary and run last check
OUT3=$(sbatch --dependency=afterany:${OUT2##* } slurm_scripts/cnn_step4.sh "$1" "$2") 
echo $OUT3

#functionality
#go to 3dunet main directory and type sbatch cnn_postprocess.sh [path to lightsheet package output directory] [path to scratch directory in file system]
