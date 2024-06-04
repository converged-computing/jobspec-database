#!/bin/bash
#SBATCH --job-name="NB1DSUMW"
#SBATCH --partition=alqadi-amd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time 96:00:00
#SBATCH --mem=200G
#SBATCH --mail-user=johannc2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --nodelist=ccc0324
##
## Cores  Tokens
##   1      5
##   2      6
##   3      7
##   4      8
##   6     10
##   8     12
##  12     14
##  16     16
##  24     19
##  32     21
##  40     23
##  48     25
##  64     28
##  80     31
## 128     38
## ICT cap: 72 tokens (as of  03/27/24)

module use /projects/eng/modulefiles
module load abaqus/2023
module load intel/20.4
module load cuda/11.1

unset SLURM_GTIDS
abaqus inp=NB1DSUMW job=NB1DSUMW user=UMAT scratch=/scratch/users/johannc2/NB1DSUMW cpus=16 gpus=1 mp_mode=mpi memory=200000mb interactive

module unload cuda/11.1
module unload intel/20.4
module unload abaqus/2023


