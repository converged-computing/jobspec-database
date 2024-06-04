#!/bin/bash

#PBS -W group_list=wolg
#PBS -m bea
#PBS -M tiankuizhang@email.arizona.edu
#PBS -q standard
#PBS -l select=1:ncpus=8:mem=40gb:ngpus=1
#PBS -l walltime=40:00:00
###PBS -l cput=320:00:00
#PBS -l place=free:shared

#PBS -N PBS_SurfaceDiffusion_nucelation

cd ~tiankuizhang/ClusterGit/7_Vesicle/src/6_0_numerical_experiments/2_0_SurfaceDiffusion

matlab -nodisplay -nosplash -r 'PBS_SurfaceDiffusion(0.9,0.5,60000,100)'


