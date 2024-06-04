#!/bin/bash

#PBS -W group_list=wolg
#PBS -m bea
#PBS -M tiankuizhang@email.arizona.edu
#PBS -q windfall
#PBS -l select=1:ncpus=8:mem=40gb:ngpus=1
#PBS -l walltime=10:00:00
###PBS -l cput=80:00:00
#PBS -l place=free:shared

#PBS -N ellipsoid_relaxation

cd ~tiankuizhang/ClusterGit/7_Vesicle/src/6_0_numerical_experiments/3_0_Single_Phase_Vesicle

matlab -nodisplay -nosplash -r 'ne1_single_phase_vesicle("Prolate",0.90,1000,10,1)'
