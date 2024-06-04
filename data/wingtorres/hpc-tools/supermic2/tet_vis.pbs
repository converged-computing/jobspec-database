#!/bin/bash
#PBS -q workq
#PBS -A hpc_sel_l01
#PBS -l nodes=1:ppn=16
#PBS -l walltime=01:00:00  
#PBS -N tet_vis
#PBS -o tet_vis.out
#PBS -e tet_vis.err
#PBS -m abe
cd $PBS_O_WORKDIR
module load matlab
matlab -nodisplay -nodesktop -nosplash  -r "cd /home/wtorres/COAWST/; tet_vis; exit"
#/bin/bash tet_vis.sh
