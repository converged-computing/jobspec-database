#!/bin/sh
#SBATCH -A fc_cosmoml
#SBATCH -p savio2_1080ti
#SBATCH -N 1
#SBATCH -n 4 
#SBATCH -c 2 
#SBATCH --gres=gpu:4 
#SBATCH -t 8:00:00

mpiexec -n 4 /global/home/users/mariusmillea/src/julia-1.5.2/bin/julia \
    --project=/global/home/users/mariusmillea/work/ptsrclens/Project.toml \
    -e 'using ClusterManagers; ClusterManagers.elastic_worker("marius          ","10.0.0.24",9312)'
