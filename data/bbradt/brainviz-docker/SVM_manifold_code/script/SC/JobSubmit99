#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80g
#SBATCH -p qTRD
#SBATCH -t 1440
#SBATCH -J sonos
#SBATCH -e /data/users2/salman/projects/multiclass/SVM_manifold_code/slurm/SC99error%A.err
#SBATCH -o /data/users2/salman/projects/multiclass/SVM_manifold_code/slurm/SC99out%A.out
#SBATCH -A PSYC0002
#SBATCH -x trendscn009.rs.gsu.edu
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=msalman@gsu.edu

sleep 10s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/ 
echo $HOSTNAME

module load Framework/Matlab2019b
matlab -batch s2SC99

sleep 10s

