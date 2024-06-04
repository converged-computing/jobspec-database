#!/usr/bin/env bash
#PBS -l select=1:ncpus=40:mem=372gb:mpiprocs=40
#PBS -l walltime=24:00:00
#PBS -M robertu@clemson.edu
#PBS -m abe
#PBS -j oe
#PBS -N run_evaluation

cd ~/projects/libpressio_opt_experiments/
source ~/git/spack/share/spack/setup-env.sh
spack env activate .

./run_evaluation.py --search_progress --nbins 10 --n_proc 40
