#! /bin/bash
#PBS -N MakingPlots
#PBS -o /home/niranjan.rajesh_asp24/thesis-manifolds/plots/logs/out.log
#PBS -e /home/niranjan.rajesh_asp24/thesis-manifolds/plots/logs/err.log
#PBS -l ncpus=10
#PBS -q cpu


rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/plots/logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/plots/logs



source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate cv

python  /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/make_plots.py
