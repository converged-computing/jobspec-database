#!/bin/bash
#SBATCH --job-name=phase_separation
#SBATCH --account=INF23_biophys_2
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=0
#SBATCH --output=%x_%j.log
#SBATCH --mem-per-cpu=1024M
source ~/.bashrc
julia -p 28 distributed_je.jl
