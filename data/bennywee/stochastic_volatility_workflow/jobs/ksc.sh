#!/bin/bash
#SBATCH --job-name=sbc_cp_ksc_model_cp_dgf_10kmcmc_3000_niter_r1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --array=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-socket=1

# source .env/bin/activate
# python sim_ksc.py
R --vanilla < scripts/sim_ksc_results.r