#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=2:00:00
#SBATCH --mem=20GB
#SBATCH --exclude=c5003
#SBATCH --job-name=grabDeut
#SBATCH --output=logs/deut_grabow.%a.log
#SBATCH --error=logs/deut_grabow.%a.slurm.log
#SBATCH --partition=short
#SBATCH --mail-user=blais.ch@northeastern.edu 
#SBATCH --mail-type=FAIL,END

#an array for the job. usually 107, testing use 2
#SBATCH --array=0-175


####################################################
source ~/_02_RMG_envs/RMG_julia_env/.config_file
source activate rmg_julia_env
# RMG_MODEL="/work/westgroup/ChrisB/_01_MeOH_repos/RMG_run_comparisons/bep_parameter_study/rmg_runs/meoh_main"
# CTI_FILE="/work/westgroup/ChrisB/_01_MeOH_repos/RMG_run_comparisons/bep_parameter_study/rmg_runs/meoh_main/base/cantera/chem_annotated.cti"
python -u Grabow_sbr_script.py  #$CTI_FILE $RMG_MODEL
