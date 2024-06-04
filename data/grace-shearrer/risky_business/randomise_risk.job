#!/bin/bash
#
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 24:00:00
###################
# %A == SLURM_ARRAY_JOB_ID
# %a == SLURM_ARRAY_TASK_ID 
#SBATCH -o risk_%a_out.txt
#SBATCH -e risk_%a_err.txt
#SBATCH -C "croatan"
#module load intelfort/14.0.3 intelc/14.0.3 netcdf/4.1.3_intel-14.0.3 openmpi/1.8.1_intel-14.0.3_ofed-3.1
#module load matlab/2015b
# Run your executable
#/usr/share/Modules/software/RHEL-6.5/fsl/5.0.9/bin/feat /projects/niblab/data/eric_data/design_files/milkshake/grace_edit/level3/cope${SLURM_ARRAY_TASK_ID}_thr.fsf
/projects/niblab/modules/software/fsl/5.0.10/bin/randomise -i /projects/niblab/data/eric_data/W1/milkshake/level3_grace_edit/cope${SLURM_ARRAY_TASK_ID}_risk_race.gfeat/cope1.feat/filtered_func_data.nii.gz -o /projects/niblab/data/eric_data/W1/milkshake/level3_grace_edit/double_check/cope${SLURM_ARRAY_TASK_ID}_riskrace_randomized -d /projects/niblab/data/eric_data/W1/milkshake/level3_grace_edit/cope${SLURM_ARRAY_TASK_ID}_risk_race.gfeat/design.mat -t /projects/niblab/data/eric_data/W1/milkshake/level3_grace_edit/cope${SLURM_ARRAY_TASK_ID}_risk_race.gfeat/design.con -n 5000 -T -m /projects/niblab/data/eric_data/W1/milkshake/level3_grace_edit/cope${SLURM_ARRAY_TASK_ID}_risk_race.gfeat/mask.nii.gz
