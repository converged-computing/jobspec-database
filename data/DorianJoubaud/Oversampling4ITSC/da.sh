#!/bin/bash -l
#SBATCH -J DA
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=dorian.joubaud@uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=1-23:59:00
#SBATCH -p batch
#SBATCH --qos=normal

#SBATCH --array=0-4

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
# Run your application as a job step,  passing its unique array id
# (based on which varying processing can be done)
#module load lang/Anaconda3/2020.02

VALUES=(ROS Jitter TW SMOTE ADASYN)


 

conda activate da
python main_100.py $1 ROCKET ${VALUES[$SLURM_ARRAY_TASK_ID]} 01 2  10 

