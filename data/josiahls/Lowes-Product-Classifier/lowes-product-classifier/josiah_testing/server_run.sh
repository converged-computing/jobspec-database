#! /bin/bash
# exec 1>$PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS OPTIONS =====
#
#PBS -N "classifier_jlaivins"
#PBS -q copperhead
#PBS -l walltime=400:00:00
#PBS -l nodes=1:ppn=1:gpus=1,mem=16GB
#PBS -V
#
# ===== END PBS OPTIONS =====
#
# ==== Main ======
cd $PBS_O_WORKDIR
mkdir log
{
module load cuda/9.2 cudnn/7.2.1-cuda9.2 tensorflow/1.10.0-anaconda3-cuda9.2
python ./Driver.py
} > log/output_"$PBS_JOBNAME"_$PBS_JOBID 2>log/errorLog_"$PBS_JOBNAME"_$PBS_JOBID
