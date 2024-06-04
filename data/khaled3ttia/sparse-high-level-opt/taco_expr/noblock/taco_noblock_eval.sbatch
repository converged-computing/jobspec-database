#!/bin/bash
#
#SBATCH --partition=disc
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=256G
#SBATCH --output=taco_noblocking_%J_stdout.txt
#SBATCH --error=taco_noblocking_%J_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=taco_noblocking
#SBATCH --mail-user=khaled.abdelaal@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/khaled/
#
#################################################
#echo "LSCRATCH Path is: $LSCRATCH"
#echo "LSCRATCH Size is: `df -h $LSCRATCH`"

#echo "output of nproc is: `nproc`"

#echo "Copying input files from scratch to LSCRATCH..."
#time cp /scratch/khaled/k21_same/*.mtx $LSCRATCH/

#echo "Copying Apptainer Image to LSCRATCH..."
#time cp /scratch/khaled/taco_py_latest.sif $LSCRATCH/

#echo "Copying bash script from home to LSCRATCH..."
#time cp /home/khaled/evalKronSpMV/taco/run_tool.sh $LSCRATCH/
cp /home/khaled/sparse-high-level-opt/taco_expr/run_tool_noblocking.sh /scratch/khaled/dask_out

echo "Starting Apptainer Container..."
apptainer exec --bind /scratch/khaled/dask_out:/data/ taco_py_latest.sif /data/run_tool_noblocking.sh /data/ /data/taco_noblock.csv 10 


echo "CPU SPEC"
lscpu
#mkdir -p ~/sys_info_gpu
#cd ~/sys_info_gpu
#lscpu > cusp_cpu_info.txt
#nvidia-smi > cusp_gpu_info.txt
#lsmem > cusp_mem_info.txt
