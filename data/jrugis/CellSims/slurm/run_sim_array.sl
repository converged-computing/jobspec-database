#!/bin/bash
#SBATCH -J Cell_Sim
#SBATCH -A nesi00119           # Project Account
#SBATCH --time=0:29:00         # Walltime HH:MM:SS
#SBATCH --mem-per-cpu=2G       # Memory
#SBATCH --ntasks=1             # number of tasks
#SBATCH --cpus-per-task=1      # number of threads
#SBATCH --output=slurm.out     # output file
#SBATCH --error=slurm.err      # error file
#SBATCH --array=1-11           # size of array job (updated automatically)
#SBATCH -C avx                 # run on ivybridge or sandybridge (faster than westmere)
##SBATCH --gres=gpu:1          # for cuda version only! (updated automatically)

# load module(s)
module load intel/2015a
module load Python/3.5.1-intel-2015a
module load CUDA/7.5.18

# check array file exists
if [ ! -f _pan_array.in ]; then
    >&2 echo "Error: _pan_array.in file does not exist"
    exit 1
fi

# get working directory and variables from file
line=$(head -n ${SLURM_ARRAY_TASK_ID} _pan_array.in | tail -1)
set $line
vExe=$2     # executable name
vModel=$3   # model name
vMesh=$4    # mesh name
vRoot=$5    # simulation root directory

# switch to working directory
echo "$HOSTNAME : $1"
cd "$1"

# run the job
srun -o "$vExe.txt" "$vRoot/executables/$vExe"

# post-processing
mv cs.dat "$vModel.dat"
mv cs.msh "$vMesh.msh"

if [ -f c.bin ]; then
    # create reduced content output files
    srun --output=reduce.txt --ntasks=1 python "$vRoot/post/cs_reduce_min-max.py" "."
fi
