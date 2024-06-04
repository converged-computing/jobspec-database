#!/bin/bash -l

# Here you can put your options (particularly job-name, time, output, and error)
# DONT REMOVE THE # IN FRONT OF THE SBATCH

#SBATCH --job-name=letizias-job
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=letizias-logfile-%j.log
#SBATCH --error=letizias-errorfile-%j.err
#SBATCH --account=sd29

args="${@}"

# Here you can put your modules. You can also load PyTorch but in that case it can't be installed in your conda environment
module load daint-gpu
module load cray-python
# module load PyTorch 

# Here you can just put the srun command you would execute. Just replace test_main.py with your script you want to run
# Make sure to replace <your conda environment> below with the name of the conda environment you want to use
srun -ul $HOME/miniconda3/envs/zioboia/bin/python wtf_10.py "${args}" -u

