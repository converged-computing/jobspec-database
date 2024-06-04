#!/bin/bash
## The following lines describe slurm job parameters to run the calculations in an HPC environment with SLURM
## They might need to be modified to the particular HPC facility's specification.

#SBATCH --job-name="MyJob"
#SBATCH --output="MyJob_%j.out"
#SBATCH --error="MyJob_%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# The following lines may need to be included to activate a conda enviroment of your choice
# module load avalaiable_conda_module_in_the_HPC_facility
# conda activate your_conda_Env

# Path for the Lammps potential
VKMC=/path/to/the/local/copy/of/the/VKMC/repository # We Need to add the full path to the local copy of the VKMC respository here.
potpath=$VKMC/Utils/pot

# Path to the directory containing the Monte Carlo Python module
LMP_MC=$VKMC/Utils/MEAM_MC


# Runs from scratch
# By default, the supercells are orthogonal supercells, unless the flag "-pr" or "--Primitive" is set. In our paper, we used orthogonal supercells.

T=1073

# We first specify a "jobID" which specifies the atom count
# for jobID=1, the atom counts will be 99, 100, 100, 100, 100
# for jobID = 2, we'll set them as 100, 99, 100, 100, 100 and so on.
jobID=1 # or 2 or 3 or 4 or 5 depending on which simulation is being done.
mkdir ${T}_${jobID}

# In each such job, We'll simulate 40 independent trajectories at 1073K, each starting from a random state.
for i in {1..40}
do
	mkdir ${T}_${jobID}/${T}_${jobID}_${i} # This is the directory in which all run results will be stored.
	cd ${T}_${jobID}/${T}_${jobID}_${i}
	python -u $LMP_MC/Init_state_MC.py -pp $potpath -T $T -etol 0.0 -ftol 0.001 -na 99 100 100 100 100 -nt 120000 -ne 250 -ns 250 -dmp -dpf args_continue_${i}.txt > JobOut_${i}.txt 2>&1 &
	cd ../../
done
wait

# continuation runs - to continue, we specify the -ckp flag (see the ReadMe.txt file)
# This will restart the simulations from two supercells before the last saved ones.
#for i in {1..40}
#do
#	cd ${T}_${jobID}/${T}_${jobID}_${i}
#	python -u $LMP_MC/Init_state_MC.py -pp $potpath -T $T -ckp -nt 120000 -ne 250 -ns 250 -dmp -dpf args_continue_${i}.txt > JobOut_${i}.txt 2>&1 &
#	cd ../../
#done
#wait
