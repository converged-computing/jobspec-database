#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --gres=gpu:1 
#SBATCH -t 01:00:00 

# module load esslurm before launching this job script on Cori
module load cuda/11.0.2
module load pgi/19.10

input=gpp214unformatted.dat

cd example-codes/GPP/

# Baseline
output=output
profilestr="ncu -k sigma_gpp_gpu -f -o $output --section-folder ../../ncu-section-files --section SpeedOfLight_HierarchicalDoubleRooflineChart "
echo Baseline version
git checkout gpp.f90
make clean
make
srun -n1 $profilestr ./gpp.x $input 

# Four optimization steps 
for n in `seq 1 4`
do
	output=output$n
	profilestr="ncu -k sigma_gpp_gpu -f -o $output --section-folder ../../ncu-section-files --section SpeedOfLight_HierarchicalDoubleRooflineChart "
	echo Patch version: $n
	git checkout gpp.f90
	patch gpp.f90 step$n.patch
	make clean
	make
	srun -n1 $profilestr ./gpp.x $input  
done
