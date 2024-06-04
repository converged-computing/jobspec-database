#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --output=YB_L0d75A0d75B0d135_checkNumberOfSuitableGrowth_growthFreq25_strain0d05_# This affects the print out of the "std::cout" in the script, make sure this is changed for different jobs.
#SBATCH --mail-user=useremail@address.com
#SBATCH --mail-type=ALL
#SBATCH --job-name="b0d135"
#SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu

module load singularity
export SINGULARITY_NV=1
module load centos
module load extra
module load GCC
module load cuda/9.1

centos.sh "module load cuda/9.1; ./virus-model -dt=0.001 Data_structure.xml"
