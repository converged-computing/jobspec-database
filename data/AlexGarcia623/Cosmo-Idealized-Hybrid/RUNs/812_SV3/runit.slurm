#!/bin/bash					
#SBATCH --job-name=S12SV3
#SBATCH --mem-per-cpu=3750mb
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=j.rose@ufl.edu
#SBATCH --mail-type=FAIL
#SBATCH --partition=hpg2-compute
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
##SBATCH --qos=paul.torrey-b
#SBATCH --dependency=singleton
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --output=./output/output_%j.out
#SBATCH --error=./output/error_%j.err

module purge
module load intel/2018.1.163
module load openmpi/3.1.2
module load gsl/2.4
module load fftw/3.3.7
module list

export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^vader,tcp,openib"
export OMPI_MCA_oob_tcp_listen_mode="listen_thread"

sbatch runit.slurm

srun --mpi=pmix_v2 ./Arepo param.txt 1
