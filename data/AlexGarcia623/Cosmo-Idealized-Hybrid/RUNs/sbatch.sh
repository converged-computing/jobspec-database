#!/bin/bash					
#SBATCH -J mid_du
#SBATCH --mem-per-cpu=3500mb
#SBATCH --time=96:00:00
#SBATCH --mail-user=jqi@ufl.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --partition=hpg2-compute
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --qos=paul.torrey-b		###narayanan	#paul.torrey
##SBATCH --account=astronomy-dept
##SBATCH --qos=astronomy-dept-b
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

srun --mpi=pmix_v2   ./arepo/Arepo   param.txt 1
