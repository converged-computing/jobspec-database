#!/bin/bash
#SBATCH --job-name=19280
#SBATCH --output=outputvc.out
#SBATCH --error=25.err
#SBATCH --partition=small
##SBATCH --gres=gpu:2
##SBATCH --time=03-00:00:00
#SBATCH --ntasks-per-node=48
#SBATCH -N 3
hostname

#------------------loading the programs-----------------------#

module load DL-CondaPy/3.7
source /home/apps/spack/share/spack/setup-env.sh
module load oneapi/compiler/2022.0.2
module load oneapi/mpi/2021.5.1
source /opt/ohpc/pub/intel/oneapi/setvars.sh
module load QE-oneapi-compiler/7.1

#----------------checking for SCRATCH folder------------------#

if [ ! -d //scratch/$USER ]
then
  mkdir -p //scratch/$USER
fi

#----------------make temp file-------------------------------#

tdir=$(mktemp -d //scratch/$USER/qe_job__$SLURM_JOBID-XXXX)

#----------------copy the files to scratch--------------------#

cd $tdir
cp $SLURM_SUBMIT_DIR/na5alh14.in .

echo "Job execution start: $(date)" 
echo "Job submitted by ${USER}" 
echo "The job ID is:${SLURM_JOBID}"  
mpirun -n $SLURM_NTASKS pw.x -in na5alh14.in  
echo "Job execution finished at: $(date)" 

