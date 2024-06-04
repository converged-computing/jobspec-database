#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=mh-kin
#SBATCH --time=3-00:00:00 #  <=====
#SBATCH --job-name="prior_mc" #  <=====
#SBATCH --array=1-20 #  <===== #chains
### Redirect stderr and stdout to the same file:
### %A will be replaced by the job ID and %a by the array index
####################
###SBATCH -o "$arg".out
###SBATCH -e "$arg".out
#SBATCH --mail-user=piero.mana@ntnu.no
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_90,FAIL
###SBATCH --exclusive
#SBATCH --mem-per-cpu=4G
#### SBATCH --nodes=1
#SBATCH --ntasks=1
##################################################################
call="source('$1')"
###call="$1"
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "Total of $SLURM_NTASKS cores"
echo ""

module purge
module load GCC/8.3.0 OpenMPI/3.1.4  R/3.6.2 netCDF/4.7.1
###module load GCC/8.2.0-2.31.1 CUDA/10.1.105 OpenMPI/3.1.3 icc/2019.1.144-GCC-8.2.0-2.31.1 impi/2018.4.274 ifort/2019.1.144-GCC-8.2.0-2.31.1 netCDF/4.6.2 R/3.6.0

Rscript -e $call ${SLURM_ARRAY_TASK_ID}
###mpirun -n 1 Rscript -e $call
###mpirun hostname
###mpirun -np 40 R --slave -f $call
