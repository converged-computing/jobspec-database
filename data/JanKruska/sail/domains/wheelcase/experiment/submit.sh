#!/bin/bash
#SBATCH --partition=hpc          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --mem=160G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=72:00:00          # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --export=ALL
#SBATCH --exclusive

# Necessary to pseudo-revert to old memory allocation behaviour
export MALLOC_ARENA_MAX=4

module load java/default
module load cuda/default
module load matlab/R2019b
module load openmpi/gnu
source ~/OpenFOAM-plus/etc/bashrc

# Run experiment
matlab -batch "wheelcase_runSail('nCases',4,'caseStart',11,'gens',6,'config','config1')"
