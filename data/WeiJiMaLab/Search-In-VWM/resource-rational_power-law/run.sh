#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=75:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=test
#SBATCH --mail-type=END
#SBATCH --output=fminbnd_nll_std__%j.out

#NAME="fminbnd_nllstd__output"
#WORKDIR="${SCRATCH}/${NAME}"

module purge
#. /etc/profile.d/modules.sh

# Use Intel compiler
module load matlab/R2013a

#Check if running as an array job

if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        IID=${SLURM_ARRAY_TASK_ID}
fi

# Run the program

#echo ${WORKDIR} ${NAME} ${IID}.job

cat<<EOF | matlab -nodisplay
addpath('/jukebox/scratch/aditis/bads-master/');
#cd('${WORKDIR}');
optim_normative_sim_pow($IID)
EOF

