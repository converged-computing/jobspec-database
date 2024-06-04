#!/bin/bash

#SBATCH --partition=t1standard
#SBATCH --ntasks=115
#SBATCH --tasks-per-node=23
#SBATCH --mail-user=npbarnes@alaska.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

echo "Setting ulimits"
ulimit -s unlimited
ulimit -f unlimited
ulimit -l unlimited

echo "Setting up modules"
module purge
module load slurm
module load intel/2016
module load openmpi/intel/1.10.7
module load PrgEnv-intel/2016
module load python/anaconda3-2.5.0

echo "Running chinook_setup script"
HYBRID="$(python chinook_setup -i -s $CENTER1/hybrid_data -m "$1")"
if [ $? -ne 0 ]
then
    echo "Chinook_setup failed."
    echo "${HYBRID}"
    exit 1
fi

echo "Change directory to ${HYBRID}"
cd "${HYBRID}"

echo "Generate machinefile"
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID
echo "Start Hybrid Code"
mpirun -np $SLURM_NTASKS --machinefile ./nodes.$SLURM_JOB_ID "${HYBRID}/hybrid" > output 2> error
RESULT=$?
echo "Remove machinefile"
rm ./nodes.$SLURM_JOB_ID

echo "Finish batch script"
exit $RESULT
