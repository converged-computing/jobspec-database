#!/bin/sh

source /etc/profile.d/modules.sh

module load singularity
#SINGULARITYENV_APPEND_PATH=/lib64/mpich-3.2/bin/
exec singularity shell -B /scratch:/scratch -B /var/spool/torque:/var/spool/torque ~/scratch/cs484.sif ${*}
