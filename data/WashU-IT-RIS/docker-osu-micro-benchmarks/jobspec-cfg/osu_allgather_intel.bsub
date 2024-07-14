#!/bin/bash
## export NP=3 && LSF_DOCKER_NETWORK=host LSF_DOCKER_IPC=host LSF_DOCKER_SHM_SIZE=20G bsub -m "compute1-exec-37 compute1-exec-38 compute1-exec-39" -n $NP < osu_allgather_intel.bsub
#BSUB -q qa
#BSUB -R "rusage[mem=100GB] span[ptile=16] select[cpumicro=cascadelake]"
#BSUB -M 100GB
#BSUB -a "docker(registry.gsc.wustl.edu/sleong/osu-micro-benchmark:oneapi)"
#BSUB -G compute-ris
#BSUB -oo lsf-%J.log

. /opt/intel/oneapi/setvars.sh
hostlist=$(echo $LSB_HOSTS | sed 's/compute1-exec-//g' | sed 's/.ris.wustl.edu//g' | tr ' ' '\n' | sort -u | tr '\n' '_' | sed 's/_$//')
mpirun -np $NP $OSU_MPI_DIR/collective/osu_allgather > ./osu_allgather-$hostlist-$LSB_JOBID.log

