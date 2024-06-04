#!/bin/bash 
#SBATCH -J erpMpiTesting
p=$1
numabind=$2
module use /gpfs/u/software/erp-spack-install/lmod/linux-centos7-x86_64/Core/
module load gcc
ompi=/gpfs/u//software/erp-rhel7/openmpi/4.0.1/2/
export PATH=$PATH:$ompi/bin
osu=/gpfs/u/home/CCNI/CCNIsmth/barn-shared/CWS/osu-micro-benchmarks-5.6.1-erp-openmpi.4.0.1-verbs-noUcx-Pmi-install/
bin=$osu/libexec/osu-micro-benchmarks/mpi/
binding="--bind-to core"
echo "processes $p"
echo "nodes $SLURM_JOB_NUM_NODES"
echo "ppn $((p/SLURM_JOB_NUM_NODES))"
echo "binding ${binding}"

srun hostname -s > /tmp/hosts.$SLURM_JOB_ID

set -x
export OMPI_MCA_btl_tcp_if_include="ib0"
export OMPI_MCA_btl_openib_allow_ib="1"
mpirun --hostfile /tmp/hosts.$SLURM_JOB_ID ${binding} --report-bindings -n $p $bin/collective/osu_allreduce
set +x
