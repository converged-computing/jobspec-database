#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=4:mpiprocs=4:host='SUTD-hpc-gn4'+1:ncpus=8:ngpus=4:mpiprocs=4:host='SUTD-hpc-gn3'
#PBS -N test_nlp
#PBS -q blackout1
#PBS -j oe

nvidia-smi

cd "$PBS_O_WORKDIR";

echo $PBS_NODEFILE
cat $PBS_NODEFILE

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)

image="/app/singularity/images/pytorch-ngc-nlp_latest.sif"

export PATH="/opt/conda/bin/:$PATH"

/Apps/openmpi-4.0.3_pbs_sif_cuda/bin/mpirun \
   -n 8 \
   --mca pml ob1 --mca btl tcp,self,vader --mca btl_tcp_if_include bond0  --hostfile $PBS_NODEFILE \
   -bind-to none -map-by slot --mca btl_openib_warn_default_gid_prefix 0 \
   -x NCCL_IB_GID_INDEX=3 \
   -x NCCL_CHECKS_DISABLE=1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_bond_0 -x NCCL_IB_CUDA_SUPPORT=1 \
   /app/singularity/3.5.3/bin/singularity exec --nv $image \
   python /home/users/uat/multinode/nlptest.py &> log.txt

echo 'Exit'
