#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --nodes 4
#SBATCH --uenv-file=/scratch/e1000/rmeli/squashfs/cp2k-dlaf-cuda.squashfs

module use /user-environment/modules
module --ignore-cache load intel-mkl

miniapp=miniapp/miniapp_eigensolver

nvidia-smi

#ldd ${miniapp}
#libtree ${miniapp} # https://github.com/haampie/libtree

ms=16384
for bs in 256 512 1024
do
    echo -e "\n\nRUNNING ${miniapp} --matrix-size ${ms} --block-size=${bs}\n"
    ${miniapp} --matrix-size ${ms} --block-size ${bs}
done
