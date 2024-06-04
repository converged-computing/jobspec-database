#!/bin/bash

#SBATCH --job-name=mlreco_p100
#SBATCH --output=gridlog_mlreco_p100.log
#SBATCH --mem-per-cpu=8g
#SBATCH --cpus-per-gpu=2
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:p100:4
#SBATCH --partition=gpu,ccgpu,wongjiradlab
#SBATCH --error=gridlog_train_larmatch.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_ubmlreco_uresnet_ppn.sh"
#echo "TEST"
