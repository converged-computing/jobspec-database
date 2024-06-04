#!/bin/bash
#PBS -q SQUID
#PBS --group=G15445
#PBS -l elapstim_req=30:00:00
#PBS -l gpunum_job=2
#PBS -l cpunum_job=76
#PBS -v OMP_NUM_THREADS=76
#PBS -M nagaharago@weblab.t.u-tokyo.ac.jp
#PBS -m bea

cd $PBS_O_WORKDIR
export SINGULARITY_BIND="`readlink -f /sqfs/work/$GROUP_ID/$USER`,$PBS_O_WORKDIR"
nvidia-smi
python --version
singularity run --nv --bind /sqfs/work/$GROUP_ID/$USER_ID:/sqfs/work/$GROUP_ID/$USER_ID /sqfs/work/$GROUP_ID/$USER_ID/sif_images/lampp.sif python image_segmentation/RedNet_inference.py --output=/sqfs/work/G15445/u6b795/lampp/modelchain --data-dir=/sqfs/work/G15445/u6b795/SUNRGBD --cuda --last-ckpt=/sqfs/work/$GROUP_ID/$USER_ID/rednet_ckpt/ckpt_epoch_245.00.pth --use-model-chaining --visualize
# singularity run --nv --bind /sqfs/work/$GROUP_ID/$USER_ID:/sqfs/work/$GROUP_ID/$USER_ID /sqfs/work/$GROUP_ID/$USER_ID/sif_images/lampp.sif python test.py