#!/bin/bash
#SBATCH --job-name horovod1
#SBATCH -D .
#SBATCH --output jobs/hvd_1_4gpus_%j.output
#SBATCH --error jobs/hvd_1_4gpus_%j.err
#SBATCH --ntasks-per-node=4     # must be added to create SLURM_NTASKS_PER_NODE var
#SBATCH --nodes=1               # ntasks must be mutiple of ntasks-per-node. nnodes is ntasks/ntasks-per-node
#SBATCH --cpus-per-task 40      # must be 40 per gpu.
#SBATCH --gres='gpu:4'          # must be ntasks-per-node
#SBATCH --time 00:45:00


module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML


horovodrun -np $SLURM_NTASKS -H localhost:$SLURM_NTASKS --gloo \
python3.7 tf2_keras_cifar_hvd.py --epochs 10 --batch_size 512 --model_name='resnet'
