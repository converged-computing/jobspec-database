#!/bin/bash
#SBATCH --job-name="Exercise_4"
#SBATCH --qos=debug
#SBATCH -D .
#SBATCH --output=output/Exercise_4_%j.out
#SBATCH --error=output/Exercise_4_%j.err
#SBATCH --cpus-per-task=160
#SBATCH --gres gpu:4
#SBATCH --time=00:15:00

module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML

python $PWD/ex4_MultiGPU/multilayer.py --ngpu 4
