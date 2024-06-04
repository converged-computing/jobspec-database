#!/bin/bash

#SBATCH -o log_2.out                            # Output file
#SBATCH -p gpu                                    # Partition
#SBATCH -w clamps
#SBATCH --gres=gpu:1                              # number of GPUs to grab
#SBATCH --ntasks=1                              # number of CPU cores to grab
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00                           
#SBATCH --mem-per-cpu=8192                        # 500 MB of RAM per CPU core

uname -a                                          # Display assigned cluster info
srun echo "I am on"
srun echo $HOSTNAME
srun echo "I got gpu number"
srun echo $CUDA_VISIBLE_DEVICES
srun nvidia-docker run --rm -e CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES` -v /data/datasets:/data/datasets -v /storage2/datasets:/storage2/datasets -v /local:/local -v /home/$USER:/home/$USER -v /storage1:/storage1 madratman/deeprl_hw /bin/bash -c /home/ratneshm/slurm_deeprl/meta_scripts/2.sh
