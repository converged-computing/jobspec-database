#!/bin/sh

#--- NAME
#SBATCH --job-name frac_train
#SBATCH --output output_train/slurm-%j.out

#--- CPU
#SBATCH --mem 64G

#--- GPU
#SBATCH -p private-cui-gpu
#SBATCH --gres=gpu:1,VramPerGpu:10G

#--- Time
#SBATCH -t 3-00:00:00

#--- Load modules.
module purge
module load GCC/10.3.0  OpenMPI/4.1.1
module load PyTorch/1.11.0-CUDA-11.3.1
module load NiBabel/3.2.1
module load matplotlib/3.4.2
module load scikit-image/0.18.1
module load torchinfo/1.5.2-PyTorch-1.11.0-CUDA-11.3.1

#--- Outputs details about the node and cores used.
nvidia-smi

#--- Execute script.
python -m main --train_image_dir ../data/ribfrac-train-images-2/Part2 --train_label_dir ../data/ribfrac-train-labels-2/Part2 --val_image_dir ../data/ribfrac-val-images --val_label_dir ../data/ribfrac-val-labels --save_model True
