#!/bin/bash
#SBATCH --job-name=train_adafac        # name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
#SBATCH -A wjm@a100
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
#SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
##SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
#SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=12:00:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=gpu_mlp_nod_meta%j.out    # name of output file
#SBATCH --error=gpu_mlp_nod_meta%j.out     # name of error file (here, in common with the output file)

# Cleans out the modules loaded in interactive and inherited by default
module purge

cd /gpfswork/rech/bao/unl88dr/learned_delayed_optim/
module load cpuarch/amd
module load cudnn/10.1-v7.5.1.10
module load python/3.11.5
source venv3/bin/activate
wandb offline

set -x

python3.11 src/main.py \
    --config config/meta_train/meta_train_mlp_image-mlp-fmst_schedule_3e-3_10000_d3.py \
    --num_local_steps 4 \
    --num_grads 8 \
    --local_learning_rate 0.5 \
    --tfds_data_dir $STORE/tf_datasets\
    --wandb_dir $STORE/jax_wandb