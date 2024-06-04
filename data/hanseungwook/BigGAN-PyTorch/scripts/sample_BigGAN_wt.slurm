#!/bin/bash
###  replace the .py file in the horovodrun call with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err

#SBATCH -J biggan_sample
#SBATCH -o biggan_sample_%j.out
#SBATCH -e biggan_sample_%j.err

#SBATCH --mail-user=seungwook.han@ibm.com
#SBATCH --mail-type=ALL
###  the following parameters will get you two nodes with 4 V100's each
###  note the run time limit is currently only 12 hours, it will be increased
###  next week
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=500g
#SBATCH --time=24:00:00
#SBATCH -p sched_system_all 

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=wmlce-ea
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

# use z_var to change the variance of z for all the sampling
# use --mybn --accumulate_stats --num_standing_accumulations 32 to 
# use running stats
python sample_wt.py \
#--model BigGANdeep \
--dataset WT64 --shuffle  --num_workers 8 --batch_size 256 \
--base_root /nobackup/users/swhan/ICLR/biggan/BigGAN-PyTorch-3/results_deep_hdf5 --data_root /data/ImageNet/ILSVRC2012/train/ \
--norm_path $HOME2/BigGAN-PyTorch/WT64_norm_values.npz \
--num_G_accumulations 2 --num_D_accumulations 2 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_ch 128 --D_ch 128 \
--G_depth 2 --D_depth 2 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 128 --shared_dim 128 \
--ema --use_ema --ema_start 20000 --G_eval_mode \
--test_every 2000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler --num_epochs=100 \
--resume --load_weights copy1 \
--experiment_name BigGAN_WT64_hdf5_seed0_Gch96_Dch96_bs256_nDa4_nGa4_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn32_Dattn32_Gshared_hier_ema/ \
--G_batch_size 512 --sample_trunc_curves 0.05_0.05_1.0 --z_var 1.0 
#--sample_class_rejection
echo "Run completed at:- "
date
