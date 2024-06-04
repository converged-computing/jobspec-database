#!/bin/bash
#SBATCH --job-name=VisualImag
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=4
#SBATCH --output=slurm-%j-%x.out

module load anaconda/3
module load cuda/10.1

source /home/d.sorge/.bashrc
conda activate venv

python train_derived.py \
-gen_bs 16 \
-dis_bs 16 \
--accumulated_times 4 \
--g_accumulated_times 4 \
--dist-url 'tcp://localhost:10641' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--dataset eegdataset \
--bottom_width 8 \
--img_size 64 \
--max_iter 500000 \
--gen_model ViT_custom_local544444_256_rp \
--dis_model ViT_scale2 \
--g_window_size 16 \
--d_window_size 16 \
--g_norm pn \
--df_dim 384 \
--d_depth 3 \
--g_depth 5,4,2,2 \
--latent_dim 56320 \
--gf_dim 256 \
--num_workers 0 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 10 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 4 \
--val_freq 10 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--diff_aug filter,translation,erase_ratio,color,hue \
--ema 0.995 \
--exp_name eeg_train
