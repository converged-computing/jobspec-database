#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=p2p_cmanaus 	# job name
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=Result-%x.%j.out
#SBATCH --error=Result-%x.%j.err
# Runtime of this job
#SBATCH --array=0
python3 train.py --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --custom_images_path /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model pix2pix --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataset_mode cmanausskfold --project p2p_tunning_cmanaus --name cmanaus_test_0_sort_$SLURM_ARRAY_TASK_ID --preprocess resize_and_scale_width --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 --load_size 256 --crop_size 256 --test 0 --sort $SLURM_ARRAY_TASK_ID --use_wandb --wandb_entity otavares --wandb_fold_id v1_cmanaus_tunning_test_0_sort_$SLURM_ARRAY_TASK_ID





