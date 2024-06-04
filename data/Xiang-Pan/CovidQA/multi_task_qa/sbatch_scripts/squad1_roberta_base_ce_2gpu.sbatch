#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=168:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=xiangpan@nyu.edu


# gr037-ib0
# module purge
# module load anaconda3/2020.11
# conda activate torch-env
conda activate covidqa
# MASTER=`/bin/hostname -s`
# SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
# ADD="tcp://$MASTER:6001"

echo $(pwd)
./scripts/squad1/roberta_base_ce_2gpu.sh



# python main_dist.py --ori_dataset_name IMAGENET --ori_net_name ResNet152 --num_epochs 1000 --batch_size 768 --train_normalization True --pgd_norm_mode True  --pgd_strength 8_0.16_1_40 --parm_init -0.2 --threshold -100 --myce_pred_normalization True --test_epoch 10 --test_epoch 5 --save_model True --pgd_save_threshold 45 --fit_msg ImageNet_search_0  --save_path .  --dist_url $ADD --dist_backend 'nccl' --multiprocessing_distributed --world_size 1 --rank 0 --gpu_DataParallel False

