#!/bin/bash
#SBATCH --ntasks=1 
#SBATCH -t 72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:4
#SBATCH -A gutintelligencelab
#SBATCH --job-name=Baseline_Ralis_CAMVID_Only
#SBATCH --output=Baseline_Ralis_CAMVID_Only_%A_%a.out
#SBATCH --error=Baseline_Ralis_CAMVID_Only_%A_%a.err 
#SBATCH --mail-type=end
#SBATCH --mail-user=pm2kb@virginia.edu

module purge
module --ignore-cache load anaconda/2019.10-py3.7
module --ignore-cache load singularity/3.5.2

ckpt_path='/scratch/pm2kb/ckpt_seg'
data_path='/home/pm2kb/RL_proj/SegNet'
    
#### Camvid ####

for al_algorithm in 'random'
    do
    for budget in 480 720 960 1200 1440 1920
        do
        for seed in 20 50 234 4560 12
            do
            singularity run --nv ~/pytorch-1.4.0-py37.sif /home/pm2kb/RL_proj/ralis-master/run.py --exp-name 'baseline_camvid_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'camvid_pretrained_dt' \
            --input-size 224 224 --only-last-labeled --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
            --patience 150 --region-size 80 90 \
             --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 50 --train --test --final-test
            done
        done
    done

for al_algorithm in 'entropy' 'bald'
    do
    for budget in 480 720 960 1200 1440 1920
        do
        for seed in 20 50 234 4560 12
            do
            singularity run --nv ~/pytorch-1.4.0-py37.sif /home/pm2kb/RL_proj/ralis-master/run.py --exp-name 'baseline_camvid_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'camvid_pretrained_dt' \
            --input-size 224 224 --only-last-labeled --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
            --patience 150 --region-size 80 90 \
             --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 10 --train --test --final-test
            done
        done
    done

