#!/bin/bash
#SBATCH --job-name=train_adv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lucia.trastulla@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/train_%j.out.log
#SBATCH --error=logs/train_%j.err.log
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

module load cuda11.7/toolkit/11.7.1
module load cudnn8.5-cuda11.7/8.5.0.96
source ${HOME}/miniforge3/etc/profile.d/conda.sh
conda activate VAE_momics_v2

GROUP_PATH='/group/iorio/lucia/'

# # test 1
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000_tcgaonly.csv.gz' \
#     --ngene='var1000' \
#     --samples='tcgaonly' \
#     --no-norm_feat \
#     --only_shared 

# # test 2
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000_tcgaonly.csv.gz' \
#     --ngene='var1000' \
#     --samples='tcgaonly' \
#     --norm_feat \
#     --only_shared

# # test 3
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000_tcgaonly.csv.gz' \
#     --ngene='var5000' \
#     --samples='tcgaonly' \
#     --no-norm_feat \
#     --only_shared

# # test 4
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000_tcgaonly.csv.gz' \
#     --ngene='var5000' \
#     --samples='tcgaonly' \
#     --norm_feat \
#     --only_shared

# # test 5
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
#     --ngene='all' \
#     --samples='tcgaonly' \
#     --no-norm_feat \
#     --only_shared

# # test 6
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
#     --ngene='all' \
#     --samples='tcgaonly' \
#     --norm_feat \
#     --only_shared

# # all samples
# # test 1
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000.csv.gz' \
#     --ngene='var1000' \
#     --no-norm_feat \
#     --only_shared

# # test 2
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000.csv.gz' \
#     --ngene='var1000' \
#     --norm_feat \
#     --only_shared
 
# # test 3
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000.csv.gz' \
#     --ngene='var5000' \
#     --no-norm_feat \
#     --only_shared

# # test 4
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000.csv.gz' \
#     --ngene='var5000' \
#     --norm_feat \
#     --only_shared

# # test 5
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
#     --ngene='all' \
#     --no-norm_feat \
#     --only_shared

# # test 6
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
#     --ngene='all' \
#     --norm_feat \
#     --only_shared

# ### repeat all, not using the shared encoder for the critic ###
# # test 1
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000_tcgaonly.csv.gz' \
#     --ngene='var1000' \
#     --samples='tcgaonly' \
#     --no-norm_feat \
#     --no-only_shared 

# # test 2
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000_tcgaonly.csv.gz' \
#     --ngene='var1000' \
#     --samples='tcgaonly' \
#     --norm_feat \
#     --no-only_shared

# # test 3
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000_tcgaonly.csv.gz' \
#     --ngene='var5000' \
#     --samples='tcgaonly' \
#     --no-norm_feat \
#     --no-only_shared

# # test 4
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000_tcgaonly.csv.gz' \
#     --ngene='var5000' \
#     --samples='tcgaonly' \
#     --norm_feat \
#     --no-only_shared

# # test 5
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
#     --ngene='all' \
#     --samples='tcgaonly' \
#     --no-norm_feat \
#     --no-only_shared

# # test 6
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
#     --ngene='all' \
#     --samples='tcgaonly' \
#     --norm_feat \
#     --no-only_shared

# # all samples
# # test 1
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000.csv.gz' \
#     --ngene='var1000' \
#     --no-norm_feat \
#     --no-only_shared

# # test 2
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var1000.csv.gz' \
#     --ngene='var1000' \
#     --norm_feat \
#     --no-only_shared
    
# # test 3
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000.csv.gz' \
#     --ngene='var5000' \
#     --no-norm_feat \
#     --no-only_shared

# # test 4
# python train/training.py \
#     --folder=experiment_1 \
#     --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_var5000.csv.gz' \
#     --ngene='var5000' \
#     --norm_feat \
#     --no-only_shared

# test 5
python train/training.py \
    --folder=experiment_1 \
    --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
    --ngene='all' \
    --no-norm_feat \
    --no-only_shared

# test 6
python train/training.py \
    --folder=experiment_1 \
    --gex_feature_file=${GROUP_PATH}'Multiomic_VAE/data/preprocessed/gene_expression_all.csv.gz' \
    --ngene='all' \
    --norm_feat \
    --no-only_shared
