#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=t1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:k80
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4508@nyu.edu

module purge
module load python/intel/2.7.12
module load pytorch/0.2.0_1
module load tensorboard_logger/0.0.3
module load scikit-learn/intel/0.18.1


python master_train_script.py --train_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top3_train_data.pkl --val_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top3_valid_data.pkl --model_dir /scratch/ag4508/nlp/mimic --attention 0 --num_workers 12 --embed_path /scratch/ag4508/nlp/stsp_model.tsv --num_epochs 15 --exp_name attention1_50_content4_top3_stsp --use_starspace 1 --multilabel 1 --batch_size 8

#python master_train_script.py --train_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top5_train_data.pkl --val_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top5_valid_data.pkl --model_dir /scratch/ag4508/nlp/mimic --attention 0 --num_workers 12 --embed_path /scratch/ag4508/nlp/stsp_model.tsv --num_epochs 15 --exp_name attention1_50_content4_top5_stsp --use_starspace 1 --multilabel 1 --batch_size 8

#python master_train_script.py --train_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top10_train_data.pkl --val_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top10_valid_data.pkl --model_dir /scratch/ag4508/nlp/mimic --attention 0 --num_workers 12 --embed_path /scratch/ag4508/nlp/stsp_model.tsv --num_epochs 15 --exp_name attention1_50_content4_top10_stsp --use_starspace 1 --multilabel 1 --batch_size 8

#python master_train_script.py --train_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top100_train_data.pkl --val_path /scratch/ag4508/nlp/mimic/50codesL5_UNK_content_4_top100_valid_data.pkl --model_dir /scratch/ag4508/nlp/mimic --attention 0 --num_workers 12 --embed_path /scratch/ag4508/nlp/stsp_model.tsv --num_epochs 15 --exp_name attention1_50_content4_top100_stsp --use_starspace 1 --multilabel 1 --batch_size 8
