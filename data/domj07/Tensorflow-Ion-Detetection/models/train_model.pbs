#! /bin/bash

#PBS -P QCL_PT
#PBS -l select=1:ncpus=1:ngpus=1:mem=30GB
#PBS -l walltime=24:00:00
#PBS -q defaultQ

module load cuda/10.0.130
module load openmpi-gcc/3.1.3

source miniconda/bin/activate training

cd models/

python research/object_detection/model_main_tf2.py --model_dir="/project/RDS-FSC-QCL_PT-RW/rcnn_model_data/" --num_train_steps=200000  --sample_1_of_n_eval_examples=1  --pipeline_config_path=pipeline.config  --alsologtostder


