#!/bin/bash

#SBATCH -G 1
#SBATCH --nodelist=ice[183,192,193]
#SBATCH -t 3-72:00
#SBATCH --array 61-127%200
#SBATCH --mem 100G
#SBATCH --chdir=/home/jrick6/repos/data
#SBATCH --job-name=gen_100_shard
#SBATCH --output=/home/jrick6/repos/data/logs/imagenet_subset_custom/per_shard/%x.%A.%a.out

if [[ ${SLURM_ARRAY_TASK_ID} -le 0 ]]; then
    echo "shard 00000"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00000-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00000-of-00128/imagenet2012_subset-train.tfrecord-00000-of-00128" \
        --input_id 0 780 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 1 ]]; then
    echo "shard 00001"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00001-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00001-of-00128/imagenet2012_subset-train.tfrecord-00001-of-00128" \
        --input_id 779 1559 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 2 ]]; then
    echo "shard 00002"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00002-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00002-of-00128/imagenet2012_subset-train.tfrecord-00002-of-00128" \
        --input_id 1559 2339 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 3 ]]; then
    echo "shard 00003"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00003-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00003-of-00128/imagenet2012_subset-train.tfrecord-00003-of-00128" \
        --input_id 2339 3119 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 4 ]]; then
    echo "shard 00004"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00004-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00004-of-00128/imagenet2012_subset-train.tfrecord-00004-of-00128" \
        --input_id 3119 3899 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 5 ]]; then
    echo "shard 00005"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00005-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00005-of-00128/imagenet2012_subset-train.tfrecord-00005-of-00128" \
        --input_id 3898 4678 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 6 ]]; then
    echo "shard 00006"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00006-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00006-of-00128/imagenet2012_subset-train.tfrecord-00006-of-00128" \
        --input_id 4678 5458 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 7 ]]; then
    echo "shard 00007"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00007-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00007-of-00128/imagenet2012_subset-train.tfrecord-00007-of-00128" \
        --input_id 5458 6238 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 8 ]]; then
    echo "shard 00008"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00008-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00008-of-00128/imagenet2012_subset-train.tfrecord-00008-of-00128" \
        --input_id 6237 7017 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 9 ]]; then
    echo "shard 00009"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00009-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00009-of-00128/imagenet2012_subset-train.tfrecord-00009-of-00128" \
        --input_id 7017 7797 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 10 ]]; then
    echo "shard 00010"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00010-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00010-of-00128/imagenet2012_subset-train.tfrecord-00010-of-00128" \
        --input_id 7797 8577 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 11 ]]; then
    echo "shard 00011"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00011-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00011-of-00128/imagenet2012_subset-train.tfrecord-00011-of-00128" \
        --input_id 8576 9356 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 12 ]]; then
    echo "shard 00012"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00012-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00012-of-00128/imagenet2012_subset-train.tfrecord-00012-of-00128" \
        --input_id 9356 10136 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 13 ]]; then
    echo "shard 00013"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00013-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00013-of-00128/imagenet2012_subset-train.tfrecord-00013-of-00128" \
        --input_id 10136 10916 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 14 ]]; then
    echo "shard 00014"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00014-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00014-of-00128/imagenet2012_subset-train.tfrecord-00014-of-00128" \
        --input_id 10916 11696 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 15 ]]; then
    echo "shard 00015"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00015-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00015-of-00128/imagenet2012_subset-train.tfrecord-00015-of-00128" \
        --input_id 11695 12475 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 16 ]]; then
    echo "shard 00016"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00016-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00016-of-00128/imagenet2012_subset-train.tfrecord-00016-of-00128" \
        --input_id 12475 13255 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 17 ]]; then
    echo "shard 00017"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00017-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00017-of-00128/imagenet2012_subset-train.tfrecord-00017-of-00128" \
        --input_id 13255 14035 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 18 ]]; then
    echo "shard 00018"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00018-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00018-of-00128/imagenet2012_subset-train.tfrecord-00018-of-00128" \
        --input_id 14034 14814 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 19 ]]; then
    echo "shard 00019"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00019-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00019-of-00128/imagenet2012_subset-train.tfrecord-00019-of-00128" \
        --input_id 14814 15594 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 20 ]]; then
    echo "shard 00020"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00020-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00020-of-00128/imagenet2012_subset-train.tfrecord-00020-of-00128" \
        --input_id 15594 16374 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 21 ]]; then
    echo "shard 00021"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00021-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00021-of-00128/imagenet2012_subset-train.tfrecord-00021-of-00128" \
        --input_id 16373 17153 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 22 ]]; then
    echo "shard 00022"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00022-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00022-of-00128/imagenet2012_subset-train.tfrecord-00022-of-00128" \
        --input_id 17153 17933 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 23 ]]; then
    echo "shard 00023"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00023-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00023-of-00128/imagenet2012_subset-train.tfrecord-00023-of-00128" \
        --input_id 17933 18713 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 24 ]]; then
    echo "shard 00024"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00024-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00024-of-00128/imagenet2012_subset-train.tfrecord-00024-of-00128" \
        --input_id 18713 19493 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 25 ]]; then
    echo "shard 00025"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00025-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00025-of-00128/imagenet2012_subset-train.tfrecord-00025-of-00128" \
        --input_id 19492 20272 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 26 ]]; then
    echo "shard 00026"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00026-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00026-of-00128/imagenet2012_subset-train.tfrecord-00026-of-00128" \
        --input_id 20272 21052 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 27 ]]; then
    echo "shard 00027"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00027-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00027-of-00128/imagenet2012_subset-train.tfrecord-00027-of-00128" \
        --input_id 21052 21832 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 28 ]]; then
    echo "shard 00028"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00028-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00028-of-00128/imagenet2012_subset-train.tfrecord-00028-of-00128" \
        --input_id 21831 22611 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 29 ]]; then
    echo "shard 00029"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00029-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00029-of-00128/imagenet2012_subset-train.tfrecord-00029-of-00128" \
        --input_id 22611 23391 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 30 ]]; then
    echo "shard 00030"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00030-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00030-of-00128/imagenet2012_subset-train.tfrecord-00030-of-00128" \
        --input_id 23391 24171 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 31 ]]; then
    echo "shard 00031"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00031-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00031-of-00128/imagenet2012_subset-train.tfrecord-00031-of-00128" \
        --input_id 24170 24950 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 32 ]]; then
    echo "shard 00032"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00032-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00032-of-00128/imagenet2012_subset-train.tfrecord-00032-of-00128" \
        --input_id 24950 25730 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 33 ]]; then
    echo "shard 00033"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00033-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00033-of-00128/imagenet2012_subset-train.tfrecord-00033-of-00128" \
        --input_id 25730 26510 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 34 ]]; then
    echo "shard 00034"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00034-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00034-of-00128/imagenet2012_subset-train.tfrecord-00034-of-00128" \
        --input_id 26510 27290 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 35 ]]; then
    echo "shard 00035"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00035-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00035-of-00128/imagenet2012_subset-train.tfrecord-00035-of-00128" \
        --input_id 27289 28069 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 36 ]]; then
    echo "shard 00036"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00036-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00036-of-00128/imagenet2012_subset-train.tfrecord-00036-of-00128" \
        --input_id 28069 28849 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 37 ]]; then
    echo "shard 00037"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00037-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00037-of-00128/imagenet2012_subset-train.tfrecord-00037-of-00128" \
        --input_id 28849 29629 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 38 ]]; then
    echo "shard 00038"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00038-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00038-of-00128/imagenet2012_subset-train.tfrecord-00038-of-00128" \
        --input_id 29628 30408 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 39 ]]; then
    echo "shard 00039"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00039-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00039-of-00128/imagenet2012_subset-train.tfrecord-00039-of-00128" \
        --input_id 30408 31188 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 40 ]]; then
    echo "shard 00040"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00040-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00040-of-00128/imagenet2012_subset-train.tfrecord-00040-of-00128" \
        --input_id 31188 31968 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 41 ]]; then
    echo "shard 00041"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00041-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00041-of-00128/imagenet2012_subset-train.tfrecord-00041-of-00128" \
        --input_id 31968 32748 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 42 ]]; then
    echo "shard 00042"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00042-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00042-of-00128/imagenet2012_subset-train.tfrecord-00042-of-00128" \
        --input_id 32747 33527 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 43 ]]; then
    echo "shard 00043"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00043-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00043-of-00128/imagenet2012_subset-train.tfrecord-00043-of-00128" \
        --input_id 33527 34307 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 44 ]]; then
    echo "shard 00044"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00044-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00044-of-00128/imagenet2012_subset-train.tfrecord-00044-of-00128" \
        --input_id 34307 35087 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 45 ]]; then
    echo "shard 00045"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00045-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00045-of-00128/imagenet2012_subset-train.tfrecord-00045-of-00128" \
        --input_id 35086 35866 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 46 ]]; then
    echo "shard 00046"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00046-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00046-of-00128/imagenet2012_subset-train.tfrecord-00046-of-00128" \
        --input_id 35866 36646 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 47 ]]; then
    echo "shard 00047"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00047-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00047-of-00128/imagenet2012_subset-train.tfrecord-00047-of-00128" \
        --input_id 36646 37426 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 48 ]]; then
    echo "shard 00048"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00048-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00048-of-00128/imagenet2012_subset-train.tfrecord-00048-of-00128" \
        --input_id 37425 38205 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 49 ]]; then
    echo "shard 00049"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00049-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00049-of-00128/imagenet2012_subset-train.tfrecord-00049-of-00128" \
        --input_id 38205 38985 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 50 ]]; then
    echo "shard 00050"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00050-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00050-of-00128/imagenet2012_subset-train.tfrecord-00050-of-00128" \
        --input_id 38985 39765 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 51 ]]; then
    echo "shard 00051"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00051-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00051-of-00128/imagenet2012_subset-train.tfrecord-00051-of-00128" \
        --input_id 39765 40545 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 52 ]]; then
    echo "shard 00052"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00052-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00052-of-00128/imagenet2012_subset-train.tfrecord-00052-of-00128" \
        --input_id 40544 41324 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 53 ]]; then
    echo "shard 00053"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00053-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00053-of-00128/imagenet2012_subset-train.tfrecord-00053-of-00128" \
        --input_id 41324 42104 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 54 ]]; then
    echo "shard 00054"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00054-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00054-of-00128/imagenet2012_subset-train.tfrecord-00054-of-00128" \
        --input_id 42104 42884 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 55 ]]; then
    echo "shard 00055"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00055-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00055-of-00128/imagenet2012_subset-train.tfrecord-00055-of-00128" \
        --input_id 42883 43663 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 56 ]]; then
    echo "shard 00056"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00056-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00056-of-00128/imagenet2012_subset-train.tfrecord-00056-of-00128" \
        --input_id 43663 44443 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 57 ]]; then
    echo "shard 00057"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00057-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00057-of-00128/imagenet2012_subset-train.tfrecord-00057-of-00128" \
        --input_id 44443 45223 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 58 ]]; then
    echo "shard 00058"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00058-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00058-of-00128/imagenet2012_subset-train.tfrecord-00058-of-00128" \
        --input_id 45222 46002 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 59 ]]; then
    echo "shard 00059"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00059-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00059-of-00128/imagenet2012_subset-train.tfrecord-00059-of-00128" \
        --input_id 46002 46782 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 60 ]]; then
    echo "shard 00060"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00060-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00060-of-00128/imagenet2012_subset-train.tfrecord-00060-of-00128" \
        --input_id 46782 47562 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 61 ]]; then
    echo "shard 00061"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00061-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00061-of-00128/imagenet2012_subset-train.tfrecord-00061-of-00128" \
        --input_id 47562 48342 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 62 ]]; then
    echo "shard 00062"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00062-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00062-of-00128/imagenet2012_subset-train.tfrecord-00062-of-00128" \
        --input_id 48341 49121 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 63 ]]; then
    echo "shard 00063"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00063-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00063-of-00128/imagenet2012_subset-train.tfrecord-00063-of-00128" \
        --input_id 49121 49901 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 64 ]]; then
    echo "shard 00064"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00064-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00064-of-00128/imagenet2012_subset-train.tfrecord-00064-of-00128" \
        --input_id 49901 50681 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 65 ]]; then
    echo "shard 00065"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00065-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00065-of-00128/imagenet2012_subset-train.tfrecord-00065-of-00128" \
        --input_id 50680 51460 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 66 ]]; then
    echo "shard 00066"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00066-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00066-of-00128/imagenet2012_subset-train.tfrecord-00066-of-00128" \
        --input_id 51460 52240 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 67 ]]; then
    echo "shard 00067"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00067-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00067-of-00128/imagenet2012_subset-train.tfrecord-00067-of-00128" \
        --input_id 52240 53020 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 68 ]]; then
    echo "shard 00068"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00068-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00068-of-00128/imagenet2012_subset-train.tfrecord-00068-of-00128" \
        --input_id 53020 53800 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 69 ]]; then
    echo "shard 00069"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00069-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00069-of-00128/imagenet2012_subset-train.tfrecord-00069-of-00128" \
        --input_id 53799 54579 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 70 ]]; then
    echo "shard 00070"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00070-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00070-of-00128/imagenet2012_subset-train.tfrecord-00070-of-00128" \
        --input_id 54579 55359 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 71 ]]; then
    echo "shard 00071"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00071-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00071-of-00128/imagenet2012_subset-train.tfrecord-00071-of-00128" \
        --input_id 55359 56139 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 72 ]]; then
    echo "shard 00072"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00072-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00072-of-00128/imagenet2012_subset-train.tfrecord-00072-of-00128" \
        --input_id 56138 56918 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 73 ]]; then
    echo "shard 00073"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00073-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00073-of-00128/imagenet2012_subset-train.tfrecord-00073-of-00128" \
        --input_id 56918 57698 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 74 ]]; then
    echo "shard 00074"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00074-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00074-of-00128/imagenet2012_subset-train.tfrecord-00074-of-00128" \
        --input_id 57698 58478 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 75 ]]; then
    echo "shard 00075"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00075-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00075-of-00128/imagenet2012_subset-train.tfrecord-00075-of-00128" \
        --input_id 58477 59257 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 76 ]]; then
    echo "shard 00076"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00076-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00076-of-00128/imagenet2012_subset-train.tfrecord-00076-of-00128" \
        --input_id 59257 60037 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 77 ]]; then
    echo "shard 00077"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00077-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00077-of-00128/imagenet2012_subset-train.tfrecord-00077-of-00128" \
        --input_id 60037 60817 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 78 ]]; then
    echo "shard 00078"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00078-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00078-of-00128/imagenet2012_subset-train.tfrecord-00078-of-00128" \
        --input_id 60817 61597 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 79 ]]; then
    echo "shard 00079"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00079-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00079-of-00128/imagenet2012_subset-train.tfrecord-00079-of-00128" \
        --input_id 61596 62376 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 80 ]]; then
    echo "shard 00080"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00080-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00080-of-00128/imagenet2012_subset-train.tfrecord-00080-of-00128" \
        --input_id 62376 63156 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 81 ]]; then
    echo "shard 00081"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00081-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00081-of-00128/imagenet2012_subset-train.tfrecord-00081-of-00128" \
        --input_id 63156 63936 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 82 ]]; then
    echo "shard 00082"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00082-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00082-of-00128/imagenet2012_subset-train.tfrecord-00082-of-00128" \
        --input_id 63935 64715 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 83 ]]; then
    echo "shard 00083"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00083-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00083-of-00128/imagenet2012_subset-train.tfrecord-00083-of-00128" \
        --input_id 64715 65495 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 84 ]]; then
    echo "shard 00084"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00084-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00084-of-00128/imagenet2012_subset-train.tfrecord-00084-of-00128" \
        --input_id 65495 66275 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 85 ]]; then
    echo "shard 00085"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00085-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00085-of-00128/imagenet2012_subset-train.tfrecord-00085-of-00128" \
        --input_id 66274 67054 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 86 ]]; then
    echo "shard 00086"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00086-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00086-of-00128/imagenet2012_subset-train.tfrecord-00086-of-00128" \
        --input_id 67054 67834 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 87 ]]; then
    echo "shard 00087"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00087-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00087-of-00128/imagenet2012_subset-train.tfrecord-00087-of-00128" \
        --input_id 67834 68614 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 88 ]]; then
    echo "shard 00088"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00088-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00088-of-00128/imagenet2012_subset-train.tfrecord-00088-of-00128" \
        --input_id 68614 69394 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 89 ]]; then
    echo "shard 00089"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00089-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00089-of-00128/imagenet2012_subset-train.tfrecord-00089-of-00128" \
        --input_id 69393 70173 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 90 ]]; then
    echo "shard 00090"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00090-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00090-of-00128/imagenet2012_subset-train.tfrecord-00090-of-00128" \
        --input_id 70173 70953 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 91 ]]; then
    echo "shard 00091"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00091-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00091-of-00128/imagenet2012_subset-train.tfrecord-00091-of-00128" \
        --input_id 70953 71733 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 92 ]]; then
    echo "shard 00092"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00092-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00092-of-00128/imagenet2012_subset-train.tfrecord-00092-of-00128" \
        --input_id 71732 72512 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 93 ]]; then
    echo "shard 00093"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00093-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00093-of-00128/imagenet2012_subset-train.tfrecord-00093-of-00128" \
        --input_id 72512 73292 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 94 ]]; then
    echo "shard 00094"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00094-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00094-of-00128/imagenet2012_subset-train.tfrecord-00094-of-00128" \
        --input_id 73292 74072 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 95 ]]; then
    echo "shard 00095"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00095-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00095-of-00128/imagenet2012_subset-train.tfrecord-00095-of-00128" \
        --input_id 74072 74852 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 96 ]]; then
    echo "shard 00096"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00096-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00096-of-00128/imagenet2012_subset-train.tfrecord-00096-of-00128" \
        --input_id 74851 75631 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 97 ]]; then
    echo "shard 00097"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00097-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00097-of-00128/imagenet2012_subset-train.tfrecord-00097-of-00128" \
        --input_id 75631 76411 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 98 ]]; then
    echo "shard 00098"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00098-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00098-of-00128/imagenet2012_subset-train.tfrecord-00098-of-00128" \
        --input_id 76411 77191 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 99 ]]; then
    echo "shard 00099"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00099-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00099-of-00128/imagenet2012_subset-train.tfrecord-00099-of-00128" \
        --input_id 77190 77970 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 100 ]]; then
    echo "shard 00100"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00100-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00100-of-00128/imagenet2012_subset-train.tfrecord-00100-of-00128" \
        --input_id 77970 78750 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 101 ]]; then
    echo "shard 00101"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00101-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00101-of-00128/imagenet2012_subset-train.tfrecord-00101-of-00128" \
        --input_id 78750 79530 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 102 ]]; then
    echo "shard 00102"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00102-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00102-of-00128/imagenet2012_subset-train.tfrecord-00102-of-00128" \
        --input_id 79529 80309 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 103 ]]; then
    echo "shard 00103"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00103-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00103-of-00128/imagenet2012_subset-train.tfrecord-00103-of-00128" \
        --input_id 80309 81089 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 104 ]]; then
    echo "shard 00104"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00104-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00104-of-00128/imagenet2012_subset-train.tfrecord-00104-of-00128" \
        --input_id 81089 81869 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 105 ]]; then
    echo "shard 00105"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00105-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00105-of-00128/imagenet2012_subset-train.tfrecord-00105-of-00128" \
        --input_id 81869 82649 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 106 ]]; then
    echo "shard 00106"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00106-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00106-of-00128/imagenet2012_subset-train.tfrecord-00106-of-00128" \
        --input_id 82648 83428 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 107 ]]; then
    echo "shard 00107"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00107-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00107-of-00128/imagenet2012_subset-train.tfrecord-00107-of-00128" \
        --input_id 83428 84208 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 108 ]]; then
    echo "shard 00108"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00108-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00108-of-00128/imagenet2012_subset-train.tfrecord-00108-of-00128" \
        --input_id 84208 84988 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 109 ]]; then
    echo "shard 00109"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00109-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00109-of-00128/imagenet2012_subset-train.tfrecord-00109-of-00128" \
        --input_id 84987 85767 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 110 ]]; then
    echo "shard 00110"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00110-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00110-of-00128/imagenet2012_subset-train.tfrecord-00110-of-00128" \
        --input_id 85767 86547 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 111 ]]; then
    echo "shard 00111"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00111-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00111-of-00128/imagenet2012_subset-train.tfrecord-00111-of-00128" \
        --input_id 86547 87327 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 112 ]]; then
    echo "shard 00112"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00112-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00112-of-00128/imagenet2012_subset-train.tfrecord-00112-of-00128" \
        --input_id 87326 88106 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 113 ]]; then
    echo "shard 00113"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00113-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00113-of-00128/imagenet2012_subset-train.tfrecord-00113-of-00128" \
        --input_id 88106 88886 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 114 ]]; then
    echo "shard 00114"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00114-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00114-of-00128/imagenet2012_subset-train.tfrecord-00114-of-00128" \
        --input_id 88886 89666 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 115 ]]; then
    echo "shard 00115"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00115-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00115-of-00128/imagenet2012_subset-train.tfrecord-00115-of-00128" \
        --input_id 89666 90446 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 116 ]]; then
    echo "shard 00116"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00116-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00116-of-00128/imagenet2012_subset-train.tfrecord-00116-of-00128" \
        --input_id 90445 91225 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 117 ]]; then
    echo "shard 00117"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00117-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00117-of-00128/imagenet2012_subset-train.tfrecord-00117-of-00128" \
        --input_id 91225 92005 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 118 ]]; then
    echo "shard 00118"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00118-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00118-of-00128/imagenet2012_subset-train.tfrecord-00118-of-00128" \
        --input_id 92005 92785 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 119 ]]; then
    echo "shard 00119"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00119-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00119-of-00128/imagenet2012_subset-train.tfrecord-00119-of-00128" \
        --input_id 92784 93564 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 120 ]]; then
    echo "shard 00120"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00120-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00120-of-00128/imagenet2012_subset-train.tfrecord-00120-of-00128" \
        --input_id 93564 94344 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 121 ]]; then
    echo "shard 00121"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00121-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00121-of-00128/imagenet2012_subset-train.tfrecord-00121-of-00128" \
        --input_id 94344 95124 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 122 ]]; then
    echo "shard 00122"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00122-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00122-of-00128/imagenet2012_subset-train.tfrecord-00122-of-00128" \
        --input_id 95123 95903 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 123 ]]; then
    echo "shard 00123"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00123-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00123-of-00128/imagenet2012_subset-train.tfrecord-00123-of-00128" \
        --input_id 95903 96683 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 124 ]]; then
    echo "shard 00124"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00124-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00124-of-00128/imagenet2012_subset-train.tfrecord-00124-of-00128" \
        --input_id 96683 97463 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 125 ]]; then
    echo "shard 00125"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00125-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00125-of-00128/imagenet2012_subset-train.tfrecord-00125-of-00128" \
        --input_id 97463 98243 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 126 ]]; then
    echo "shard 00126"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00126-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00126-of-00128/imagenet2012_subset-train.tfrecord-00126-of-00128" \
        --input_id 98242 99022 \
        --use_range \
        --half
elif [[ ${SLURM_ARRAY_TASK_ID} -le 127 ]]; then
    echo "shard 00127"
    /home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
        -tfp "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id/custom_100c_1000/5.0.0/imagenet2012_subset-train.tfrecord-00127-of-00128" \
        -o "/home/jrick6/tensorflow_datasets/imagenet2012_subset_id_variations/custom_100c_1000/5.0.0/dir_imagenet2012_subset-train.tfrecord-00127-of-00128/imagenet2012_subset-train.tfrecord-00127-of-00128" \
        --input_id 99022 99802 \
        --use_range \
        --half
fi