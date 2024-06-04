#!/bin/bash
#PBS -P CS4248_GAT
#PBS -j oe
#PBS -N CS4248_GAT
#PBS -q volta_gpu
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);

cd /hpctmp/yk/CS4248/GAT/

image="/app1/common/singularity-img/3.0.0/pytorch_1.9_cuda_11.3.0-ubuntu20.04-py38-ngc_21.04.simg"

singularity exec -e $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PYTHONPATH:/home/svu/e0741024/PyPackages/CS4248/GAT/lib/python3.8/site-packages
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

cd /hpctmp/yk/CS4248/GAT/

# train
python main.py --train data/fulltrain_1.csv --train_partition=1 --batch_size 4 --max_epochs 100 --config gat --max_sent_len 50 --encoder 4 --mode 0

# test
# python main.py --test data/balancedtest.csv --batch_size 4 --max_sent_len 50 --encoder 4 --model_file model_gat_partition_1_epoch54.t7 --mode 1
# python main.py --test data/covid_unreliable_news_cleaned_numbered.csv --batch_size 4 --max_sent_len 50 --encoder 4 --model_file model_gat_partition_1_epoch54.t7 --mode 1

EOF
