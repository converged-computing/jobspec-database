#!/bin/bash
#PBS -N bdl-travel-swag-interaction
#PBS -o ./log/parallel/travel/4rounds-swag/5e-5-20-0.001/4rounds-5epochs-mp.output
#PBS -e ./log/parallel/travel/4rounds-swag/5e-5-20-0.001/4rounds-5epochs-mp.err
#PBS -l select=1:ncpus=1:ngpus=1:mem=30G
#PBS -l walltime=2:50:00
#PBS -J 0-4
cd "${PBS_O_WORKDIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo PBS job ID is "${PBS_JOBID}"
echo This jobs runs on the following machines:
echo $(cat "${PBS_NODEFILE}" | uniq)
echo PBS ARRAY ID: ${PBS_ARRAY_ID}
num_q=144
start_q=`expr $PBS_ARRAY_INDEX \* $num_q`
echo start question id is $start_q

module add lang/python/anaconda/pytorch
# pip install --upgrade torch
# module add lang/python/anaconda/3.7-2019.03
# conda create -n env python=3.7 
# source activate env 
# pip install pandas numpy torch==1.6.0 tqdm transformers --user
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/
# module add lang/python/anaconda/3.7.3-2019.03-tensorflow-2.0
# module add lang/python/anaconda/3.7-2019.10

fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --sample_nums 20 --wd 0.001 --lr_init 5e-5 --topic travel --n_iter_rounds 4 --batch_size 1 --save_dir /work/zu20361/BDL/model/ --model_name swag_bert_6epoch_1e-4-16-0.01 --interactive True --test_mode valid_interaction`
echo "$fake" > ./log/parallel/travel/4rounds-swag/5e-5-20-0.001/start-${start_q}.output
# time python3 main.py --epochs 2 --topic cooking --n_iter_rounds 8 --batch_size 10 --do_test True --model_name 8rounds-1 --interactive True --pretrained_model roberta-base
# mv ./log/parallel/4rounds-vallina-mp.output ./log/parallel/${PBS_JOBID}.output
# mv ./log/parallel/4rounds-vallina-mp.err ./log/parallel/${PBS_JOBID}.err