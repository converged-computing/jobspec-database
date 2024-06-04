#!/bin/bash
#SBATCH --gpus=2 
#参数在脚本中可以加上前缀“#SBATCH”指定，和在命令参数中指定功能一致，如果脚本中的参数和命令指定的参数冲突，则命令中指定的参数优先级更高。在此处指定后可以直接sbatch ./run.sh 提交。
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境(pytorch环境需要自己部署)
module load anaconda/2021.11
source activate pt

# # 后台循环采集，每间隔 1s 采集一次GPU数据。
# # 采集的数据将输出到本地 log_[作业ID]/gpu.log 文件中
# X_LOG_DIR="log_${SLURM_JOB_ID}"
# X_GPU_LOG="${X_LOG_DIR}/gpu.log"
# mkdir "${X_LOG_DIR}"
# function gpus_collection(){
#    sleep 15
#    process=`ps -ef | grep python | grep $USER | grep -v "grep" | wc -l`
#    while [[ "${process}" > "0" ]]; do
#       sleep 1
#       nvidia-smi >> "${X_GPU_LOG}" 2>&1
#       echo "process num:${process}" >> "${X_GPU_LOG}" 2>&1
#       process=`ps -ef | grep python | grep $USER | grep -v "grep" | wc -l`
#    done
# }
# gpus_collection &

 #python程序运行，需在.py文件指定调用GPU，并设置合适的线程数，batch_size大小等
export PYTHONPATH=./
PYTHON=python
TRAIN_CODE=train_final.py
dataset=abc
exp_name=test
exp_dir=exp/${dataset}_final/${exp_name}
model_dir=${exp_dir}/model
config=config/abc/abc_debug.yaml
mkdir -p ${model_dir}
cp tool/run.sh model/pointtransformer/pointtransformer.py tool/${TRAIN_CODE} ${config} ${exp_dir}
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir}
