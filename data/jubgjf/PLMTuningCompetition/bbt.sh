#!/bin/bash

#SBATCH -J bbt                                # 作业名
#SBATCH -o slurm-%j.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j.err                       # 屏幕上的错误输出文件重定向到 slurm-%j.err , %j 会替换成jobid
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 10:00:00                           # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

# source ~/.bashrc
source ~/.local/bin/miniconda3/etc/profile.d/conda.sh

# 设置运行环境
conda activate bbt

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python bbt.py --seed 8 --task_name 'sst2'
python bbt.py --seed 8 --task_name 'qnli'
python bbt.py --seed 8 --task_name 'qqp'
python bbt.py --seed 8 --task_name 'snli'
python bbt.py --seed 8 --task_name 'dbpedia'

#------ -w gpu29                              # 指定运行作业的节点是 gpu16，若不填写系统自动分配节点
