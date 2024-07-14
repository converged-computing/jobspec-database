#!/bin/bash
#SBATCH --partition=amd-gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=60G
#SBATCH -e results/logs/slurm-%j.err
#SBATCH -o results/logs/slurm-%j.out
##SBATCH --mail-type=end
##SBATCH --mail-user=jiajingnan2222@gmail.com

module load Miniconda3
module load git


eval "$(conda shell.bash hook)"
# conda init bash

conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/logs
echo job_id is $job_id
##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
# git will not detect the current file because this file may be changed when this job was run
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh

# Passing shell variables to ssh
# https://stackoverflow.com/questions/15838927/passing-shell-variables-to-ssh
# The following code will ssh to loginnode and git commit to synchronize commits from different nodes.

# But sleep some time is required otherwise multiple commits by several experiments at the same time
# will lead to commit error: fatal: could not parse HEAD


ssh -tt jjia@nodelogin01 /bin/bash << ENDSSH
echo "Hello, I an in nodelogin01 to do some git operations."
echo $job_id

jobs="$(squeue -u jjia --sort=+i | grep [^0-9]0:[00-60] | awk '{print $1}')"  # "" to make sure multi lines assigned
echo "Total jobs in one minutes:"
echo \$jobs

accu=0
for i in \$jobs; do
    if [[ \$i -eq $job_id ]]; then
    echo start sleep ...
    sleep \$accu
    echo sleep \$accu seconds
    fi

    echo \$i
    ((accu+=5))  # self increament
    echo \$accu
done

cd data/lung_function
echo $job_id
scontrol write batch_script "${job_id}" lung_function/scripts/current_script.sh  # for the git commit latter

git add -A
sleep 2  # avoid error: fatal: Could not parse object (https://github.com/Shippable/support/issues/2932)
git rm --cached *mlrunsdb.db-journal*

git commit -m "jobid is ${job_id}"
sleep 2
git push origin master
sleep 2
exit
ENDSSH

echo "Hello, I am back in $(hostname) to run the code"

# module load Miniconda3
# module load git

conda activate py38




# stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --ct_sp='ori' --net='pointnet2_reg' --PNB=28000 --npoint_base=1024 --radius_base=60 --nsample_base=64 --batch_size=10 --mode='train' --epochs=500 --workers=6 --test_pat='random_as_ori' --target='FVC-DLCO_SB-FEV1-TLC_He' --remark="radius_base=60, correct random shuffle"

root_dir="/home/jjia/data/lung_function"
cd ${root_dir}

script_dir=${root_dir}/lung_function/scripts

pwd
which conda
conda info --envs
conda list
export PATH=$PATH:"/cm/shared/easybuild/GenuineIntel/software/git/2.38.1-GCCcore-12.2.0-nodocs/bin/git"

# shellcheck disable=SC2046
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u ${script_dir}/run.py 2>${script_dir}/${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${script_dir}/${slurm_dir}/slurm-${job_id}_${idx}_out.txt --hostname="$(hostname)" --jobid=${job_id} --epochs=100 --remark="change first conv layer and all 1,2,2, conv layer to have 2048,2,2,2 final hidden features, 100 epochs"
# idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u combined_run.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --target='DLCOc_SB-FEV1-FVC-TLC_He' --pretrained_ct='ct' --remark="combined_run,negtive add cos_loss "
# idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --net='x3d_m' --pretrained_imgnet=True --batch_size=1 --input_mode="ct_masked_by_lung" --epochs=100 --target='TLC_He' --remark="1 outputs, from pretraining" &
# wait
# idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u gcn.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --target='DLCOc_SB-FEV1-FVC-TLC_He' --lr=0.001 --batch_size=32 --epochs=100 --remark="layer_nb=1, GAT, batchNorm"


