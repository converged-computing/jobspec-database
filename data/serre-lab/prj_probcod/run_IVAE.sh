#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J SVI_seeds
#SBATCH -p gpu 
#SBATCH -o logs/%x_%A_%a_%J.out
#SBATCH -e logs/%x_%A_%a_%J.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=12
#SBATCH --account=carney-tserre-condo

# SBATCH -C quadrortx
# SBATCH --array=6,10-14


module load anaconda/3-5.2.0
source activate py36


## architecture

# type=IVAE
type=SVI

hdim1=512
hdim2=256
zdim=2

activation_function=tanh
layer=fc
decoder_type='gaussian'


# declare -a beta_list=(0 0.5 1 1.5 2 2.5)
# beta_list=(0 0.5 1 1.5 2 2.5)
# beta_list=(1.5 2 2.5)
# beta_list=(0 0.5 1 1.5 2)
beta_list=(1.5)
# beta_list=(1)

#(0 0.5 1 1.5 2 2.5)

seed=$SLURM_ARRAY_TASK_ID
# seed=0
# beta=${beta_list[$SLURM_ARRAY_TASK_ID]}

## inference
svi_lr=1e-2
#svi_lr=1e-4
nb_it=100
#nb_it=10000
svi_optimizer=Adam

var_init=0.001

## training
lr=1e-3
nb_epoch=200
train_optimizer=Adam
# seed=$SLURM_ARRAY_TASK_ID

# device=4
verbose=1

EXP_PATH="/users/azerroug/scratch"
DATA_DIR="${EXP_PATH}/MNIST/"

path_db="../probcod_dbs/"

for beta in ${beta_list[@]}; do

# for i in {0..4}; do

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
exp_name="${NOW}_${type}_svi_lr=${svi_lr}_lr=${lr}_beta=${beta}_nb_it=${nb_it}_[${hdim1},${hdim2},${zdim}]_af=${activation_function}_layer=${layer}_decoder=${decoder_type}_varinit=${var_init}_seed=${seed}"
path="${EXP_PATH}/prj_probcod_exps/$exp_name"
rm -rf $path

echo $exp_name

# CUDA_VISIBLE_DEVICES=$device \
# nohup \
python train_vae.py \
    --exp_name $exp_name \
    --path_db $path_db \
    --type $type \
    --svi_lr $svi_lr \
    --nb_it $nb_it \
    --nb_epoch $nb_epoch \
    --lr $lr \
    --path $path \
    --arch $hdim1 $hdim2 \
    --z_dim $zdim \
    --activation_function $activation_function \
    --layer $layer \
    --var_init $var_init \
    --svi_optimizer $svi_optimizer \
    --train_optimizer $train_optimizer \
    --seed $seed \
    --beta $beta \
    --decoder_type $decoder_type \
    --verbose $verbose \
    --data_dir $DATA_DIR \
    # > logs/${exp_name}.log 2>&1 &

done

