#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:09:00

# set name of job
#SBATCH --job-name=pytorch_test

#SBATCH --error=pytorch_test.err

#SBATCH --output=pytorch_test.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#module load pytorch


#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

#python Second_order_Receptive_field.py --experiment 2 --optimiser "kfac"
#
#python Second_order_Receptive_field.py --experiment 2 --optimiser "sam"
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#python -c "import os; print(os.environ)"
#printf "Start Test \n"
#python test_backwards.py
#printf "End Test \n"
#module load pytorch
#echo "After loading the pytorch module"
#which python
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#python -c "import os; print(os.environ)"
#printf "Start Test \n"
#python test_backwards.py
#printf "End Test \n"
##echo "============ 2 workers ============================"
##python hao_models_pruning_test.py --workers 2
#echo "============ 4 workers ============================"
#python hao_models_pruning_test.py --workers 4
#echo "============ 8 workers ============================"
#python hao_models_pruning_test.py --workers 2 --experiment 2 --model $1
#echo "============ 16 workers ============================"
#python hao_models_pruning_test.py --workers 16
#echo "============ 32 workers ============================"
#python hao_models_pruning_test.py --workers 32

echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
#eval "$(conda shell.bash hook)"
#conda activate work
which python
export LD_LIBRARY_PATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#l=$(which python)
#
#lib_path_of_current_enviroment="${l%%"bin/python"}"
#echo "Ld library ${lib_path_of_current_enviroment}"
#export LD_LIBRARY_PATH="$lib_path_of_current_enviroment/lib":$LD_LIBRARY_PATH

python -c "import os; print(os.environ)"
#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY

#python -c "import os; print(os.environ)"

#python train_CIFAR10.py  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9

#python train_CIFAR10.py --resume --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --batch_size 128  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --resume_solution "${10}"

#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_3_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_42.89.pth"

#############################################################
#     Train model
#############################################################

# With FFCV
python train_CIFAR10.py --ffcv --record_time --record_flops  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --ffcv_train "${10}" --ffcv_val "${11}"

# Without FFCV
#python train_CIFAR10.py --record_time --record_flops  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9

#############################################################
#     One shot with specific pruning rate results
#############################################################

#python prune_models.py --name "recording_200_ffcv" --ffcv --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8

#############################################################
#   Soup Idea applied to stochastic pruning
#############################################################

#python main.py --experiment 1 --batch_size 518 --modeltype "alternative" --pruner "global" --population 5 --epochs 10 --pruning_rate  $1 --architecture $2 --sigma $3 --dataset $4
