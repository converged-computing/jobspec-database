#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=2 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=23:59:59 # Run for 1 day, at most
#SBATCH --job-name=GPU-Example # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address

# Load modules
module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/anaconda/anaconda3/lib

source activate vit_ass
python3.9 --version
pip --version

#export PYTHONNOUSERSITE=1
#conda install -c intel mkl_fft

pip install -r /home/cap6411.student19/Ass2/CAP6411_Ass2/requirements_pip.txt

# python3 train.py --dataset cifar10 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_cifar10_train.txt --checkpoint_path models/checkpoints/cifar10_vit_l_16.bin
# python3 eval.py --dataset cifar10 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_cifar10.txt --pretrained_dir models/checkpoints/cifar10_vit_l_16.bin
# python3 train.py --dataset cifar10 --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_cifar10_train.txt --checkpoint_path models/checkpoints/cifar10_vit_h_14.bin
# python3 eval.py --dataset cifar10 --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_cifar10.txt --pretrained_dir models/checkpoints/cifar10_vit_h_14.bin
# python3 train.py --dataset cifar10 --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_cifar10_train.txt --checkpoint_path models/checkpoints/cifar10_bit_l_res.bin
# python3 eval.py --dataset cifar10 --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_cifar10.txt --pretrained_dir models/checkpoints/cifar10_bit_l_res.bin
# python3 train.py --dataset cifar10 --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_cifar10_train.txt --checkpoint_path models/checkpoints/cifar10_eff_net_l2.bin
# python3 eval.py --dataset cifar10 --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_cifar10.txt --pretrained_dir models/checkpoints/cifar10_eff_net_l2.bin


# python3 train.py --num_epochs 20 --dataset cifar100 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_cifar100_train.txt --checkpoint_path models/checkpoints/cifar100_vit_l_16.bin
# python3 eval.py --dataset cifar100 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_cifar100.txt --pretrained_dir models/checkpoints/cifar100_vit_l_16.bin
# python3 train.py --num_epochs 20 --dataset cifar100 --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_cifar100_train.txt --checkpoint_path models/checkpoints/cifar100_vit_h_14.bin
# python3 eval.py --dataset cifar100 --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_cifar100.txt --pretrained_dir models/checkpoints/cifar100_vit_h_14.bin
# python3 train.py --num_epochs 20 --dataset cifar100 --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_cifar100_train.txt --checkpoint_path models/checkpoints/cifar100_bit_l_res.bin
# python3 eval.py --dataset cifar100 --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_cifar100.txt --pretrained_dir models/checkpoints/cifar100_bit_l_res.bin
# python3 train.py --num_epochs 20 --dataset cifar100 --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_cifar100_train.txt --checkpoint_path models/checkpoints/cifar100_eff_net_l2.bin
# python3 eval.py --dataset cifar100 --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_cifar100.txt --pretrained_dir models/checkpoints/cifar100_eff_net_l2.bin

# python3 train.py --num_epochs 40 --dataset oxford_iiit --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_oxford_iiit_train.txt --checkpoint_path models/checkpoints/oxford_iiit_vit_l_16.bin
# python3 eval.py --dataset oxford_iiit --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_oxford_iiit.txt --pretrained_dir models/checkpoints/oxford_iiit_vit_l_16.bin
# python3 train.py --num_epochs 40 --dataset oxford_iiit --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_oxford_iiit_train.txt --checkpoint_path models/checkpoints/oxford_iiit_vit_h_14.bin
# python3 eval.py --dataset oxford_iiit --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_oxford_iiit.txt --pretrained_dir models/checkpoints/oxford_iiit_vit_h_14.bin
# python3 train.py --num_epochs 40 --dataset oxford_iiit --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_oxford_iiit_train.txt --checkpoint_path models/checkpoints/oxford_iiit_bit_l_res.bin
# python3 eval.py --dataset oxford_iiit --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_oxford_iiit.txt --pretrained_dir models/checkpoints/oxford_iiit_bit_l_res.bin
# python3 train.py --num_epochs 40 --dataset oxford_iiit --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_oxford_iiit_train.txt --checkpoint_path models/checkpoints/oxford_iiit_eff_net_l2.bin
# python3 eval.py --dataset oxford_iiit --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_oxford_iiit.txt --pretrained_dir models/checkpoints/oxford_iiit_eff_net_l2.bin

# python3 train.py --num_epochs 40 --dataset flowers102 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_flowers102_train.txt --checkpoint_path models/checkpoints/flowers102_vit_l_16.bin
# python3 eval.py --dataset flowers102 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_flowers102.txt --pretrained_dir models/checkpoints/flowers102_vit_l_16.bin
# python3 train.py --num_epochs 40 --dataset flowers102 --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_flowers102_train.txt --checkpoint_path models/checkpoints/flowers102_vit_h_14.bin
# python3 eval.py --dataset flowers102 --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_flowers102.txt --pretrained_dir models/checkpoints/flowers102_vit_h_14.bin
# python3 train.py --num_epochs 40 --dataset flowers102 --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_flowers102_train.txt --checkpoint_path models/checkpoints/flowers102_bit_l_res.bin
# python3 eval.py --dataset flowers102 --model_type bit_l_res --env newton --metric_path models/metrics/bit_l_res_flowers102.txt --pretrained_dir models/checkpoints/flowers102_bit_l_res.bin
# python3 train.py --num_epochs 40 --dataset flowers102 --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_flowers102_train.txt --checkpoint_path models/checkpoints/flowers102_eff_net_l2.bin
# python3 eval.py --dataset flowers102 --model_type eff_net_l2 --env newton --metric_path models/metrics/eff_net_l2_flowers102.txt --pretrained_dir models/checkpoints/flowers102_eff_net_l2.bin

#python3 eval.py --dataset imagenet --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_imagenet.txt

python3 generate_outputs.py --model_type vit_h_14 --dataset imagenet --image_folder models/images/imagenet --env newton
python3 generate_outputs.py --model_type vit_h_14 --dataset imagenet_real --image_folder models/images/imagenet_real --env newton
python3 generate_outputs.py --model_type vit_h_14 --dataset flowers102 --image_folder models/images/flowers102 --env newton --pretrained_dir models/checkpoints/flowers102_vit_h_14.bin
python3 generate_outputs.py --model_type vit_h_14 --dataset oxford_iiit --image_folder models/images/oxford_iiit --env newton --pretrained_dir models/checkpoints/oxford_iiit_vit_h_14.bin
python3 generate_outputs.py --model_type vit_h_14 --dataset cifar10 --image_folder models/images/cifar10 --env newton --pretrained_dir models/checkpoints/cifar10_vit_h_14.bin
python3 generate_outputs.py --model_type vit_h_14 --dataset cifar100 --image_folder models/images/cifar100 --env newton --pretrained_dir models/checkpoints/cifar100_vit_h_14.bin

