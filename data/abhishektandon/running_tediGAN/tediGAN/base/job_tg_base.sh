#!/bin/bash
#PBS -N tg
#PBS -l nodes=1:ppn=16
#PBS -q external
#PBS -o log.log
#PBS -e out_p3.log

cd $PBS_O_WORKDIR
echo "Running on: " 
cat ${PBS_NODEFILE}
echo "Program Output begins: " 

##conda create --name stylegan2 python=3.8
##conda install pip 
##pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
##pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
##git clone https://github.com/os-netizen/styleGAN2.git
##git clone https://github.com/os-netizen/tediGAN.git
##pip install ftfy regex tqdm
##pip install git+https://github.com/openai/CLIP.git

module load anaconda3

source activate stylegan2

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda/
echo $CUDA_HOME > out_teenage_boy.txt

# CUDA_VISIBLE_DEVICES=1  python invert.py --mode='gen' --description='young indian boy wearing spectacles' --learning_rate=0.1 --num_iterations=100 --num_results=2 --loss_weight_clip=3.0 > out.txt
# CUDA_VISIBLE_DEVICES=1  python invert.py --mode='gen' --description='very old indian man' --learning_rate=0.1 --num_iterations=100 --num_results=2 --loss_weight_clip=4.0 > out_old_indian_man_3.log
# CUDA_VISIBLE_DEVICES=1  python invert.py --mode='gen' --description='he is an indian man, he is wearing a hat' --learning_rate=0.1 --num_iterations=120 --num_results=2 --loss_weight_clip=2.0 > out_1705.txt
# CUDA_VISIBLE_DEVICES=1  python invert.py --mode='gen' --description='elderly indian man with a white beard' --learning_rate=0.1 --num_iterations=100 --num_results=2 --loss_weight_clip=3.0 > out_elder_white_beard.txt
# CUDA_VISIBLE_DEVICES=1  python invert.py --mode='gen' --description='young indian woman with black hair' --learning_rate=0.1 --num_iterations=100 --num_results=2 --loss_weight_clip=2.0 > out_woman_black_hair.txt
CUDA_VISIBLE_DEVICES=1  python invert.py --mode='gen' --description='indian teenage boy with curly hair and brown eyes' --learning_rate=0.1 --num_iterations=100 --num_results=2 --loss_weight_clip=3.0 >> out_teenage_boy.txt
