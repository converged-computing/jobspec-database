#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p core
#SBATCH -n 4
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH -J train_nn_rom
#SBATCH -D ./

conda activate tf

nvidia-smi

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

echo "Train networks..."


##### ROM #####
# ConvAuto
# python3 train_nn.py rom ConvAuto sobel -stride 2 -nimages 4080
# python3 train_nn.py rom ConvAuto sobel -stride 5 -nimages 4080
# python3 train_nn.py rom ConvAuto ssim -stride 2 -nimages 4080
# python3 train_nn.py rom ConvAuto ssim -stride 5 -nimages 4080

# ConvNN
# python3 train_nn.py rom ConvNN sobel -stride 5 -nimages 4080
# python3 train_nn.py rom ConvNN ssim -stride 5 -nimages 4080

python3 train_nn.py rtm ConvNNshallow sobel -stride 5 -nimages 4080
python3 train_nn.py rtm ConvNNshallow ssim -stride 5 -nimages 4080

python3 train_nn.py rtm ConvNNdeep sobel -stride 5 -nimages 4080
python3 train_nn.py rtm ConvNNdeep ssim -stride 5 -nimages 4080

# ResNet
# python3 train_nn.py rom ResNet sobel -stride 2 -nimages 4080
# python3 train_nn.py rom ResNet sobel -stride 5 -nimages 4080
# python3 train_nn.py rom ResNet ssim -stride 2 -nimages 4080
# python3 train_nn.py rom ResNet ssim -stride 5 -nimages 4080

# UNet
# python3 train_nn.py rom UNet sobel -stride 2 -nimages 4080
# python3 train_nn.py rom UNet sobel -stride 5 -nimages 4080
# python3 train_nn.py rom UNet ssim -stride 2 -nimages 4080
# python3 train_nn.py rom UNet ssim -stride 5 -nimages 4080


echo " "
echo "Finished calculations"