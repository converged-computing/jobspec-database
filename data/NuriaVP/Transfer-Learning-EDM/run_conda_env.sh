#!/bin/bash

#Informacion del trabajo
#SBATCH --job-name=mnist
#SBATCH -o scayle/out/conda_env_falta_5_%j.out
#SBATCH -e scayle/err/conda_env_falta_5_%j.err
#Recursos
#SBATCH --partition=cascadelakegpu
#SBATCH --qos=normal
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH --time=24:00:00
#SBATCH --mail-user=nvp1002@alu.ubu.es
#SBATCH --mail-type=ALL

#Directorio de trabajo
#SBATCH -D .

#Cargamos las variables necesarias y el entorno conda
export PATH=/home/ubu_eps_1/COMUNES/miniconda3/bin:$PATH
source /home/ubu_eps_1/COMUNES/miniconda3/etc/profile.d/conda.sh
conda activate env

python TFG_EMD_CV_DAonfly.py Xception True Parcial_all 16 K5 rgb
#Desactivamos el entorno
conda deactivate

#VGG16
#VGG19
#Xception
#ResNet50V2
#ResNet101
#ResNet152
#InceptionResNetV2
#InceptionV3
#DenseNet121
#DenseNet201
#EfficientNetB0