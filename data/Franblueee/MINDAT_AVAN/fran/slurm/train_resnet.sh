#!/bin/bash
#SBATCH --job-name train              # Nombre del proceso
#SBATCH --partition dgx,dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=50GB                           # Numero de gpus a usar
#SBATCH --out=train_resnet.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
#conda activate /mnt/homeGPU/fcastro/conda-envs/LULC
conda activate /mnt/homeGPU/fcastro/conda-envs/newlulc
export LD_LIBRARY_PATH=/mnt/homeGPU/fcastro/conda-envs/newlulc/lib

python code/train.py --model_name=bitresnet50x3_baseline --DA_name=DA4 --ft_mode=0 --batch_size=32
python code/train.py --model_name=bitresnet50x3_v0 --DA_name=DA4 --ft_mode=0 --batch_size=32
python code/train.py --model_name=bitresnet50x3_v1 --DA_name=DA4 --ft_mode=0 --batch_size=32

python code/train.py --model_name=bitresnet101x3_baseline --DA_name=DA4 --ft_mode=0 --batch_size=32
python code/train.py --model_name=bitresnet101x3_v0 --DA_name=DA4 --ft_mode=0 --batch_size=32
python code/train.py --model_name=bitresnet101x3_v1 --DA_name=DA4 --ft_mode=0 --batch_size=32

python code/train.py --model_name=bitresnet152x3_baseline --DA_name=DA4 --ft_mode=0 --batch_size=32
python code/train.py --model_name=bitresnet152x3_v0 --DA_name=DA4 --ft_mode=0 --batch_size=32
python code/train.py --model_name=bitresnet152x3_v1 --DA_name=DA4 --ft_mode=0 --batch_size=32

python code/train.py --model_name=bitresnet50x3_baseline --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8
python code/train.py --model_name=bitresnet50x3_v0 --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8
python code/train.py --model_name=bitresnet50x3_v1 --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8

python code/train.py --model_name=bitresnet101x3_baseline --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8
python code/train.py --model_name=bitresnet101x3_v0 --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8
python code/train.py --model_name=bitresnet101x3_v1 --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8

python code/train.py --model_name=bitresnet152x3_baseline --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8
python code/train.py --model_name=bitresnet152x3_v0 --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8
python code/train.py --model_name=bitresnet152x3_v1 --DA_name=DA4 --ft_mode=1 --load_model=True --lr=0.000001 --batch_size=8