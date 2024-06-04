#!/bin/bash
#PBS -N codiff
#PBS -l nodes=1:ppn=16
#PBS -q external
#PBS -o log.log
#PBS -e out.log

cd $PBS_O_WORKDIR
echo "Running on: " 
cat ${PBS_NODEFILE}
echo "Program Output begins: " 
module load anaconda3 


# conda env create -f environment.yaml
source activate codiff
# pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
## kornia 0.6.4 requires torch 2.0 which contradicts the previous pytorch version 1.7.0 as per the requirements file, thererfore following installations are done (torch 1.7.1)
# conda install -c anaconda git
# pip install git+https://github.com/arogozhnikov/einops.git
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# pip install kornia=0.4.1

CUDA_VISIBLE_DEVICES=1 python main.py --logdir 'outputs/512_codiff_mask_text' --base 'configs/512_codiff_mask_text.yaml' -t  --gpus 0, > train_log_test.txt

# CUDA_VISIBLE_DEVICES=1 python generate_512.py --mask_path test_data/512_masks/27007.png --input_text "This man has beard of medium length. He is in his thirties." > out.txt
