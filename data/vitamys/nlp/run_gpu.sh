#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=vinzenz.uhr@gmail.com

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=BEGIN,FAIL,END



# Job name
#SBATCH --job-name="nlp-project"

# Runtime and memory
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=12G
# ##SBATCH --cpus-per-task=8

# Partition
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx3090:1
##SBATCH --gres=gpu:gtx1080ti:2


#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err


#### Your shell commands below this line ####


# module load CUDA/10.1.243
# module load cuDNN/7.6.0.64-gcccuda-2019a

singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install tensorboard
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install -U scikit-learn
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install datasets
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install transformers
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install matplotlib
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install pandas
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install spacy
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install nltk
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install ftfy regex tqdm 
#singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install simpletransformers

#### Start training,testing, evaluation ####
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime python3 project_split_rows_with_pp.py
