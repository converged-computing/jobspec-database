#!/bin/bash
#“#SBATCH” directives that convey submission options:
##### The name of the job
#SBATCH --job-name=bert2bert 
##### When to send e-mail: pick from NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=BEGIN,END,FAIL
##### Resources for your job
# number of physical nodes
#SBATCH --nodes=1
# number of task per a node (number of CPU-cores per a node)
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16000m
#SBATCH --cpus-per-gpu=4
# memory per a CPU-core
##### Maximum amount of time the job will be allowed to run
##### Recommended formats: MM:SS, HH:MM:SS, DD-HH:MM
#SBATCH --time=04:00:00
##### The resource account; who pays
#SBATCH --account=bioinf585w23_class
########## End of preamble! #########################################
# No need to “cd”. Slurm starts the job in the submission directory.
#####################################################################
# The application(s) to execute along with its input arguments and options:
my_job_header

/bin/hostname
module purge

module load python/3.10.4
module load pytorch
!pip install transformers bert-extractive-summarizer datasets evaluate rouge_score sacrebleu git-python==1.0.3

python3 train_bert2bert.py
