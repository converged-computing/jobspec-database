#!/bin/bash
#SBATCH --ntasks=1           ### How many CPU cores do you need?
#SBATCH --mem=64G            ### How much RAM memory do you need?
#SBATCH -p express           ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH -t 0-01:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o whas500_out_%j.log        ### Where to store the console output (%j is the job number)
#SBATCH -e whas500_error_%j.log      ### Where to store the error output
#SBATCH --job-name=tf_hello  ### Name your job so you can distinguish between jobs

# Load the modules

module purge
# automatically loads all dependencies such as cuda
# replace with required tensorflow version! (check with module avail which tensorflow version are available)
# module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4  
module load Python/3.7.4-GCCcore-8.3.0

# use when you need to read/write many files quickly in tmp directory:
# source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env

# activate virtualenv after loading tensorflow/python module
# replace with your own virtualenv!
source /trinity/home/hmo/hmo/hmo_dl/bin/activate 
echo "deepsurv toturial"

python3 deepsurv_pytorch.py -dataset "whas500.xls" -model ~/hmo/dl_sa_tutorial/experiments/deepsurv/models/whas_model_relu_revision.0.json --update_fn "adam" --num_epochs "300"
# python tensorflow_hello.py
