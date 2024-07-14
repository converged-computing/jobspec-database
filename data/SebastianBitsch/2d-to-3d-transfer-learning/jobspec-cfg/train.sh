#!/bin/bash

### -- set the job Name -- 
#BSUB -J Train

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
# -- end of LSF options --

### -- specify queue -- 
#BSUB -q gpua100

### -- ask for number of cores -- 
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"

### -- set walltime limit: hh:mm --
#BSUB -W 23:00

### -- set the email address --
#BSUB -u s204163@student.dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N

# Run with bsub < train.sh -env "config_name=baseline2d"

# Setup env
source .env/bin/activate

# Set the default config to use. kinda hacky ngl
config_name="${config_name:=config}"

# -cd tl_2d3d/conf 
python3 tl_2d3d/train_model.py --config-name=$config_name