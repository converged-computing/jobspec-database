#!/bin/bash
#
#SBATCH --job-name=RURIKO_TENSORFLOW # Job name
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks=20 # Number of cores
#SBATCH --output=train_network.out #File to which STDOUT will be written
#SBATCH --error=train_network.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=raimai@ucdavis.edu # Email to which notification will be sent

python train_network.py

