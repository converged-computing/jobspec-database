#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:05:00
#SBATCH --job-name=TensorFlowTest
#SBATCH -o TensorFlowTest.o%j
#SBATCH --partition=gpu
#SBATCH --mail-user=jma7@mdanderson.org
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end 

module load gcc/4.9.1 cuda/8.0 cudnn/5.1 python3/3.5.2 tensorflow-gpu/1.0.0

python3 /home/05268/junma7/TensorFlow-Examples/examples/2_BasicModels/random_forest.py 
