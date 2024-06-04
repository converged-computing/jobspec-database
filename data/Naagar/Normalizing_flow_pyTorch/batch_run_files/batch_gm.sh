#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=S_Cif_q
#SBATCH --qos=medium
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --output=Output_file/logQ%j.out       # Output file.
#SBATCH --mail-type=END                # Enable email
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0

source venv/bin/activate
cd Auto.\ Navigation/gen_models/Normalizing-flow_San/
mpiexec -n 2 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 7 --flow_coupling 2 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 2501   --restore 
