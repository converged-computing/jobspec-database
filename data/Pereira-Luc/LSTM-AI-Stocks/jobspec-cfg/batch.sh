#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=20
##SBATCH -p batch

#SBATCH -G 2
#SBATCH --time=01:00:00
#SBATCH -p gpu


####SBATCH --reservation=hpcschool
module purge
module load devel/protobuf-python/3.14.0-GCCcore-10.2.0 
module load mpi/OpenMPI/4.0.5-GCC-10.2.0
module load system/CUDA/11.1.1-iccifort-2020.4.304

python3 -m pip install --no-cache --user mpi4py scikit-learn matplotlib
python3 -m pip install --no-cache --user numba
python3 -m pip install --no-cache --user pandas
python3 -m pip install --user torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

#python3 main.py
mpirun -np 1 python3 main.py 

#Batch script contains DOS line breaks -> dos2unix batch.sh