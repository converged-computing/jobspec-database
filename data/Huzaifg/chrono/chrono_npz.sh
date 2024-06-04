#!/usr/bin/sh

#SBATCH --time=48:0:0
#SBATCH -o job-%x-%A.out
#SBATCH -e job-%x-%A.err
#SBATCH -p gpu-a100-small
#SBATCH -A BCS20003
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)

module load intel/19.1.1
module load impi/19.0.9
module load mvapich2-gdr/2.3.7
module load mvapich2/2.3.7

module load phdf5/1.10.4
module load python3/3.9.7

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

PARENT="/work/09874/tliangwi/ls6/"
source "${PARENT}/gns/venv/bin/activate"
pip install pandas

cd $PARENT/chrono
python -u chrono_npz.py 197 sph_data
