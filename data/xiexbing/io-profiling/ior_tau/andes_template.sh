#!/bin/bash -l
#SBATCH --mem=0 
#SBATCH -A stf008 
#SBATCH -t 1:00:00
#SBATCH -N 1 
##SBATCH -d PREV
#SBATCH -J RUN
#SBATCH -o o.hacc
#SBATCH -e e.hacc


module unload darshan-runtime
module load hdf5
EXEC=/gpfs/alpine/stf008/scratch/bing/ior_tau/src/ior
export PATH=/gpfs/alpine/stf008/scratch/bing/hacc_io/tau/tau2/x86_64/bin:$PATH

#rm -rf e.hacc o.hacc logs data_dir/ tau_log/
module list
mkdir data_dir logs
srun -n 4 tau_exec -T mpi -io -skel $EXEC -b 1m -t 1m -i 1 -v -v -v -k -a POSIX -e -w -o data_dir/posix_write&>>logs/posix_write 

mkdir -p tau_log/posix_write
mv skel/* tau_log/posix_write/


srun -n 4 tau_exec -T mpi -io -skel $EXEC -b 1m -t 1m -i 1 -v -v -v -k -a POSIX -r -o data_dir/posix_write&>>logs/posix_read 

mkdir -p tau_log/posix_read
mv skel/* tau_log/posix_read/

srun -n 4 tau_exec -T mpi -io -skel $EXEC -b 1m -t 1m -i 1 -v -v -v -k -a HDF5 -e -w -o data_dir/hdf5_write&>>logs/hdf5_write 

mkdir -p tau_log/hdf5_write
mv skel/* tau_log/hdf5_write/


srun -n 4 tau_exec -T mpi -io -skel $EXEC -b 1m -t 1m -i 1 -v -v -v -k -a HDF5 -r -o data_dir/hdf5_write&>>logs/hdf5_read 

mkdir -p tau_log/hdf5_read
mv skel/* tau_log/hdf5_read/



