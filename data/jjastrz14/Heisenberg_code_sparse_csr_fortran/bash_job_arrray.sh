#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-core=1
#SBATCH --mem=1gb
#SBATCH --mail-user=jakub.jastrzebski99@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=10-12%1  # Adjust as necessary for your needs: IDs 10 to 20 with step 2

source /usr/local/sbin/modules.sh

module load GCCcore/12.2.0 
module load zlib/1.2.12-GCCcore-12.2.0 
module load binutils/2.39-GCCcore-12.2.0 
module load intel-compilers/2022.2.1 
module load numactl/2.0.16-GCCcore-12.2.0 
module load UCX/1.13.1-GCCcore-12.2.0 
module load impi/2021.7.1-intel-compilers-2022.2.1 
module load imkl/2022.2.1 
module load iimpi/2022b 
module load imkl-FFTW/2022.2.1-iimpi-2022b 
module load intel/2022b 
module load FEAST/4.0

ifort main_spin_code_feast_time_window.f90 -L/opt/FEAST/4.0/lib/x64 -lfeast -qopenmp -qmkl -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -xHost -O3 -static-intel -heap-arrays

#cp .f90 to lustre
cp a.out /lustre/tmp/slurm/$SLURM_JOB_ID/a.out
#go to luste
cd /lustre/tmp/slurm/$SLURM_JOB_ID
echo "cd DONE!"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./a.out $SLURM_ARRAY_TASK_ID 1.0 #> log_feast_$SLURM_ARRAY_TASK_ID.out

#cp log_feast_$SLURM_ARRAY_TASK_ID.out /home/jjastrz9/tmp/Heisenberg_program_feast/energy_window/log_feast_$SLURM_ARRAY_TASK_ID.out
#cp slurm-$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.out /home/jjastrz9/tmp/Heisenberg_program_feast/energy_window_1May/slurm-$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.out
cp Eigenvalues_results_${SLURM_ARRAY_TASK_ID}_feast.dat /home/jjastrz9/tmp/Heisenberg_program_feast/energy_window_2May/Eigenvalues_results_${SLURM_ARRAY_TASK_ID}_feast.dat
cp Cpu_Time_details_${SLURM_ARRAY_TASK_ID}.dat /home/jjastrz9/tmp/Heisenberg_program_feast/energy_window_2May/Cpu_Time_details_${SLURM_ARRAY_TASK_ID}.dat
