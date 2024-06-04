#PBS -S /bin/bash
#PBS -V
#PBS -l nodes=1:ppn=8
#PBS -l walltime=0:10:00
#PBS -l mem=5MB
#PBS -e /home/mjw/HOD_MockRun/mpi_helloworld_stderr.pbs
#PBS -o /home/mjw/HOD_MockRun/mpi_helloworld_stdout.pbs

mpirun -np 8 /home/mlam/anaconda/bin/python /home/mjw/HOD_MockRun/mpi_helloworld.py

chmod +rx /home/mjw/HOD_MockRun/mpi_helloworld_stdout.pbs
