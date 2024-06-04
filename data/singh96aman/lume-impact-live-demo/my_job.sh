#!/bin/bash
#SBATCH --job-name apptainer-mpi
#SBATCH -N 32 # total number of nodes
#SBATCH --time=00:05:00 # Max execution time
/sdf/sw/gcc-4.8.5/openmpi-4.0.4/bin/mpirun -n 32 singularity exec --cleanenv --env SCRATCH=$SCRATCH,LCLS_LATTICE=$LCLS_LATTICE,PATH="/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/slurm/slurm-curr/bin/" --bind ./slurm_config/:/run/slurm/conf/,/etc/passwd,/etc/group,/opt/slurm,/usr/lib64/libreadline.so.6,/usr/lib64/libhistory.so.6,/usr/lib64/libtinfo.so.5,/var/run/munge,/usr/lib64/libmunge.so.2,/usr/lib64/libmunge.so.2.0.0,/run/munge,/opt/slurm/slurm-curr/bin/srun,/scratch,$LCLS_LATTICE,$LUME_OUTPUT_FOLDERS:/app/output/,/sdf/group/ard/thakur12/lume-impact-live-demo:/app/ --pwd /app lume-impact-live-demo_main.sif /bin/bash -c 'conda run -n lume-live-dev ipython lume-impact-live-demo.py -- -t "singularity" -p 32'
