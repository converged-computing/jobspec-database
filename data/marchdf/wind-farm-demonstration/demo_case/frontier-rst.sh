#!/bin/bash
#SBATCH -J demo_case
#SBATCH -o %x.o%j
#SBATCH -A CFD162
#SBATCH -t 12:00:00
#SBATCH -N 184
#SBATCH -S 0
##SBATCH -d afterany:1881424 # 00 -> 01
##SBATCH -d afterany:1881429 # 01 -> 02
##SBATCH -d afterany:1881430 # 02 -> 03
##SBATCH -d afterany:1881431 # 03 -> 04
##SBATCH -d afterany:1881433 # 04 -> 05

set -e
cmd() {
  echo "+ $@"
  eval "$@"
}

ofile="demo_case_rst04.inp"
rfile="demo_case_rst05.inp"

cmd "Xvfb :123 & export DISPLAY=:123"
cmd "source /ccs/proj/cfd162/lcheung/condaenv/frontier1/bin/activate"
cmd "python3 ../submods/amr-wind-frontend/utilities/restartAMRWind.py -o ${rfile} ${ofile}"

cmd "module unload PrgEnv-cray"
cmd "module load PrgEnv-amd"
cmd "module load amd/5.4.3"
cmd "export FI_MR_CACHE_MONITOR=memhooks"
cmd "export FI_CXI_RX_MATCH_MODE=software"
cmd "export MPICH_SMP_SINGLE_COPY_MODE=NONE"
cmd "export MPICH_GPU_SUPPORT_ENABLED=1"
cmd "export SPACK_MANAGER=${HOME}/exawind/spack-manager"
cmd "source ${SPACK_MANAGER}/start.sh && spack-start"
cmd "spack env activate -d ${SPACK_MANAGER}/environments/amr-wind-of-debug"
cmd "spack load amr-wind"
cmd "which amr_wind"
cmd "srun -N184 -n1472 -c1 --gpus-per-node=8 --gpu-bind=closest amr_wind ${rfile} amrex.abort_on_out_of_gpu_memory=1 amrex.the_arena_is_managed=0 amr.blocking_factor=16 amr.max_grid_size=128 amrex.use_profiler_syncs=0 amrex.async_out=0"
