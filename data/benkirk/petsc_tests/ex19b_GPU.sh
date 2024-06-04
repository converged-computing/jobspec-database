#!/bin/bash
#PBS -A <project_code>
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=64:mpiprocs=64:ompthreads=1:ngpus=4

### Set temp to scratch
[ -d /glade/gust/scratch/${USER} ] && export TMPDIR=/glade/gust/scratch/${USER}/tmp && mkdir -p $TMPDIR

. config_env.sh || exit 1

# force a specific runtime environment
# module purge
# module load crayenv
# module load PrgEnv-gnu/8.3.2 craype-x86-rome craype-accel-nvidia80 libfabric cray-pals cpe-cuda
# module list

### Interrogate Environment
env | sort | uniq | egrep -v "_LM|_ModuleTable|Modules|lmod_sh"



cd ${PETSC_DIR}/src/snes/tutorials || exit 1
make ex19

[ -x ./ex19 ] || { echo "cannot find tests: ex19"; exit 1; }

ps auxww | grep "nvidia-cuda-mps-control"
nvidia-smi -a > "nvidia-smi_a-${PBS_JOBID}.txt"
nvidia-smi

status="SUCCESS"




echo "------------------------------------------------"
echo " ex19:"
echo "------------------------------------------------"
ldd ex19

echo && echo && echo "********* Intra-Node (CPU) *****************"
mpiexec -n 32 --ppn 32 \
        ./ex19 -cuda_view -snes_monitor -pc_type mg -da_refine 10 -snes_view -pc_mg_levels 9 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -log_view \
    || status="FAIL"

mpiexec -n 64 --ppn 64 \
        ./ex19 -cuda_view -snes_monitor -pc_type mg -da_refine 10 -snes_view -pc_mg_levels 9 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -log_view \
    || status="FAIL"

echo && echo && echo "********* Intra-Node (GPU) *****************"
mpiexec -n 4 --ppn 4 ${top_dir}/get_local_rank \
        ./ex19 -cuda_view -snes_monitor -pc_type mg -dm_mat_type aijcusparse -dm_vec_type cuda -da_refine 10 -snes_view -pc_mg_levels 9 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -log_view \
    || status="FAIL"

echo && echo && echo "********* Inter-Node (GPU) *****************"
mpiexec -n 8 --ppn 4 ${top_dir}/get_local_rank \
        ./ex19 -cuda_view -snes_monitor -pc_type mg -dm_mat_type aijcusparse -dm_vec_type cuda -da_refine 10 -snes_view -pc_mg_levels 9 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -log_view \
    || status="FAIL"

echo && echo && echo
echo "${status}: Done at $(date)"
