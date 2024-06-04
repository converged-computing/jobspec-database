#!/bin/bash
#PBS -A <project_code>
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=8:mpiprocs=8:ompthreads=1:ngpus=4

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
mpiexec -n 2 --ppn 2 \
        ./ex19 -da_refine 3 -snes_monitor -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view  \
    || status="FAIL"

mpiexec -n 4 --ppn 4 \
        ./ex19 -da_refine 4 -snes_monitor -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view  \
    || status="FAIL"

echo && echo && echo "********* Intra-Node (GPU) *****************"
mpiexec -n 2 --ppn 2 ${top_dir}/get_local_rank \
        ./ex19 -da_refine 3 -snes_monitor -dm_mat_type mpiaijcusparse -dm_vec_type mpicuda -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view  \
    || status="FAIL"

mpiexec -n 4 --ppn 4 ${top_dir}/get_local_rank \
        ./ex19 -da_refine 4 -snes_monitor -dm_mat_type mpiaijcusparse -dm_vec_type mpicuda -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view  \
    || status="FAIL"

echo && echo && echo "********* Inter-Node (GPU) *****************"
mpiexec -n 8 --ppn 4 ${top_dir}/get_local_rank \
        ./ex19 -da_refine 4 -snes_monitor -dm_mat_type mpiaijcusparse -dm_vec_type mpicuda -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view  \
    || status="FAIL"

# fails with:
# (GTL DEBUG: 13) cuIpcOpenMemHandle: resource already mapped, CUDA_ERROR_ALREADY_MAPPED, line no 272
# (GTL DEBUG: 15) cuIpcOpenMemHandle: resource already mapped, CUDA_ERROR_ALREADY_MAPPED, line no 272
# (GTL DEBUG: 10) cuIpcOpenMemHandle: resource already mapped, CUDA_ERROR_ALREADY_MAPPED, line no 272
#mpiexec -n 16 --ppn 8 ${top_dir}/get_local_rank \
#        ./ex19 -da_refine 3 -snes_monitor -dm_mat_type mpiaijcusparse -dm_vec_type mpicuda -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view  \
#    || status="FAIL"

echo && echo && echo
echo "${status}: Done at $(date)"
