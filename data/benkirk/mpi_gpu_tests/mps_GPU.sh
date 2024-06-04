#!/bin/bash
#PBS -A SCSG0001
#PBS -j oe
#PBS -q main
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=32:mpiprocs=32:ompthreads=1:ngpus=4

### Set temp to scratch
[ -d /glade/gust/scratch/${USER} ] && export TMPDIR=/glade/gust/scratch/${USER}/tmp && mkdir -p $TMPDIR

. config_env.sh || exit 1

### Interrogate Environment
env | sort | uniq | egrep -v "_LM|_ModuleTable|Modules|lmod_sh"

TESTS_DIR=${inst_dir}

[ -d ${TESTS_DIR} ] || { echo "cannot find tests: ${TESTS_DIR}"; exit 1; }

ps auxww | grep "nvidia-cuda-mps-control"
nvidia-smi -a > "nvidia-smi_a-${PBS_JOBID}.txt"
nvidia-smi --query | grep 'Compute Mode'
nvidia-smi


strace /glade/u/home/benkirk/mpi_gpu_tests/install-ncarenv/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoallv >/dev/null 2>&1


exit 0

status="SUCCESS"

echo "------------------------------------------------"
echo " pt2pt tests:"
echo "------------------------------------------------"
for tool in $(find ${TESTS_DIR} -type f -executable -name osu_bw -o -name osu_bibw -o -name osu_latency | sort); do

    echo ${tool} && ldd ${tool}

    echo && echo && echo "********* Intra-Node-CPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 2 $(which get_local_rank) ${tool} || status="FAIL"

    echo && echo && echo "********* Inter-Node-CPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 1 $(which get_local_rank) ${tool} || status="FAIL"

    echo && echo && echo "********* Intra-Node-GPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 2 $(which get_local_rank) ${tool} D D || status="FAIL"

    echo && echo && echo "********* Inter-Node-GPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 1 $(which get_local_rank) ${tool} D D || status="FAIL"
done

echo "------------------------------------------------"
echo " collective tests:"
echo "------------------------------------------------"
for tool in $(find ${TESTS_DIR} -type f -executable -name "osu_alltoall*" | sort); do

    echo ${tool} && ldd ${tool}

    echo && echo && echo "********* Intra-Node-CPU *****************"
    echo ${tool} && mpiexec -n 4 --ppn 4 $(which get_local_rank) ${tool} || status="FAIL"

    echo && echo && echo "********* Inter-Node-CPU *****************"
    echo ${tool} && mpiexec -n 8 --ppn 4 $(which get_local_rank) ${tool} || status="FAIL"

    echo && echo && echo "********* Intra-Node-GPU  (nranks == ngpus) *****************"
    echo ${tool} && mpiexec -n 4 --ppn 4 $(which get_local_rank) ${tool} -d managed || status="FAIL"

    echo && echo && echo "********* Inter-Node-GPU (nranks == ngpus) *****************"
    echo ${tool} && mpiexec -n 8 --ppn 4 $(which get_local_rank) ${tool} -d managed || status="FAIL"

    echo && echo && echo "********* Intra-Node-GPU (nranks == 4*ngpus) *****************"
    echo ${tool} && mpiexec -n 32 --ppn 32 $(which get_local_rank) ${tool} -d managed || status="FAIL"

    echo && echo && echo "********* Inter-Node-GPU (nranks == 4*ngpus) *****************"
    echo ${tool} && mpiexec -n 64 --ppn 32 $(which get_local_rank) ${tool} -d managed || status="FAIL"
done

echo && echo && echo
echo "${status}: Done at $(date)"
