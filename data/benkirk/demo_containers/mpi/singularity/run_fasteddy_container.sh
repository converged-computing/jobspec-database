#!/bin/bash
#PBS -A SCSG0001
#PBS -q main
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l select=6:ncpus=64:mpiprocs=4:ngpus=4

### Set temp to scratch
[ -d /glade/gust/scratch/${USER} ] && export TMPDIR=/glade/gust/scratch/${USER}/tmp && mkdir -p $TMPDIR

. config_env.sh >/dev/null 2>&1 || exit 1

nnodes=$(cat ${PBS_NODEFILE} | sort | uniq | wc -l)
nranks=$(cat ${PBS_NODEFILE} | sort | wc -l)
nranks_per_node=$((${nranks} / ${nnodes}))

status="SUCCESS"

#container_image=./rocky8-openhpc-fasteddy-sandbox/
container_image=./rocky8-openhpc-fasteddy.sif

echo "ldd, container native:"
singularity \
    exec \
    ${container_image} \
    ldd /opt/local/FastEddy-model/SRC/FEMAIN/FastEddy

echo "ldd, host bind:"
singularity \
    exec \
    --bind /run \
    --bind /usr/lib64:/host/lib64 \
    --bind /opt/cray \
    --env LD_LIBRARY_PATH=${CRAY_MPICH_DIR}/lib-abi-mpich:/opt/cray/pe/lib64:${LD_LIBRARY_PATH}:/host/lib64 \
    --env MPICH_SMP_SINGLE_COPY_MODE=NONE \
    --env MPICH_GPU_SUPPORT_ENABLED=1 \
    --env MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1 \
    ${container_image} \
    ldd /opt/local/FastEddy-model/SRC/FEMAIN/FastEddy

[[ "x${PBS_NODEFILE}" != "x" ]] || { echo "Not in a PBS Job, exiting..."; exit 0; }

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

[[ -f ./Example02_CBL.in ]] || wget https://raw.githubusercontent.com/NCAR/FastEddy-tutorials/main/examples/Example02_CBL.in \
    || { echo "Cannot download input file!!"; exit 1; }

mkdir ./output

tstart=$(date +%s)
echo "# --> BEGIN execution"

mpiexec --np ${nranks} --ppn ${nranks_per_node} --no-transfer \
  set_gpu_rank \
  singularity \
  exec \
  --bind /run \
  --bind /usr/lib64:/host/lib64 \
  --bind /opt/cray \
  --bind ${SCRATCH} \
  --bind ${WORK} \
  --pwd $(pwd) \
  --env LD_LIBRARY_PATH=${CRAY_MPICH_DIR}/lib-abi-mpich:/opt/cray/pe/lib64:${LD_LIBRARY_PATH}:/host/lib64 \
  --env MPICH_SMP_SINGLE_COPY_MODE=NONE \
  --env MPICH_GPU_SUPPORT_ENABLED=1 \
  --env MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1 \
  --env LD_PRELOAD=/opt/cray/pe/mpich/${CRAY_MPICH_VERSION}/gtl/lib/libmpi_gtl_cuda.so.0 \
  ${container_image} \
  /opt/local/FastEddy-model/SRC/FEMAIN/FastEddy \
  ./Example02_CBL.in \
    || status="FAILED"

echo "# --> END execution"

tstop=$(date +%s)
echo $((${tstop}-${tstart})) " elapsed seconds"
echo && echo && echo ${status} $(date)
