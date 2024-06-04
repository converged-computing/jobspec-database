#!/bin/sh

# ------------------------------- input
build_dir=$HOME/build/tsgemm
source_dir=$HOME/code/tsgemm
job_time=10                 # [min]
exe_name=v1                 # v2 v2_pool v2_priority
arch=mc                     # or 'gpu'

apex=1                      # or 0 for no APEX
len_m=3000
len_n=3000
len_k=523000
tile_m=512
tile_n=512
tile_k=10000
pgrid_rows=5
pgrid_cols=2
blk_rows=512
blk_cols=512
# --------------------------------

if [[ ${arch} == mc ]]; then
  job_nprocs_per_node=2
  job_threads_per_proc=18
elif [[ ${arch} == gpu ]]; then
  job_nprocs_per_node=1
  job_threads_per_proc=12
else
  echo "arch must be gpu or mc!\n"
  exit 1
fi

nprocs="$(( ${pgrid_rows} * ${pgrid_cols} ))"
if [[ $(( ${nprocs} % ${job_nprocs_per_node} )) -ne 0 ]]; then
    echo "nprocs must be divisible by job_nprocs_per_node!\n"
    exit 1
fi

job_queue=normal
job_nodes="$(( ${nprocs} / ${job_nprocs_per_node} ))"
date_str=$(date '+%Y.%m.%d')
job_name="${date_str}__${exe_name}_\
${len_m}.${len_n}.${len_k}_${tile_m}.${tile_n}.${tile_k}_\
${pgrid_rows}.${pgrid_cols}_${blk_rows}.${blk_cols}"

# exe parameters
#  "--hpx:print-bind --hpx:threads=${job_threads_per_proc}"
params="--hpx:queuing=shared-priority --hpx:use-process-mask \
--len_m ${len_m} --len_n ${len_n} --len_k ${len_k} \
--tile_m ${tile_m} --tile_n ${tile_n} --tile_k ${tile_k} \
--pgrid_rows ${pgrid_rows} --pgrid_cols ${pgrid_cols} \
--blk_rows ${blk_rows} --blk_cols ${blk_cols}"
exe_path=${build_dir}/apps/${exe_name}

mkdir ${job_name}
cd ${job_name}

# generate job file
cat << JOB_EOF > job.sh
#!/bin/sh
#SBATCH --job-name=${job_name}
#SBATCH --constraint=${arch}
#SBATCH --partition=${job_queue}
#SBATCH --nodes=${job_nodes}
#SBATCH --time=${job_time}
#SBATCH --ntasks-per-node=${job_nprocs_per_node}
#SBATCH --cpus-per-task=${job_threads_per_proc}
#SBATCH --hint=nomultithread
#SBATCH --output=output.txt
#SBATCH --error=error.txt

device=daint
source ${source_dir}/scripts/env.sh

# MPI_THREAD_MULTIPLE for Cray
export MPICH_MAX_THREAD_SAFETY=multiple

# modules snapshot
module list &> modules.txt

# environment snapshot
printenv > env.txt

# libraries snapshot
ldd ${exe_path} > libs.txt

# run code
srun ./cmd.sh
JOB_EOF
chmod +x job.sh

# generate command file
cat << CMD_EOF > cmd.sh
#!/bin/sh

if [[ \${SLURM_PROCID} == 0 ]]; then
  export APEX_OTF2=${apex}
  #export APEX_PROFILE=1
  #export APEX_SCREEN_OUTPUT=1
fi

${exe_path} ${params}
CMD_EOF
chmod +x cmd.sh
