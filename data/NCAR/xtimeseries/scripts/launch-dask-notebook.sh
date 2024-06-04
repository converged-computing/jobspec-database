#!/bin/bash
set -e

#-- set defaults
# expected environment variables
conda_env=${DEFAULT_CONDA_ENV}
job_account=${JOB_ACCOUNT}

walltime=11:59:00
queue=economy
workers=4
notebook_dir=${HOME}
add_workers=false

#-- define usage
usage () {
cat << EOF
Usage: launch dask
Possible options are:
 -a,--account: account
 -w,--walltime: walltime
 -q,--queue: queue
 -n,--workers: workers
 -d,--direcory: notebook directory
 -e,--env: Conda environment
 --add-workers: add workers to existing scheduler
EOF
exit
}

#-- parse input args
while [[ $# -gt 0 ]]; do
  key="${1}"
  case ${key} in
    -h|--help)
      usage ;;
    -a|--account)
      job_account="${2}"
      shift 2
      ;;
    -w|--walltime)
      walltime="${2}"
      shift 2
      ;;
    -q|--queue)
      queue="${2}"
      shift 2
      ;;
    -n|--workers)
      workers="${2}"
      shift 2
      ;;
    -d|--directory)
      notebook_dir="${2}"
      shift 2
      ;;
    -e|--env)
      conda_env="${2}"
      shift 2
      ;;
    --add-workers)
      add_workers=true
      shift
      ;;
    *)
      echo "ERROR: unknown argument: ${key}"
      usage
      ;;
  esac
done

#-- check inputs
if [ -z "${job_account}" ]; then
  source /glade/u/apps/ch/opt/usr/bin/getacct.sh
fi

if [ -z "${walltime}" ]; then
  echo "ERROR: walltime not set."
  exit 1
fi
if [ -z "${conda_env}" ]; then
  echo "ERROR: conda_env not set."
  exit 1
fi
if [ -z "${queue}" ]; then
  echo "ERROR: queue not set."
  exit 1
fi
if [ -z "${workers}" ]; then
  echo "ERROR: workers not set."
  exit 1
fi
if [[ -z ${WORKDIR} ]]; then
    WORKDIR=/glade/scratch/${USER}/dask
fi


#-- make sure the working directory exists
if [[ ! -d ${WORKDIR} ]]; then
  mkdir -vp ${WORKDIR}
fi

#-- if not in "add_workers" mode, start dask-scheduler
if [[ ${add_workers} =~ false ]]; then
  echo "Launching dask scheduler"
  s=$(qsub << EOF
#!/bin/bash
#PBS -N dask-scheduler
#PBS -q ${queue}
#PBS -A ${job_account}
#PBS -l select=1:ncpus=36:mpiprocs=9:ompthreads=4:mem=109GB
#PBS -l walltime=${walltime}
#PBS -j oe
#PBS -m ae

# Writes scheduler.json file
# Connect with
# >>> from dask.distributed import Client
# >>> client = Client(scheduler_file='~/scheduler.json')

# Setup Environment
module purge
source activate ${conda_env}

SCHEDULER=${WORKDIR}/scheduler.json
rm -f \${SCHEDULER}
mpirun --np 9 dask-mpi --nthreads 4 \
    --memory-limit 12e9 \
    --interface ib0 \
    --local-directory ${WORKDIR} \
    --scheduler-file=\${SCHEDULER}
EOF
)
  sjob=${s%.*}
  echo ${s}
fi

#-- lauch worker jobs
echo "Launching dask workers (${workers})"
for i in $(seq 1 ${workers}); do
    s=$(qsub << EOF
#!/bin/bash
#PBS -N dask-worker
#PBS -q ${queue}
#PBS -A ${job_account}
#PBS -l select=1:ncpus=36:mpiprocs=9:ompthreads=4:mem=109GB
#PBS -l walltime=${walltime}
#PBS -j oe
#PBS -m abe

# Setup Environment
module purge
source activate ${conda_env}

# Setup dask worker
SCHEDULER=${WORKDIR}/scheduler.json
mpirun --np 9 dask-mpi --nthreads 4 \
    --memory-limit 12e9 \
    --interface ib0 \
    --no-scheduler --local-directory ${WORKDIR} \
    --scheduler-file=\${SCHEDULER}
EOF
)
done

#-- if not in "add_workers" mode, wait on scheduler job and start notebook
if [[ ${add_workers} =~ false ]]; then
  qstat ${sjob}

  # block until the scheduler job starts
  while true; do
      status=$(qstat ${sjob} | tail -n 1)
      echo -n ..
      if [[ ${status} =~ " R " ]]; then
          break
      fi
      sleep 1
  done

  echo "Setting up Jupyter Lab, Notebook dir: ${notebook_dir}"
  source activate ${conda_env}
  setup-jlab.py --log_level=DEBUG --jlab_port=8877 --dash_port=8878 \
      --notebook_dir $notebook_dir --scheduler_file $WORKDIR/scheduler.json
fi
