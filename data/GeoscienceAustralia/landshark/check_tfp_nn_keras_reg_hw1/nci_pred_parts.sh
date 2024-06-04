#!/bin/bash
#use this nci_pred_parts.sh script to run query and predictions from jobfs/local disc
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l jobfs=400GB
#PBS -q normal
#PBS -P ge3
#PBS -l walltime=24:00:0
#PBS -l storage=gdata/dg9+gdata/dz56+gdata/ge3+gdata/jl14
#PBS -l wd
#PBS -j oe
set -e
export name="ceno"
export halfwidth=2
export config="./nn_regression_keras_global_local.py"

export config_basename=${config##*/}
export config_stem=${config_basename%.*}
echo $config  $config_basename $config_stem


pid=$(grep ^Pid /proc/self/status)
corelist=$(grep Cpus_allowed_list: /proc/self/status | awk '{print $2}')
host=$(hostname | sed 's/.gadi.nci.org.au//g')
echo subtask $1 running in $pid using cores $corelist on compute node $host

module load intel-mkl/2021.4.0   python3/3.10.4   gdal/3.5.0   cuda/11.6.1  cudnn/8.2.2-cuda11.4  nccl/2.11.4 openmpi/4.1.2  tensorflow/2.8.0  parallel

source /g/data/ge3/sudipta/venvs/3p10land/bin/activate

export PYTHONPATH=/apps/gdal/3.5.0/lib/python3.10/site-packages/:/apps/tensorflow/2.8.0/lib/python3.10/site-packages/:/g/data/ge3/sudipta/venvs/3p10land/lib/python3.10/site-packages/:/apps/python3/3.10.4/lib/python3.10/site-packages/


function query_predict {
    echo starting query and preict $1 of $2
    n=$1
    N=$2
    mkdir -p $PBS_JOBFS/query_${name}_strip${1}of${2}
    landshark-extract --nworkers 1 --batch-mb 0.1 query \
        --features ./features_${name}.hdf5 \
        --strip ${n} ${N} --name ${name} \
        --halfwidth ${halfwidth} --remote $PBS_JOBFS/
    landshark -v DEBUG --keras-model --batch-mb 1 predict \
        --proba false \
        --config ${config} \
        --checkpoint ${config_stem}_model_1of1 \
        --data $PBS_JOBFS/query_${name}_strip${n}of${N} --pred_ensemble_size 1000
    echo done query and preict $1 of $2
    # rm -r /jobfs/query_${name}_strip${1}of${2}  # prediction written. Remove query
}
export -f query_predict

