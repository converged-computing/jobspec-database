#!/bin/bash
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l jobfs=10GB
#PBS -q normal
#PBS -P ge3
#PBS -l walltime=2:00:00
#PBS -l storage=gdata/dg9+gdata/dz56+gdata/ge3
#PBS -l wd
#PBS -j oe
set -e
export name="sirsam"
export halfwidth=1
export config="./nn_regression_keras_global_local.py"

export config_basename=${config##*/}
export config_stem=${config_basename%.*}
echo name: $name, halfwidth: $halfwidth
echo using config: $config,  basename: $config_basename, stem: $config_stem

pid=$(grep ^Pid /proc/self/status)
corelist=$(grep Cpus_allowed_list: /proc/self/status | awk '{print $2}')
host=$(hostname | sed 's/.gadi.nci.org.au//g')
echo subtask $1 running in $pid using cores $corelist on compute node $host

module load tensorflow/2.6.0 python3/3.9.2 openmpi/4.1.1 gdal/3.0.2 parallel
source /g/data/ge3/sudipta/venvs/land3p9n/bin/activate

export PYTHONPATH=/apps/tensorflow/2.6.0/lib/python3.9/site-packages:/g/data/ge3/sudipta/venvs/land3p9n/lib/python3.9/site-packages/:/apps/python3/3.9.2/lib/python3.9/site-packages/


function query {
  echo starting query $1 of $2
  n=$1
  N=$2
  mkdir -p query_${name}_strip${1}of${2}
  landshark-extract --nworkers 1 --batch-mb 0.1 query \
          --features ./features_${name}.hdf5 \
          --strip ${n} ${N} --name ${name} \
          --halfwidth ${halfwidth}

}

function predict {
  echo starting predict $1 of $2
  n=$1
  N=$2
  mkdir -p query_${name}_strip${1}of${2}
  landshark --keras-model --batch-mb 1 predict \
      --proba true \
      --config ${config} \
      --checkpoint ${config_stem}_model_1of10 \
      --data query_${name}_strip${n}of${N}
  echo done query and preict $1 of $2
  # rm -r query_${name}_strip${1}of${2}  # prediction written. Remove query
}
export -f query
export -f predict

function query_predict {
  n=$1
  N=$2
  query $n $N
  preidct $n $N
}

export -f query_predict
