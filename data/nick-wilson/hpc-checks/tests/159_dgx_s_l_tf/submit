#!/bin/sh

# Test settings
BASE=/home/projects/benchmarks/workload
test=159_dgx_s_l_tf
failmetric=1750.0
layers=152
batch=64
dataset="resnet"
data="/scratch/projects/ai/datasets/imagenet/$dataset"
i_latest="nvcr.io/nvidia/tensorflow:19.06-py3"
image=${IMAGE:-$i_latest}

# run on any host
target=1
# run on host specified on command line
pbsname="$test"
if [ x"$1" != x ] ; then target="host=$1" ; pbsname="${test}_${1}" ; fi

# Environment settings
PATH="${PATH}:/app/pbs/bin:/opt/pbs/bin" ; export PATH

# Script settings
d=`date +%Y/%m/%d/%H/%M/%S`
rundir="$BASE/run/$d/$test"
datadir="$BASE/data/$test"
script=run.pbs
csv="$BASE/results.csv"

tst=""
if [ x"$MULTI" == x ] ; then
 # Only have one test in queue at a time
 qname="${pbsname:0:15}"
 tst=`qstat | awk '{if ( $2 == "'$qname'" ) {print $0}}'`
fi
if [ x"$tst" != x ] ; then exit ; fi

# Prepare and submit test
mkdir -p $rundir && cd $rundir || exit

cat << EOF > run.pbs
#!/bin/bash
#PBS -N $pbsname
#%@FUJITSU:NSCC:CONTINUOUS_PERFORMANCE_TESTING@%#
#PBS -l select=${target}:ncpus=40:ngpus=8
#PBS -q dgx
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -P 90000001
set -e
env
nvidia-smi
if [ -d "\$PBS_O_WORKDIR" ] ; then cd "\$PBS_O_WORKDIR" ; fi
nvidia-smi
image="$image"
image="\$SINGULARITY_IMAGES/\$image.simg"
ls -l "\$image"
md5sum "\$image"
singularity exec --nv "\$image" /bin/bash < stdin > stdout.\$PBS_JOBID 2> stderr.\$PBS_JOBID
grep 'Created TensorFlow device .*/device:GPU' stderr.\$PBS_JOBID || { echo job fail ; echo FAIL ; exit 1 ; }
grep -x '  Step Epoch Img/sec   Loss  LR' stdout.\$PBS_JOBID || { echo job fail ; echo FAIL ; exit 1 ; }
failmetric="$failmetric"
metric=\`grep -x --after-context=21 '  Step Epoch Img/sec   Loss  LR' stdout.\$PBS_JOBID | awk '/^ *190 / {print \$3}'\`
echo metric=\"\$metric\"
echo check performance
echo threshold: \$failmetric
echo metric: \$metric
echo \$metric | awk '{if ( \$1 > '\$failmetric' ) {printf "PASS\n"}}' | grep PASS || { echo FAIL ; exit 1 ; }
EOF

cat << EOF > stdin
dataset="$dataset"
layers="$layers"
batch="$batch"
data="$data"
cd /workspace/nvidia-examples/cnn
mpiexec --allow-run-as-root -np 8 python \${dataset}.py \
 --layers \$layers \
 -b \$batch \
 -i 200 \
 -u batch \
 --data_dir \$data
EOF

jobid=`qsub $QSUB_ARGS run.pbs`
if [ x"$ECHO_JOBID" != x ] ; then echo $jobid ; fi
echo "$test , $jobid , Q , STIME , UNKNOWN , 0 , 00:00:00 , 0.0 , HOSTNAME , $PWD " >> "$csv"
