#!/bin/sh

# Test settings
BASE=/home/projects/benchmarks/workload
test=055_linpack_cuda
failmetric=900.0

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
tmp=tmp
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
#PBS -l select=${target}:ngpus=1:ncpus=24:mpiprocs=1:ompthreads=24:mem=96gb
#PBS -q gpu
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -P 90000001

set -e

if [ -d "\$PBS_O_WORKDIR" ] ; then cd "\$PBS_O_WORKDIR" ; fi

failmetric="$failmetric"

#module load cuda80/toolkit/8.0.44 intelmpi mkl
#HPL_DIR=/app/benchmarks/linpack-cuda
module load cuda/10.1 intelmpi mkl
HPL_DIR=/app/benchmarks/linpack-cuda-10.1
export KMP_AFFINITY="granularity=fine,compact"
CPU_CORES_PER_GPU=24
export MKL_NUM_THREADS=\$CPU_CORES_PER_GPU
export MKL_DYNAMIC=FALSE
export LD_LIBRARY_PATH="\$HPL_DIR/lib:\$LD_LIBRARY_PATH"
export CUDA_DGEMM_SPLIT=0.70
export CUDA_DTRSM_SPLIT=0.60

echo clean node memory
$datadir/drop_caches
memhog 90g > /dev/null
for f in meminfo buddyinfo vmstat zoneinfo ; do cat /proc/\$f > proc.\$f ; done
numastat -n > numastat-n
numastat -m > numastat-m
cat /sys/kernel/debug/extfrag/extfrag_index > extfrag_index

echo == hostname
hostname
echo == uname
uname -a
echo == rpm
rpm -qa
echo == env
env
echo == ulimit
ulimit -a
echo == sysctl
sysctl -a
echo == uptime
uptime
echo == free
free
echo == ps
ps -ef
echo == nvidia-smi
nvidia-smi -q
echo == check ulimit -s
ulimit -s | grep unlimited
echo == check ulimit -l
ulimit -l | grep unlimited
echo == check ulimit -v
ulimit -v | grep unlimited

echo run warmup
cp -p "$datadir"/HPL.dat.* .
ln -s HPL.dat.warmup HPL.dat
nvidia-smi -ac 3004,745
mpirun -np 1 \$HPL_DIR/bin/xhpl > /dev/null 2>&1
echo run production
rm -f HPL.dat
ln -s HPL.dat.production HPL.dat
mpirun -np 1 \$HPL_DIR/bin/xhpl > linpack_cuda.\$PBS_JOBID.stdout 2>&1
nvidia-smi -ac 3004,875

metric=\`awk '{if ( \$2 == 96000 ) {printf "%.1f\n",\$NF}}' linpack_cuda.\$PBS_JOBID.stdout\`
if [ x"\$metric" == x ] ; then metric=0.0 ; fi
echo metric=\"\$metric\"

echo check residual
grep ' PASSED$' linpack_cuda.\$PBS_JOBID.stdout
grep ' 1 tests completed and passed residual checks' linpack_cuda.\$PBS_JOBID.stdout

echo check for throttling
dmesg  | grep "Package temperature above threshold" && echo FAIL && exit 1 || echo OK

echo check performance
echo metric: \$metric
echo \$metric | awk '{if ( \$1 > '\$failmetric' ) {printf "PASS\n"}}' | grep PASS
EOF

jobid=`qsub $QSUB_ARGS run.pbs`
if [ x"$ECHO_JOBID" != x ] ; then echo $jobid ; fi
echo "$test , $jobid , Q , STIME , UNKNOWN , 0 , 00:00:00 , 0.0 , HOSTNAME , $PWD " >> "$csv"
