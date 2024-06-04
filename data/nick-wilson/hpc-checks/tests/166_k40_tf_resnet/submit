#!/bin/sh

# Test settings
BASE=/home/projects/benchmarks/workload
test=166_k40_tf_resnet
failmetric=50.0
checkout=${CHECKOUT:-0}
keep=${KEEP:-0}

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
#PBS -l select=${target}:ngpus=1:ncpus=24:mpiprocs=1:ompthreads=24:mem=96gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -P 90000001
#PBS -q gpu

set -e

if [ -d "\$PBS_O_WORKDIR" ] ; then cd "\$PBS_O_WORKDIR" ; fi

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
echo === df
df /
df /var/tmp
df /var/spool/pbs/spool
df /var/spool/pbs/undelivered
df /home
df /data
df /app
df /scratch
df /seq
df /ime
echo === ofed_info
ofed_info
echo == ps
ps -ef
echo == nvidia-smi
nvidia-smi -q
echo == lscpu
lscpu
echo == lspci
lspci

echo == check ulimit -s
ulimit -s | grep unlimited
echo == check ulimit -l
ulimit -l | grep unlimited
echo == check ulimit -v
ulimit -v | grep unlimited

echo == check persistence mode ==
nvidia-smi -q | egrep 'Persistence Mode[[:space:]]*:[[:space:]]Enabled' || { echo ERROR: Persistence Mode ; echo FAIL ; exit 1 ; }
echo == check clock speed ==
nvidia-smi -q -d CLOCK | grep --after-context=4 'Max Clocks' | egrep 'Graphics[[:space:]]*:[[:space:]]*875 MHz' || { echo ERROR: Clock speed ; echo FAIL ; exit 1 ; }
echo == change that normal user can set clock speed ==
nvidia-smi -ac 3004,810 || { echo ERROR: Change clock speed failed ; echo FAIL ; exit 1 ; }
echo == set clock speed back to correct value ==
nvidia-smi -ac 3004,875
echo == check for GPU memory ECC errors
nvidia-smi -q -d PAGE_RETIREMENT  | egrep 'Pending[[:space:]]*:[[:space:]]*YES' && { echo ERROR: GPU memory ECC errors ; echo FAIL ; exit 1 ; }

if [ x"\${HOSTNAME:0:3}" == xstd -o x"\${HOSTNAME:0:3}" == xgpu ] ; then
echo == check number of cores and NUMA configuration ==
ncpu=24
nnuma=4
lscpu | egrep "^CPU.s.:[[:space:]]*\${ncpu}\\\$" || { echo number of cores ; echo FAIL ; exit 1 ; }
lscpu | egrep "^NUMA node.s.:[[:space:]]*\${nnuma}\\\$" || { echo number of cores ; echo FAIL ; exit 1 ; }
fi

echo == check GPFS protection ==
for pid in \`pidof mmfsd\` ; do echo \$pid ; grep -- -1000 /proc/\$pid/oom_score_adj ; done

echo == check simple GPU test
$BASE/data/bin/deviceQuery

echo == check GPU Direct kernel modules
lsmod | grep ^nv
lsmod | grep ^nv_peer_mem > /dev/null || { echo ERROR: Missing GPU direct kernel module ; echo FAIL ; exit 1 ; }

echo === stats
for f in meminfo buddyinfo vmstat zoneinfo ; do cat /proc/\$f ; done
numastat -n
numastat -m
cat /sys/kernel/debug/extfrag/extfrag_index

checkout=$checkout
if [ \$checkout -eq 1 ] ; then
 git clone -b cnn_tf_v1.10_compatible https://github.com/tensorflow/benchmarks.git
else
 tar zxf $datadir/benchmarks.tgz
fi

module load singularity/latest
singularity exec \$SINGULARITY_IMAGES/tensorflow/tensorflow.1.11.0-gpu.simg \
    python benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --model resnet50 \
        --variable_update horovod > stdout.\$PBS_JOBID 2> stderr.\$PBS_JOBID

grep 'Devices:.*horovod/gpu:0' stdout.\$PBS_JOBID > /dev/null  || { echo job fail ; echo FAIL ; exit 1 ; }
grep '^total images/sec:' stdout.\$PBS_JOBID > /dev/null  || { echo job fail ; echo FAIL ; exit 1 ; }
keep=$keep
if [ \$keep -ne 1 ] ; then rm -rf benchmarks ; fi

failmetric="$failmetric"
metric=\`grep '^total images/sec:' stdout.\$PBS_JOBID | awk '{print \$NF}'\`
echo metric=\"\$metric\"
echo check performance
echo threshold: \$failmetric
echo metric: \$metric
echo \$metric | awk '{if ( \$1 > '\$failmetric' ) {printf "PASS\n"}}' | grep PASS || { echo FAIL ; exit 1 ; }
EOF

jobid=`qsub $QSUB_ARGS run.pbs`
if [ x"$ECHO_JOBID" != x ] ; then echo $jobid ; fi
echo "$test , $jobid , Q , STIME , UNKNOWN , 0 , 00:00:00 , 0.0 , HOSTNAME , $PWD " >> "$csv"
