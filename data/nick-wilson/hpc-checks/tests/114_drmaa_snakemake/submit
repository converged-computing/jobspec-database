#!/bin/sh

# Test settings
BASE=/home/projects/benchmarks/workload
test=114_drmaa_snakemake

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

## # Only have one test in queue at a time
## tst=`qstat -w | awk '{if ( $2 == "'$pbsname'" ) {print $0}}'`
## if [ x"$tst" != x ] ; then exit ; fi

# Prepare and submit test
mkdir -p $rundir && cd $rundir || exit

cat << EOF > run.pbs
#!/bin/bash
#PBS -N $pbsname
#%@FUJITSU:NSCC:CONTINUOUS_PERFORMANCE_TESTING@%#
#PBS -l select=${target}:ncpus=1:mem=1gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -P 90000001

set -e

if [ -d "\$PBS_O_WORKDIR" ] ; then cd "\$PBS_O_WORKDIR" ; fi

cp -a $datadir/* .

module load pbs-drmaa/1.0.19.fix
PATH=/app/snakemake/bin:\$PATH ; export PATH

snakemake --drmaa " -q normal -l select=1:ncpus=1:mem=1gb -l walltime=1:00:00 -v PBS_O_WORKDIR=\${PWD},project=90000001" -j 32
./hello_world
EOF

jobid=`qsub $QSUB_ARGS run.pbs`
if [ x"$ECHO_JOBID" != x ] ; then echo $jobid ; fi
echo "$test , $jobid , Q , STIME , UNKNOWN , 0 , 00:00:00 , 0.0 , HOSTNAME , $PWD " >> "$csv"
