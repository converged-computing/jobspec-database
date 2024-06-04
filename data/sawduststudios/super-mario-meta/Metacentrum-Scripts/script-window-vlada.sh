#!/bin/bash
#PBS -q default@cerit-pbs.cerit-sc.cz
#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=1gb:spec=5.1:cluster=gita:pbs_server=cerit-pbs.cerit-sc.cz:cpu_vendor=amd
#PBS -e /storage/brno2/home/horkyvl/job_logs
#PBS -o /storage/brno2/home/horkyvl/job_logs

DATADIR=/storage/brno2/home/horkyvl/repo/super-mario-meta
RESULTDIR=/storage/brno2/home/horkyvl/results/window
ROOT=/storage/brno2/home/horkyvl

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add openjdk-17

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -R $DATADIR/Mario-AI-Framework $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $SCRATCHDIR/Mario-AI-Framework

# compile and run Java
javac -cp src src/mff/agents/benchmark/AgentBenchmarkMetacentrum.java
java -cp src mff.agents.benchmark.AgentBenchmarkMetacentrum $searchSteps $rightBorderWindow

# move output to data dir
#cp -a agent-benchmark/. $ROOT || { echo >&2 "Error while copying output file(s) to root!"; exit 2; }
mkdir -p $RESULTDIR
cp -a agent-benchmark/. $RESULTDIR/ || { echo >&2 "Error while copying output file(s) to result dir!"; exit 2; }
#cd agent-benchmark
#cp astar.csv $DATADIR

clean_scratch
