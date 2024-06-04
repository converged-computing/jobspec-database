#!/bin/bash

#SBATCH --time=120
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --partition=west
#SBATCH --output=job.out
#SBATCH --error=job.err

export ENSEMBLES_MPIEXEC_PATH=mpiexec
export ENSEMBLES_MPIEXEC_NODES=1
export ENSEMBLES_NUMIO_PATH=/home/oosting/Numio/numio-posix
export ENSEMBLES_FIND_PATH=find
export ENSEMBLES_FIND_SEARCH_PATH="/home/oosting"
export ENSEMBLES_NUMIO_READ=True
export ENSEMBLES_READ_PATH="/home/oosting/numnew/r"
export ENSEMBLES_WRITE_PATH="/home/oosting/numnew/w"
export ENSEMBLES_NUMIO_W_NOFILESYNC=False
export ENSEMBLES_NUMIO_W_IMMEDIATE=False # can't be activee at the same time as nofilesync
export ENSEMBLES_NUMIO_FPISIN=True
export ENSEMBLES_NUMIO_FILE_PER_PROCESS=True
export ENSEMBLES_ITERATIONS=9000 # count of numio iterations
export ENSEMBLES_LINES=500 # how large the numio matrix should be |lines| = |columns|
export ENSEMBLES_PERT=True # use the sinus perturbation function
export ENSEMBLES_NUMIO_WRITE=True
export ENSEMBLES_NUMIO_RW_FREQUENCY=64 # which frequency to read and write with (read is frequency + 1 since numio will break otherwise)
export ENSEMBLES_NUMIO_RW_PATH=matrix.out # what path to write the matrix to
export ENSEMBLES_NUMIO_R_IMMEDIATE=True
export ENSEMBLES_NUMIO_COLLECTIVE_COMMS=True
export ENSEMBLES_NUMIO_COLLECTIVE_COM_SIZE=200 # how much data to send during fake communication
export ENSEMBLES_NUMIO_COLLECTIVE_COM_FREQ=64 # how often to write data using fake communication
export ENSEMBLES_NUMIO_ASYNC_WRITE=False
export ENSEMBLES_NUMIO_GZIP=False
export ENSEMBLES_NUMIO_LZ4=False
export ENSEMBLES_NUMIO_CHUNK_MULTIPLIER=1
export ENSEMBLES_NUMIO_COMPRESS_LEVEL=1
export ENSEMBLES_NUMIO_THREADS=1
export ENSEMBLES_IDLE_ONLY=False # just sleep instead of running numio
export ENSEMBLES_IDLE_ONLY_TIME=12 # when sleeping instead of running numio wait for x seconds
export ENSEMBLES_GRAPH_OUT_PATH=plots # where to store the generated graphs (make sure the folder exists)
export ENSEBLES_DATA_OUT_PATH=data # where to store the generated log data (make sure the folder exists)
export ENSEMBLES_GRAPH_FILETYPE=pdf # supported file extension for matplotlib
export ENSEMBLES_BACKGROUND_PROCESS_LIST='["iperf"]' # json list of background processes to run
export ENSEMBLES_LOG_EVERY_X_SECONDS=3
export ENSEMBLES_IPERF_PATH="iperf3"
export ENSEMBLES_IPERF_SERVER_IP="136.172.61.247"
export ENSEMBLES_IPERF_PORT=5201


mpiexec python src/main.py # run the script