#!/bin/bash
#PBS -b <%= nodes %>
#PBS -q gpu
#PBS -T openmpi
#PBS -A NBB
#PBS -v NQSV_MPI_VER=4.1.5/gcc9.4.0-cuda11.8.0
#PBS -l elapstim_req=00:10:00

BIN=tag

module load use.own
module load "openmpi/${NQSV_MPI_VER}"
module load "ucx/1.13.1/cuda11.8.0"

export RUST_LOG=Info
export SERVER_THREAD_COUNT="<%= server_thread_count %>"
export CLIENT_THREAD_COUNT="<%= client_thread_count %>"
export CLIENT_TASK_COUNT="<%= client_task_count %>"
export RUST_BACKTRACE="1"

mpirun ${NQSV_MPIOPTS} \
    -x RUST_BACKTRACE \
    -x RUST_LOG \
    -x SERVER_THREAD_COUNT \
    -x CLIENT_THREAD_COUNT \
    -x CLIENT_TASK_COUNT \
    -np 2 \
    --report-bindings \
    --map-by ppr:1:node:PE=48 \
    "/work/0/NBB/mnakano/bench-async-ucx/target/release/${BIN}"
