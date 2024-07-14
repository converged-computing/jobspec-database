#!/bin/bash

# Run the benchmark_main several times and save the results to a file. Called by
# scipts/sbatch/benchmark.sbatch It is expected that the \*.tdbg file is within
# benchmark_objects/index folder

if [ $# -ne 2 ]; then
  echo "Usage: ./scripts/benchmark/index_search_d20.sh <output_file> <nvidia|amd>"
  exit 1
fi

## Set loglevel to trace because we use this to get timing statistics
export SPDLOG_LEVEL=TRACE

benchmark_out="$1"

if [ ! -d "benchmark_objects/index" ] || [ ! -f "benchmark_objects/index/index.tdbg" ]; then
  echo "This script expects a 'benchmark_objects/index' folder with 'index.tdbg' to be present within that folder"
  exit 1
fi

dbg_file="benchmark_objects/index/index.tdbg"
input_files=(
  "benchmark_objects/list_files/input/unzipped_seqs.list"
  "benchmark_objects/list_files/input/zipped_seqs.list"
)
output_file="benchmark_objects/list_files/output/index_search_results_running.list"
printing_modes=(
  "ascii"
  "binary"
  "bool"
)

if [ $2 = "nvidia" ]; then
  devices=("nvidia")
elif [ $2 = "amd" ]; then
  devices=("amd")
else
  echo "2nd argument is incorrect"
fi

streams_options=(1 2 3 4 5 6 7 8)

for device in "${devices[@]}"; do
  . scripts/build/release.sh ${device} >&2
  for streams in "${streams_options[@]}"; do
    for input_file in "${input_files[@]}"; do
      for printing_mode in "${printing_modes[@]}"; do
        echo "Now running: File ${input_file} with ${streams} streams in ${printing_mode} format on ${device} device"
        echo "Now running: File ${input_file} with ${streams} streams in ${printing_mode} format on ${device} device" >> "${benchmark_out}"
        ./build/bin/sbwt_search index \
          -i "${dbg_file}" \
          -q "${input_files}" \
          -o "${output_file}" \
          -p "${printing_mode}" \
          -s "${streams}" \
          -u 10GB \
          -k "benchmark_objects/index/index_d20.tcolors" \
          >> "${benchmark_out}"
        printf "Size of outputs: "
        ls -lh "benchmark_objects/running" | head -1
        rm benchmark_objects/running/*
      done
    done
  done
done
