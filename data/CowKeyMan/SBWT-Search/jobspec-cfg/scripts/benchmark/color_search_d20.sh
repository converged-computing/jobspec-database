#!/bin/bash

# Run the benchmark_main several times and save the results to a file. Called by
# scipts/sbatch/benchmark.sbatch It is expected that the \*.tdbg file is within
# benchmark_objects/colors folder

if [ $# -ne 2 ]; then
  echo "Usage: ./scripts/benchmark/index_search_d20.sh <output_file> <nvidia|amd>"
  exit 1
fi

## Set loglevel to trace because we use this to get timing statistics
export SPDLOG_LEVEL=TRACE

benchmark_out="$1"

if [ ! -d "benchmark_objects/index" ] || [ ! -f "benchmark_objects/index/index_d20.tcolors" ]; then
  echo "This script expects a 'benchmark_objects/index' folder with 'index_d20.tcolors' to be present within that folder"
  exit 1
fi

colors_file="benchmark_objects/index/index_d20.tcolors"
input_files=(
  "benchmark_objects/list_files/input/index_search_results_d20_ascii.list"
  "benchmark_objects/list_files/input/index_search_results_d20_binary.list"
)
input_files_aliases=(
  "AsciiIndexes"
  "BinaryIndexes"
)

output_file="benchmark_objects/list_files/output/color_search_results_running.list"
printing_modes=(
  "ascii"
  "binary"
  # "csv"
)
if [ $2 = "nvidia" ]; then
  devices=("nvidia")
elif [ $2 = "amd" ]; then
  devices=("amd")
else
  echo "2nd argument is incorrect"
fi

streams_options=(1 2 3 4 5 6 7 8)

. scripts/build/release.sh ${devices[0]} >&2

./build/bin/sbwt_search index \
  -i "benchmark_objects/index/index.tdbg" \
  -q "benchmark_objects/list_files/input/unzipped_seqs.list" \
  -o "benchmark_objects/list_files/output/index_search_results_d20_ascii.list" \
  -s "4" \
  -p "ascii" \
  -u 10GB \
  -k "benchmark_objects/index/index_d20.tcolors" \
  >> /dev/null

./build/bin/sbwt_search index \
  -i "benchmark_objects/index/index.tdbg" \
  -q "benchmark_objects/list_files/input/unzipped_seqs.list" \
  -o "benchmark_objects/list_files/output/index_search_results_d20_binary.list" \
  -s "4" \
  -p "binary" \
  -u 10GB \
  -k "benchmark_objects/index/index_d20.tcolors" \
  >> /dev/null

for device in "${devices[@]}"; do
  . scripts/build/release.sh ${device} >&2
  for streams in "${streams_options[@]}"; do
    for input_file_idx in "${!input_files[@]}"; do
      for printing_mode in "${printing_modes[@]}"; do
        echo "Now running: File ${input_files_aliases[input_file_idx]} with ${streams} streams in ${printing_mode} format on ${device} device"
        echo "Now running: File ${input_files_aliases[input_file_idx]} with ${streams} streams in ${printing_mode} format on ${device} device" >> "${benchmark_out}"
        ./build/bin/sbwt_search colors \
          -k "${colors_file}" \
          -q "${input_files[input_file_idx]}" \
          -o "${output_file}" \
          -s "${streams}" \
          -p "${printing_mode}" \
          -u 10GB \
          -t 0.7 \
          >> "${benchmark_out}"
        printf "Size of outputs: "
        ls -lh "benchmark_objects/running" | head -1
        if [ "${printing_mode}" = "ascii" ]; then
          diff -qr "benchmark_objects/running" "benchmark_objects/color_search_results_t0.7"
        fi
        rm benchmark_objects/running/*
      done
    done
  done
done

rm benchmark_objects/index_search_results_d20_binary/*
