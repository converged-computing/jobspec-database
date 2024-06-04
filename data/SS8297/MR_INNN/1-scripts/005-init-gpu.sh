#!/usr/bin/env bash
#SBATCH -A sens2022521
#SBATCH -p core
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH --gpus-per-node 1
#SBATCH -o /proj/sens2022521/1-shuai/9-logs/%u-slurm-%j.out
#SBATCH --mail-user shuai1997@hotmail.se

# set -euo pipefail
cd /proj/sens2022521/MindReader
julia --project "/proj/sens2022521/MindReader/src/ReadMind.jl" \
  --input "0001LB.edf" \
  --inputDir "/proj/sens2022521/EEGcohortMX/" \
  --params "Parameters.jl" \
  --paramsDir "/proj/sens2022521/MindReader/src/" \
  --annotation "0001LB.xlsx" \
  --annotDir "/proj/sens2022521/EEGcohortMX/" \
  --outDir "/proj/sens2022521/1-shuai/2-results/" \
  --additional "annotationCalibrator.jl,fileReaderXLSX.jl" \
  --addDir "/proj/sens2022521/EEG/src/annotation/functions/"
cd /proj/sens2022521/1-shuai/1-scripts


