#!/bin/bash
#SBATCH --get-user-env
#SBATCH -p prawnew
#SBATCH --mem=20GB
#SBATCH --time=0-01:00:00

#PBS -N G_zheevd
#PBS -l nodes=1, mem=20G,walltime=1:00:00

# ---------- Inputs ---------- #
DEVICE="GPU"
ROUTINE="zheevd"
# ---------- (END)Inputs ---------- #

programDIR=$(cd $(dirname $0); pwd)
HOST=$(hostname)
program=$rootDIR/bin/Diagonalization/${DEVICE}/testing_${ROUTINE}_${DEVICE}.out
OUTPUT="${ROUTINE}_${DEVICE}_(${HOST})_time.txt"
INFO="${ROUTINE}_${DEVICE}_(${HOST})_info.txt"

mkdir -p $rootDIR/data/Diagonalization
cd $rootDIR/data/Diagonalization

echo \#$(hostname)>$OUTPUT
echo \#$(hostname)>$INFO
echo \# 1.Dim of matrix 2.T_diagonalization 3.T_matrixProducts>>$OUTPUT
set +x
for n in $(seq 1 100);do $program $((500*$n)); done 1>>$INFO 2>>$OUTPUT
set -x
echo Program: $program
echo Output file: $(pwd)/$OUTPUT
