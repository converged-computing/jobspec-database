#!/bin/bash
# file to submit non interactive jobs to bigred2

#PBS -l nodes=1:ppn=16
#PBS -l gres=ccm
#PBS -q debug_gpu
#PBS -l walltime=00:30:00

#PBS -o STDOUT.txt
#PBS -e STDERR.txt

#PBS -m bea
#PBS -M supun.nakandala@gmail.com

module load ccm
module load anaconda2/4.2.0
source activate tensorflow_env
ccmrun python ~/airavata/code_tf.py