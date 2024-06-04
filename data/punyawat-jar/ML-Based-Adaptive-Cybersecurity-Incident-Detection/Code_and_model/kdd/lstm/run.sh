#!/bin/bash
#PBS -q GPU-1
#PBS -N LSTMKDD
#PBS -l select=1
#PBS -j oe


source /etc/profile.d/modules.sh
module load singularity/3.9.5
cd ${/home/s2316002/capstone_project}
singularity exec /home/s2316002/capstone_project/runcode.sif python /home/s2316002/ML-Based-Adaptive-Cybersecurity-Incident-Detection/Code_and_model/kdd/lstm/lstm.py