#!/bin/bash

#run_gadi.pbs
#Execute with "qsub run_gadi.pbs"

#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l walltime=00:10:00
#PBS -l mem=10GB
#PBS -l ngpus=1
#PBS -l wd
#PBS -l storage=scratch/md01
#PBS -P md01
#PBS -N detect
#PBS -j oe

module load singularity
module load cuda/11.2.2
cd $PBS_O_WORKDIR

/usr/bin/time -v singularity run --nv mega.img /bin/bash -c "python /build/cameratraps/detection/run_detector_batch.py /build/blobs/md_v5b.0.0.pt /build/cameratraps/test_images/test_images/ /scratch/md01/npb56/mdv4test.json --output_relative_filenames --recursive"
