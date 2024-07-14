#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=00:30:00
#PBS -N preparePRF
#PBS -V

module load singularity 2> /dev/null

rm -rf output
rm -f finished

mkdir surfaces

time singularity exec -e docker://brainlife/freesurfer:6.0.0 bash -c "echo $FREESURFER_LICENSE > /usr/local/freesurfer/license.txt && ./convert.sh"

singularity exec -e docker://stevengeeky/python-nibabel ./main.py

if [ -f "surfaces/surfaces.json" ]
then
    echo 0 > finished
else
    echo "surfaces/surfaces.json not found"
    echo 1 > finished
    exit 1
fi