#!/bin/sh

#OAR --array-param-file params.txt
#OAR -l /nodes=1,walltime=6:00:00
#OAR -p host="orval02"
#OAR -t besteffort
#OAR --notify mail:jerome.buisine@univ-littoral.fr
#OAR -O /nfs/home/lisic/jbuisine/projects/launchers/logs/Thesis-NoiseDetection-CNN.%jobid%.out
#OAR -E /nfs/home/lisic/jbuisine/projects/launchers/logs/Thesis-NoiseDetection-CNN.%jobid%.err

# Activiate venv used by python
. ~/opt/venvs/thesis-venv/bin/activate

# run command
python ~/projects/Thesis-NoiseDetection-CNN/generate/generate_reconstructed_data.py $@