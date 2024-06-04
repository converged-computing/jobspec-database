#!/bin/bash -l
#OAR -n job $1
#OAR -l nodes=1/core=1,walltime=1

module load swenv/default-env/v0.1-20170602-production
module load lang/Python/3.6.0-foss-2017a
source ../collectedForHPC/lion-environment/bin/activate
echo job $1 started
python lion_extended_accuracy_plot_data.py $1
