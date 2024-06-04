#!/bin/bash -l
#OAR -n job $1
#OAR -l nodes=1/core=1,walltime=1

#module load swenv/default-env/v0.3-20170713-production
module load lang/Python/3.6.0-foss-2017a
source ./lion-environment/bin/activate
echo job $1 started
python exp_letter_A_test_GD.py $1
