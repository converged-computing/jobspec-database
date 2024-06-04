#!/bin/bash 
#PBS -j oe 
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process,walltime=40:00:00
#PBS -q gpu
#PBS -m a

chmfile=production
ngpu=1

#! Set up input file based on parameters passed to submit script
cd $PBS_O_WORKDIR
echo "Starting coords are $INPUT"
echo "Final coords are $OUTPUT"
sed -i "s/set input .*/set input $INPUT/" $chmfile.inp
sed -i "s/set output .*/set output $OUTPUT/" $chmfile.inp
module load cuda/toolkit/7.5.18
module load apps/openmm-7.0.1

#! Run nvidia-smi on visible GPUs
nvidia-smi
./get_X_GPUs.sh $ngpu
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo PBS job ID is $PBS_JOBID
echo This jobs runs on the following GPUs:
echo `cat $PBS_GPUFILE | uniq`

python -m simtk.testInstallation
charmm -i $chmfile.inp > $chmfile.out

##! Run the executable
#count=1
#flag=1
#while [ "$flag" -ne 0 ]
#do
#    echo "Running $chmfile.inp: attempt $count"
#    charmm -i $chmfile.inp > $chmfile.out
#    flag=$?
#    echo "Exit status : $flag"
#    count=$((count+1))
#done
mv $chmfile.out $INPUT.out

