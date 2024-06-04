#!/bin/sh
### Set the job name (for your reference)
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=04:00:00

#PBS -l software=python
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/ucs4/gnu/447/pip/9.0.1/gnu
module load pythonpackages/3.6.0/ucs4/gnu/447/setuptools/34.3.2/gnu
module load pythonpackages/3.6.0/ucs4/gnu/447/wheel/0.30.0a0/gnu
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/scipy/1.1.0/gnu
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load pythonpackages/3.6.0/ucs4/gnu/447/six/1.10.0/gnu
module load pythonpackages/3.6.0/absl-py/0.4.0/gnu
module load pythonpackages/3.6.0/h5py/2.8.0/gnu
module load pythonpackages/3.6.0/markdown/2.6.11/gnu
module load pythonpackages/3.6.0/protobuf/3.6.1/gnu
module load pythonpackages/3.6.0/pyyaml/3.13/gnu
module load pythonpackages/3.6.0/tensorflow_tensorboard/1.10.0/gnu
module load pythonpackages/3.6.0/scikit-learn/0.21.2/gnu
module load apps/pythonpackages/3.6.0/tensorflow/1.9.0/gpu
module load apps/pythonpackages/3.6.0/keras/2.2.2/gpu
module load pythonpackages/3.6.0/networkx/2.1/gnu
#job 

python main.py Yeast/167.txt_graph gen_graph.txt > output.txt
#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
