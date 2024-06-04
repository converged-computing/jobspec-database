# that is required. 
#
# In this example, we request 12 cores per node on a single node and
# 47 GB of RAM. Requesting an amount of RAM is useful for servers on which 
# many jobs can run on a single node. This will help the scheduler
# to launch the correct number of jobs on a node. If we request a full node
# we do not always need to request how much RAM is required. It is however
# useful for clusters that do not have the same amount of memory on every node.
# This will tell the scheduler to choose a node with enough memory.
#
#PBS -l nodes=1:ppn=16:gpus=2
#
# Be careful how much RAM you are requesting. For example, on a server which have
# nodes with 48 GB of RAM, requesting -l mem=48gb will not work. This is because
# in reality, part of the memory is not available for the jobs. You need to request
# less memory than the maximum amount.
 
# --------------------------------------------------------------
#
# Resources requested: wallclock during which the job can run.
# Here, 3 hours.
#
#PBS -l walltime=15:00:00
 
# Name of the job (default value is the name of the script, possibly truncated).
#
#PBS -N DBN-D-75
 
# --------------------------------------------------------------
#
# Standard output file (stdout)
#
#PBS -o ./output-DBN-D-75.out
#
# (by default, the name of the job with the extension .oNNNN where
# NNNN is the job ID)
#
# By default, we cannot use environment variables within the name of the
# output file except those exported with the -v or -V options (see below), or
# the PBS variables (see below).
 
# Merge the standard error and output.
#
#PBS -j oe
 
# --------------------------------------------------------------
#
# File which will contain the standard error (stderr) if we do not use the
# -j oe option above.
#  #PBS -e path/output.err
# (by default, the job name with extension .eNNNN)
 
# Permissions of the error and output files. By default, those files
# can only be read by the user submitting the job. Here, we add reading
# permissions for everybody. Note: this option does not work with the
# Torque version installed on Briarée, but should work on all other servers.
#
#PBS -W umask=022
 
# --------------------------------------------------------------
#
# Name of the submission queue for the job. A default queue will be used if
# this option is not provided. To know which queue to us, see the page
# "Running jobs" on our wiki. It is often not useful to specify this.
#
# #PBS -q soumet
 
# --------------------------------------------------------------
#
# Files which must be copied on the compute node (in a folder that already exists)
# before the script is run. You can also make copies within the script.
# In the following example, the compute node is the local host.
# Files within a shared folder or on a parallel file system such as $HOME or $SCRATCH
# do not need to be copied.
# 
# #PBS -W stagein=local-file@remote-host:remote-file
 
# --------------------------------------------------------------
#
# Files on the compute node which need to be copied elsewhere after
# running the job.
# 
# #PBS -W stageout=local-file@remote-host:remote-file
#
# All files specified in the stagein and stageout are erased from the
# compute host after the job has finished.
 
# --------------------------------------------------------------
#
# Time after which PBS will queue the job
# MMDDhhmm ( warning: hours are based on 24... )
#      ( example : February 6th at 15h06 will be 02061506 ) 
# 
# #PBS -a 02061506
 
# --------------------------------------------------------------
#
# Sending of email:
#       
#                "b" : when the job begins
#                "e" : when the job ends
#                "a" : when the job aborts (ends with an error)
#
#    Warning! If you use multiple email options, they must be groupe
#    within the same line. Otherwise, only the last line will be considered.
#
#PBS -m bea
 
# Defining an email address. If you want to send emails to a specific
# email address, specify it with the following option
#
#PBS -M cedric.noiseux15@gmail.com
 
# --------------------------------------------------------------
#
# Tells wheither the job can be interrupted and started from scratch 
# without undesirable side effects. Warning, on some servers, the jobs
# are restartable by default.
#
#PBS -r n
 
# --------------------------------------------------------------
#
# Exporting environment variables from the submission shell to the job shell
#
#  #PBS -V
#
#
# Environment variables available within a job's environment
#     
#   Predefined variables:
#         Variables defined on the compute host
#         Variables transfered from the submission host with the 
#         -v option (specific variables) or -V (all variables)
#   PBS-defined variables
#
# Variables describing a job's submission environment 
#
#     PBS_O_HOST    Host from which the job was submitted
#     PBS_O_LOGNAME Username that submitted the job
#     PBS_O_HOME    Home directory on the submit host
#     PBS_O_WORKDIR Working directory (i.e. directory from which the job was submitted)
#
# Variables describing the job's running environment
#     PBS_ENVIRONMENT
#       two possible values: 
#                  PBS_BATCH: job to be run in batch mode
#                  PBS_INTERACTIVE: interactive job submitted with the -I option
#     PBS_O_QUEUE   Queue in which the job has been initially submitted
#     PBS_QUEUE     Queue in which the job is running
#     PBS_JOBID     PBS job ID
#     PBS_JOBNAME   Job name
#
 
# Dependencies
#
# The job can be run after job 344 has begun running.
# #PBS -W depend=after:344
#
# The job can run when job 345 has ended successfully (will not be run if it fails)
# #PBS -W depend=afterok:345
#
# The job can be run if job 346 ends with an error (will not be run if it succeeds)
# #PBS -W depend=afternotok:346
#
# The job can be run whenever job 347 ends (with an error or not)
# #PBS -W depend=afterany:347
 
# We start a job array, for qsub. Jobs are here numbered from 1 to 100.
# The job script must use the environment variable $PBS_ARRAYID to 
# execute different tasks on different jobs.
# 
# #PBS -t 1-100
#
# We can also give a list of indices or ranges separated by commas:
#
#     #PBS -t 1-5,10,21-25
#
# We can also add a limit on the number of tasks to run at the same time:
#
#     #PBS -t 1-100%5
#
# No more than 5 tasks will be run at the same time in the example above.
 
# With msub, the syntax is slightly different:
#
#    #PBS -t [1-100]
#
# and the environment variable is $MOAB_JOBARRAYINDEX.
# To know the number of tasks within a job array, you can refer to the variable
# $MOAB_JOBARRAYRANGE. You can also skip values within the range:
#
#    #PBS -t [1-101:10]%7
#
# The example above will run the script with inexes 1, 11, 21, 31, ..., 101
# and will run no more than 7 tasks at the same time.
 
#Other options available only in command line:
#
#     qsub -I -X -lnodes=1:ppn=12 -lwalltime=1:0:0
#
#  or
#
#     qsub -I -X script
#
# The -I option will start an interactive job. The script is optional in this case.
# If a script is specified, it will only help gives further options to PBS with the #PBS directives.
#
# The -X option (which requires -I) redirects graphic output through X Window.
# This will only work if you can do X window from the submit host and the compute node.
 
# --------------------------------------------------------------
# The first line not starting with a # ends all PBS directives. 
###

# setup just in case

module load iomkl/2015b
module load scikit-learn/0.16.1-Python-3.5.0
module load foss/2015b
module load Tensorflow/1.0.0-Python-3.5.2

# create my env copy ...

virtualenv $SCRATCH/deep-belief-network/tensorflow

# activate my virtualenv

source $SCRATCH/deep-belief-network/tensorflow/bin/activate

# install packages

pip install tensorflow
pip install pyyaml
pip install scikit-learn
pip install scipy
pip install numpy
pip install scikit-cuda
pip install -U nltk
pip install keras

# run training

# In general, you should change the directory at the beginning of the script because 
# the default when the job starts is $HOME
cd $SCRATCH/deep-belief-network

python classification_demo.py --hidden_layers_structure "1000,1000,1000" --learning_rate_rbm 0.01 --learning_rate 0.1 --n_epochs_rbm 100 --n_iter_backprop 500 --dropout_p 0.75

















