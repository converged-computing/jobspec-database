#!/bin/bash




###############################################################
#                                                             #
#    Bourne shell script for submitting a serial job to the   #
#    PBS queue using the qsub command.                        #
#                                                             #
###############################################################

#     Remarks: A line beginning with # is a comment.
#       A line beginning with #PBS is a PBS directive.
#              PBS directives must come first; any directives
#                 after the first executable statement are ignored.
#
   
##########################
#                        #
#   The PBS directives   #
#                        #
##########################

#          Set the name of the job (up to 15 characters, 
#          no blank spaces, start with alphanumeric character)

#PBS -N repertoire_creation

#          By default, the standard output and error streams are sent
#          to files in the current working directory with names:
#              job_name.osequence_number  <-  output stream
#              job_name.esequence_number  <-  error stream
#          where job_name is the name of the job and sequence_number 
#          is the job number assigned when the job is submitted.
#          Use the directives below to change the files to which the
#          standard output and error streams are sent.

#    #PBS -o stdout_file
#    #PBS -e stderr_file

#          The directive below directs that the standard output and
#          error streams are to be merged, intermixed, as standard
#          output. 

#    #PBS -j oe

#          Specify the maximum cpu and wall clock time. The wall
#          clock time should take possible queue waiting time into
#          account.  Format:   hhhh:mm:ss   hours:minutes:seconds
#          Be sure to specify a reasonable value here.
#          If the job does not finish by the time reached,
#          the job is terminated.


#PBS -l walltime=10:00:00
#PBS -J 1-20

# number of nodes
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000


######################################################################
#                                                                    #
#   To minimize communications traffic, it is best for your job      #
#   to work with files on the local disk of the compute node.        #
#   Hence, one needs to transfer files from your permanent home      #
#   directory tree to the directory ${WORKDIR} automatically         #
#   created by PBS on the local disk before program execution,       #
#   and to transfer any important output files from the local        #
#   disk back to the permanent home directory tree after program     #
#   execution is completed.                                          #
#                                                                    #
#   There are essentially two ways to achieve this: (1) to use the   #
#   PBS stagein and stageout utilities, or (2) to manually copy the  #
#   files by commands in this script.  The stagein and stageout      #
#   features of OpenPBS are somewhat awkward, especially since       #
#   wildcards and macros in the file lists cannot be used.  This     #
#   method also has some timing issues.  Hence, we ask you to use    #
#   the second method, and to use secure copy (scp) to do the file   #
#   transfers to avoid NSF bottlenecks.                              #
#                                                                    #
######################################################################

#####################################################
#                                                   #
#    Specify the permanent directory(ies) on the    #
#    server host.  Note that when the job begins    #
#    execution, the current working directory at    #
#    the time the qsub command was issued becomes   #
#    the current working directory of the job.      #
#                                                   #
#####################################################


###############################################################
#                                                             #
#    Transfer files from server to local disk.                #
#                                                             #
###############################################################

stagein()
{
 echo ' '
 echo Transferring files from server to compute node
 cp $HOME/final_20knumiter_ancestors_gen.sif ./
 echo Files in node work directory are as follows:
 ls -l
}

############################################################
#                                                          #
#    Execute the run.  Do not run in the background.       #
#                                                          #
############################################################

runprogram()
{
    module load cuda/11.4.2
    export
    nvidia-smi
    # ./run final_10_4.sif
    # export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    echo $CUDA_VISIBLE_DEVICES
    echo $PBS_ARRAY_INDEX
    time singularity -d run --nv final_20knumiter_ancestors_gen.sif $PBS_ARRAY_INDEX

}

###########################################################
#                                                         #
#   Copy necessary files back to permanent directory.     #
#                                                         #
###########################################################

stageout()
{
 echo ' '
 echo Transferring files from compute nodes to server
 ls -l
 cp -r ./results $HOME/ 
 exit
}

#####################################################################
#                                                                   #
#  The "qdel" command is used to kill a running job.  It first      #
#  sends a SIGTERM signal, then after a delay (specified by the     #
#  "kill_delay" queue attribute (set to 60 seconds), unless         #
#  overridden by the -W option of "qdel"), it sends a SIGKILL       #
#  signal which eradicates the job.  During the time between the    #
#  SIGTERM and SIGKILL signals, the "cleanup" function below is     #
#  run. You should include in this function commands to copy files  #
#  from the local disk back to your home directory.  Note: if you   #
#  need to transfer very large files which make take longer than    #
#  60 seconds, be sure to use the -W option of qdel.                #
#                                                                   #
#####################################################################

early()
{
 echo ' '
 echo ' ############ WARNING:  EARLY TERMINATION #############'
 echo ' '
 }

trap 'early; stageout' 2 9 15


##################################################
#                                                #
#   Staging in, running the job, and staging out #
#   were specified above as functions.  Now      #
#   call these functions to perform the actual   #
#   file transfers and program execution.        #
#                                                #
##################################################

stagein
runprogram
echo ' ############ End RUNPROGRAMM #############'
stageout 

###############################################################
#                                                             #
#   The epilogue script automatically deletes the directory   #
#   created on the local disk (including all files contained  #
#   therein.                                                  #
#                                                             #
###############################################################

exit
