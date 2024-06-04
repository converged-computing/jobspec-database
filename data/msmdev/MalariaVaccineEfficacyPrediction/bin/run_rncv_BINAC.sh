#!/bin/bash

# Copyright (c) 2022 Bernhard Reuter and Jacqueline Wistuba-Hamprecht.
# ------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------------------------------------
# Jacqueline Wistuba-Hamprecht and Bernhard Reuter (2022)
# https://github.com/msmdev/MalariaVaccineEfficacyPrediction
# ------------------------------------------------------------------------------------------------
# This is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------------------------

# @Author: Bernhard Reuter

###############################################################
#                                                             #
#    Bourne shell script for submitting a job array to the   #
#    PBS queue using the qsub command.                        #
#                                                             #
###############################################################

#     Remarks: A line beginning with # is a comment.
#            A line beginning with #PBS is a PBS directive.
#              PBS directives must come first; any directives
#                 after the first executable statement are ignored.
#
   
##########################
#                        #
#   The PBS directives   #
#                        #
##########################

# #PBS -W x=ADVRES:tu_iizre01.30

# #PBS -S /bin/bash  

#          Specifies the task ids of a job array. Single task arrays
#          are allowed. The array_request argument is an integer id 
#          or a range of integers. Multiple ids or id ranges can be 
#          combined in a comma delimted list.  
#          Examples  :  -t  1-100  or  -t 1,10,50-100
#          An optional  slot  limit  can be specified to limit the 
#          amount of jobs that can run concurrently in the job array. 
#          The default value is unlimited. The slot limit must be the 
#          last thing specified in the array_request and is delimited 
#          from the array by a percent sign (%).
#          Example: qsub script.sh -t 0-299%5
#          This sets the slot limit to 5. Only 5 jobs from this array 
#          can run at the same time.
#          Note: You can use qalter to modify slot limits on an array.
#          The server parameter max_slot_limit can be used to set a 
#          global slot limit policy. 

# #PBS -t 1-16099

#          Set the name of the job (up to 15 characters, 
#          no blank spaces, start with alphanumeric character)

# #PBS -N "${DRUG}BCSf"

#          By default, the standard output and error streams are sent
#          to files in the current working directory with names:
#              job_name.osequence_number  <-  output stream
#              job_name.esequence_number  <-  error stream
#          where job_name is the name of the job and sequence_number 
#          is the job number assigned when the job is submitted.
#          Use the directives below to change the files to which the
#          standard output and error streams are sent.

# #PBS -o stdout_file
# #PBS -e stderr_file

#          The directive below directs that the standard output and
#          error streams are to be merged, intermixed, as standard
#          output. 

# #PBS -j oe

#          Specify the maximum cpu and wall clock time. The wall
#          clock time should take possible queue waiting time into
#          account.  Format:   hhhh:mm:ss   hours:minutes:seconds
#          Be sure to specify a reasonable value here.
#          If the job does not finish by the time reached,
#          the job is terminated.

# #PBS -l     cput=6:00:00
# #PBS -l walltime=168:00:00

#          Specify the queue.  The CMU cluster currently has three queues:
#          "green", "blue", and "red".  Jobs submitted to these queues
#          will run in cpu-dedicated mode; if all cpu's assigned to the
#          queue are occupied with a job, then new jobs are queued and will
#          not run until a cpu is freed up.  You should take this waiting
#          time into account when setting "walltime". 

# #PBS -q long

#          Specify the maximum amount of physical memory required.
#          kb for kilobytes, mb for megabytes, gb for gigabytes.
#          Take some care in setting this value.  Setting it too large
#          can result in your job waiting in the queue for sufficient
#          resources to become available.

# #PBS -l mem=120gb

#          PBS can send informative email messages to you about the
#          status of your job.  Specify a string which consists of
#          either the single character "n" (no mail), or one or more
#          of the characters "a" (send mail when job is aborted),
#          "b" (send mail when job begins), and "e" (send mail when
#          job terminates).  The default is "a" if not specified.
#          You should also specify the email address to which the
#          message should be send via the -M option.

#PBS -m abe
# #PBS -m ae

#PBS -M bernhard.reuter@uni-tuebingen.de

#          Declare the time after which the job is eligible for execution.
#          If you wish the job to be immediately eligible for execution,
#          comment out this directive.  If you wish to run at some time in 
#          future, the date-time argument format is
#                      [DD]hhmm
#          If the day DD is not specified, it will default to today if the
#          time hhmm is in the future, otherwise, it defaults to tomorrow.
#          If the day DD is specified as in the future, it defaults to the
#          current month, otherwise, it defaults to next month.

# #PBS -a 2215  commented out

#          Specify the priority for the job.  The priority argument must be
#          an integer between -1024 and +1023 inclusive.  The default is 0.

# #PBS -p 0

#          Specify the number of nodes requested and the
#          number of processors per node. 

# #PBS -l nodes=1:ppn=28

#          Define the interval at which the job will be checkpointed,
#          if checkpointing is desired, in terms of an integer number
#          of minutes of CPU time.

# #PBS -c c=2

# # check if $SCRIPT_FLAGS is "set"
# if [ -n "${SCRIPT_FLAGS}" ] ; then
#     ## but if positional parameters are already present
#     ## we are going to ignore $SCRIPT_FLAGS
#     if [ -z "${*}"  ] ; then
# 	set -- ${SCRIPT_FLAGS}
#     fi
# fi

##########################################
#                                        #
#   Output some useful job information.  #
#                                        #
##########################################

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat "$PBS_NODEFILE"
echo ------------------------------------------------------
echo PBS: qsub is running on "$PBS_O_HOST"
echo PBS: originating queue is "$PBS_O_QUEUE"
echo PBS: executing queue is "$PBS_QUEUE"
echo PBS: working directory is "$PBS_O_WORKDIR"
echo PBS: execution mode is "$PBS_ENVIRONMENT"
echo PBS: job identifier is "$PBS_JOBID"
echo PBS: job name is "$PBS_JOBNAME"
echo PBS: node file is "$PBS_NODEFILE"
echo PBS: current home directory is "$PBS_O_HOME"
echo PBS: PATH = "$PBS_O_PATH"
echo ------------------------------------------------------

##########################################

# give the scheduler some time to "sort himself"
sleep 30

cd "$PBS_O_WORKDIR" || { echo "Couldn't cd into ${PBS_O_WORKDIR}"; exit 1; }

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate malaria_env

err="runRNCV_${PBS_JOBNAME}_${PBS_JOBID}.err"
out="runRNCV_${PBS_JOBNAME}_${PBS_JOBID}.out"

if [ "$METHOD" = 'multitaskSVM' ]; then
    combinations=('RPP' 'RPR' 'RRP' 'RRR')
    cp -r "${KERNEL_DIR}/"* "$TMPDIR" || { echo "Couldn't copy Kernels to ${TMPDIR}"; exit 1; }
    for combination in "${combinations[@]}"; do
        ana_dir="${ANA_DIR}/${combination}"
        cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
        python -u rncv.py --analysis-dir "$ana_dir" --data-file "$DATA_FILE" --method "$METHOD" --njobs="$NJOBS" --combination "$combination" --kernel-identifier "$IDENTIFIER" --kernel-dir "$TMPDIR" 1> "${out}" 2> "${err}"
    done
else
    python -u rncv.py --analysis-dir "$ANA_DIR" --data-file "$DATA_FILE" --method "$METHOD" --njobs="$NJOBS" 1> "${out}" 2> "${err}"
fi

exit
