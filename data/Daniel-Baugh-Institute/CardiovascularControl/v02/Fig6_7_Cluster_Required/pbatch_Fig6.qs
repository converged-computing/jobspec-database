#!/bin/bash -l
#
# DARWIN job script template, generated 2021-09-16T15:37:44-0400
#
# Sections of this script that can/should be edited are delimited by a
# [EDIT] tag.  All Slurm job options are denoted by a line that starts
# with "#SBATCH " followed by flags that would otherwise be passed on
# the command line.  Slurm job options can easily be disabled in a
# script by inserting a space in the prefix, e.g. "# SLURM " and
# reenabled by deleting that space.
#
# This is a batch job template for Matlab scripts, which will only run
# with multiprocess/thread parallelism on our cluster.
#
# Do not alter the --nodes option!
#SBATCH --nodes=1
#
# [EDIT] Alter the --ntasks for Distributed Computing jobs that
#        will use more than one local task.
#
#SBATCH --ntasks=64
#
# [EDIT] If you want Matlab to use thread parallelism (in itself
#        or in each Distributed Computing task) then change the
#        --cpus-per-task to something other than 1.  NOTE:  with
#        Distributed Computing it is usually not optimal to change
#        this from 1.
#
#SBATCH --cpus-per-task=1
#
#SBATCH --mem-per-cpu=12G
#
# [EDIT] It can be helpful to provide a descriptive (terse) name for
#        the job (be sure to use quotes if there's whitespace in the
#        name):
#
#SBATCH --job-name=RSA_GPR_robust
#
# [EDIT] The partition determines which nodes can be used and with what
#        maximum runtime limits, etc.  Partition limits can be displayed
#        with the "sinfo --summarize" command.
#
#        PLEASE NOTE:  On DARWIN every job is **required** to include the
#                      --partition flag in its submission!
#
#SBATCH --partition=idle
# [EDIT] Jobs that will run in one of the GPU partitions can request GPU
#        resources using ONE of the following flags:
#
#          --gpus=<count>
#              <count> GPUs total for the job, regardless of node count
#          --gpus-per-node=<count>
#              <count> GPUs are required on each node allocated to the job
#          --gpus-per-socket=<count>
#              <count> GPUs are required on each socket allocated to the
#              job
#          --gpus-per-task=<count>
#              <count> GPUs are required for each task in the job
#
#       PLEASE NOTE:  On DARWIN the --gres flag should NOT be used to
#                     request GPU resources.  The GPU type will be
#                     inferred from the partition to which the job is
#                     submitted if not specified.
#
# SBATCH --gpus=1
# SBATCH --gpus-per-task=1
# SBATCH --gpus-per-node=1
# SBATCH --gpus-per-socket=2
#
# [EDIT] The maximum runtime for the job; a single integer is interpreted
#        as a number of minutes, otherwise use the format
#
#          d-hh:mm:ss
#
#        Jobs default to the default runtime limit of the chosen partition
#        if this option is omitted.
#
#SBATCH --time=0-01:25:00
#
#        You can also provide a minimum acceptable runtime so the scheduler
#        may be able to run your job sooner.  If you do not provide a
#        value, it will be set to match the maximum runtime limit (discussed
#        above).
#
# SBATCH --time-min=0-00:10:00
#
# [EDIT] By default SLURM sends the job's stdout to the file "slurm-<jobid>.out"
#        and the job's stderr to the file "slurm-<jobid>.err" in the working
#        directory.  Override by deleting the space between the "#" and the
#        word SBATCH on the following lines; see the man page for sbatch for
#        special tokens that can be used in the filenames:
#
# SBATCH --output=%x-%j.out
# SBATCH --error=%x-%j.out
#
# [EDIT] Slurm can send emails to you when a job transitions through various
#        states: NONE, BEGIN, END, FAIL, REQUEUE, ALL, TIME_LIMIT,
#        TIME_LIMIT_50, TIME_LIMIT_80, TIME_LIMIT_90, ARRAY_TASKS.  One or more
#        of these flags (separated by commas) are permissible for the
#        --mail-type flag.  You MUST set your mail address using --mail-user
#        for messages to get off the cluster.
#
# SBATCH --mail-user='xxx@xxx.edu'
# SBATCH --mail-type=ALL
#
# [EDIT] By default we DO NOT want to send the job submission environment
#        to the compute node when the job runs.
#
#SBATCH --export=NONE
#
#
# [EDIT] If you're not interested in how the job environment gets setup,
#        uncomment the following.
#
#UD_QUIET_JOB_SETUP=YES

# [EDIT] Define a Bash function and set this variable to its
#        name if you want to have the function called when the
#        job terminates (time limit reached or job preempted).
#
#        PLEASE NOTE:  when using a signal-handling Bash
#        function, any long-running commands should be prefixed
#        with UD_EXEC, e.g.
#
#                 UD_EXEC mpirun vasp
#
#        If you do not use UD_EXEC, then the signals will not
#        get handled by the job shell!
#
#job_exit_handler() {
#  # Copy all our output files back to the original job directory:
#  cp * "$SLURM_SUBMIT_DIR"
#
#  # Don't call again on EXIT signal, please:
#  trap - EXIT
#  exit 0
#}
#export UD_JOB_EXIT_FN=job_exit_handler

#
# [EDIT] By default, the function defined above is registered
#        to respond to the SIGTERM signal that Slurm sends
#        when jobs reach their runtime limit or are
#        preempted.  You can override with your own list of
#        signals using this variable -- as in this example,
#        which registers for both SIGTERM and the EXIT
#        pseudo-signal that Bash sends when the script ends.
#        In effect, no matter whether the job is terminated
#        or completes, the UD_JOB_EXIT_FN will be called.
#
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"

#
# Load a specific Matlab package into the runtime environment:
#
vpkg_require matlab/default

#
# Do standard OpenMP environment setup:
#
. /opt/shared/slurm/templates/libexec/openmp.sh

#
# Execute your Matlab script(s) here; prefix with UD_EXEC to
# ensure the job can/will respond to preemption/termination
# signals by calling your UD_JOB_EXIT_FN.
#
# Duplicate all three commands for each Matlab script you run
# in sequence below.
#
UD_EXEC matlab -nodisplay -batch RSA_GPR_robust_v3
matlab_rc=$?
if [ $matlab_rc -ne 0 ]; then exit $matlab_rc; fi

