#!/bin/bash
#
#-----------------------------------------------------------------------------
# This Maverick job script is designed to create an ipython session on 
# visualization nodes through the SLURM batch system. Once the job
# is scheduled, check the output of your job (which by default is
# stored in your home directory in a file named ipython.out)
# and it will tell you the port number that has been setup for you so
# that you can attach via a separate web browser to any Maverick login 
# node (e.g., login1.maverick.tacc.utexas.edu).
#
# Note: you can fine tune the SLURM submission variables below as
# needed.  Typical items to change are the runtime limit, location of
# the job output, and the allocation project to submit against (it is
# commented out for now, but is required if you have multiple
# allocations).  
#
# To submit the job, issue: "sbatch /share/doc/slurm/job.ipython" 
#
# For more information, please consult the User Guide at: 
#
# http://www.tacc.utexas.edu/user-services/user-guides/maverick-user-guide
#-----------------------------------------------------------------------------
#
#SBATCH -J tvp_ipython                    # Job name
#SBATCH -o ipython.out                # Name of stdout output file (%j expands to jobId)
#SBATCH -p vis                        # Queue name
#SBATCH -N 1                          # Total number of nodes requested (20 cores/node)
#SBATCH -n 20                         # Total number of mpi tasks requested
#SBATCH -t 04:00:00                   # Run time (hh:mm:ss) - 4 hours
#SBATCH -A cs395t_f17

#--------------------------------------------------------------------------
# ---- You normally should not need to edit anything below this point -----
#--------------------------------------------------------------------------

echo job $JOB_ID execution at: `date`

# our node name
NODE_HOSTNAME=`hostname -s`
echo "TACC: running on node $NODE_HOSTNAME"

# set memory limits to 95% of total memory to prevent node death
NODE_MEMORY=`free -k | grep ^Mem: | awk '{ print $2; }'`
NODE_MEMORY_LIMIT=`echo "0.95 * $NODE_MEMORY / 1" | bc`
ulimit -v $NODE_MEMORY_LIMIT -m $NODE_MEMORY_LIMIT
echo "TACC: memory limit set to $NODE_MEMORY_LIMIT kilobytes"

# module purge
# module load TACC intel/15.0.3 python


IPYTHON_BIN=`which ipython`
echo "TACC: using ipython binary $IPYTHON_BIN"

NB_SERVERDIR=$HOME/.ipython/profile_default
IP_CONFIG=$NB_SERVERDIR/ipython_notebook_config.py

# make .ipython dir for logs
mkdir -p $NB_SERVERDIR
# Check whether an iPython password file exists.  If not, complain and exit.
if [ \! -e $IP_CONFIG ] ; then
	echo 
	echo "==========================================================================="
	echo "   You must run 'ipython.password' once before launching an iPython session"
	echo "==========================================================================="
	echo
	exit 1
fi

# Check whether a ipython passwd exists.  If not, complain and exit.
grep "^[^#;]" $IP_CONFIG | grep -q "c.NotebookApp.password"
NOPASS=$?
if [ $NOPASS != 0 ] ; then
       echo 
       echo "==========================================================================="
       echo "   You must run 'ipython.password' once before launching an iPython session"
       echo "==========================================================================="
       echo
       exit 1
fi

# launch ipython
IPYTHON_ARGS="notebook --port 5902 --ip=* --no-browser --logfile=$HOME/.ipython/ipython.$NODE_HOSTNAME.log --config=/home/00832/envision/tacc-tvp/server/scripts/maverick/iPython/ipython.tvp.config.py"
#echo "TACC: using ipython command: $RSTUDIO_BIN $RSTUDIO_ARGS"
nohup $IPYTHON_BIN $IPYTHON_ARGS &> /dev/null && rm $HOME/.ipython.lock &
IPYTHON_PID=$!
echo "$NODE_HOSTNAME $IPYTHON_PID" > $HOME/.ipython.lock

#LOCAL_IPY_PORT=8888
LOCAL_IPY_PORT=5902
IPY_PORT_PREFIX=2
#echo "TACC: local (compute node) Rstudio port is $LOCAL_RS_PORT"
#echo "TACC: remote ipython port prefix is $IPY_PORT_PREFIX"

# the largemem queue has traditional numbering, other queues have row-node numbering
#if [ $SLURM_QUEUE == "largemem" ]; then
#    LOGIN_IPY_PORT="$IPY_PORT_PREFIX`echo $NODE_HOSTNAME | perl -ne 'print $1.$2 if /c\d(\d\d)-\d(\d\d)/;'`"
#else
#    LOGIN_IPY_PORT="$IPY_PORT_PREFIX`echo $NODE_HOSTNAME | perl -ne 'print $1.$2.$3 if /c\d(\d\d)-(\d)\d(\d)/;'`"
#fi

# queues have row-node numbering, put login port in 50K - 51K block to avoid collisions with stampede ports
LOGIN_IPY_PORT="$((49+$IPY_PORT_PREFIX))`echo $NODE_HOSTNAME | perl -ne 'print $1.$2.$3 if /c\d\d(\d)-(\d)\d(\d)/;'`"

echo "TACC: got login node ipython port $LOGIN_IPY_PORT"

# create reverse tunnel port to login nodes.  Make one tunnel for each login so the user can just
# connect to maverick.tacc
for i in `seq 3`; do
    echo ====== ssh -f -g -N -R $LOGIN_IPY_PORT:$NODE_HOSTNAME:$LOCAL_IPY_PORT login$i
    ssh -f -g -N -R $LOGIN_IPY_PORT:$NODE_HOSTNAME:$LOCAL_IPY_PORT login$i
done
echo "TACC: created reverse ports on Maverick logins"

echo "Your ipython notebook server is now running!"
echo "Please point your favorite web browser to https://vis.tacc.utexas.edu:$LOGIN_IPY_PORT"
# info for TACC Visualization Portal
echo "vis.tacc.utexas.edu" > $HOME/.ipython_address
echo "$LOGIN_IPY_PORT" > $HOME/.ipython_port
echo "success" > $HOME/.ipython_status
echo "$SLURM_JOB_ID" > $HOME/.ipython_job_id

# write job start time and duration (in hours) to file
date +%s > $HOME/.ipython_job_start
echo "4" > $HOME/.ipython_job_duration

# Warn the user when their session is about to close
# see if the user set their own runtime
#TACC_RUNTIME=`qstat -j $JOB_ID | grep h_rt | perl -ne 'print $1 if /h_rt=(\d+)/'`  # qstat returns seconds
TACC_RUNTIME=`squeue -l -j $SLURM_JOB_ID | grep $SLURM_QUEUE | awk '{print $7}'` # squeue returns HH:MM:SS
if [ x"$TACC_RUNTIME" == "x" ]; then
	TACC_Q_RUNTIME=`sinfo -p $SLURM_QUEUE | grep -m 1 $SLURM_QUEUE | awk '{print $3}'`
	if [ x"$TACC_Q_RUNTIME" != "x" ]; then
		# pnav: this assumes format hh:dd:ss, will convert to seconds below
		#       if days are specified, this won't work
		TACC_RUNTIME=$TACC_Q_RUNTIME
	fi
fi

if [ x"$TACC_RUNTIME" != "x" ]; then
	# there's a runtime limit, so warn the user when the session will die
	# give 5 minute warning for runtimes > 5 minutes
        H=$((`echo $TACC_RUNTIME | awk -F: '{print $1}'` * 3600))		
        M=$((`echo $TACC_RUNTIME | awk -F: '{print $2}'` * 60))		
        S=`echo $TACC_RUNTIME | awk -F: '{print $3}'`
        TACC_RUNTIME_SEC=$(($H + $M + $S))
        
	if [ $TACC_RUNTIME_SEC -gt 300 ]; then
        	TACC_RUNTIME_SEC=`expr $TACC_RUNTIME_SEC - 300`
        	sleep $TACC_RUNTIME_SEC && wall "$USER's iPython notebook session on $VNC_DISPLAY will end in 5 minutes.  Please save your work now." | wall &
        fi
fi

# spin on .ipython.lock file to keep job alive
while [ -f $HOME/.ipython.lock ]; do
  sleep 30
done


# job is done!

# wait a brief moment so ipython can clean up after itself
sleep 1

echo "TACC: job $SLURM_JOB_ID execution finished at: `date`"
