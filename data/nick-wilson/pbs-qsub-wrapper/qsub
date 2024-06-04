#!/bin/bash
if [ x"$DEBUG" == xy ] ; then
set -x
fi

## MAINTENANCE: Inform users why they cannot submit jobs
inform_maintenance=0
if [ $inform_maintenance -eq 1 ] ; then
 echo INFO: queues have been stopped to drain the system for maintenance session. 1>&2
 echo 1>&2
fi

# Location of qsub, override this in pbs modulefile if necessary
QSUB_COMMAND="${QSUB_COMMAND:-/opt/pbs/bin/qsub}"

# Pass through some common options
if [ x"$1" == x--help ] ; then
 exec $QSUB_COMMAND "$@"
elif [ x"$1" == x--version ] ; then
 exec $QSUB_COMMAND "$@"
elif [ x"$1" == x--usage ] ; then
 exec $QSUB_COMMAND "$@"
fi

# some script settings:
#  directory to capture script information
script_store=/home/app/.qsub/log
#  temporary file to capture stdin
###tmpfile="${TMPDIR:-/tmp}/qsub.$LOGNAME.$$"
tmpfile="/tmp/qsub.$LOGNAME.$$"
tmpscript="/tmp/qsub.unix.$LOGNAME.$$"
# error messages
error_msg_place_excl="Use of '-l place=excl' is not allowed, instead please request all cpus in a node"
warning_ppn="Use of '-l nodes=...:ppn=...' is deprecated, please consider using '-l select=...:ncpus=...:mpiprocs=...:opmthreads=...:mem=...' instead."
# Available getopt flags
flags=':a:A:c:C:e:hIj:J:k:l:m:M:N:o:p:P:q:r:S:u:v:VW:Z:'
ncpus=""
vnodes=""
mpiprocs=""
mem_v=""
mem_t=""
model=""
ngpus=""

cleanup_exit () {
 rm -f "$tmpfile" "$tmpscript"
 timeout 10 rm -f /home/app/pbs/.cache/"$LOGNAME".* > /dev/null 2>&1
 exit ${1:-0}
}

# parse command line arguments
interactive=0
job_is_array=0
job_has_project=0
projectid=""
while getopts "$flags" opt; do
  case $opt in
    q)
       queue=$OPTARG
       ;;
    l)
      if [ x`echo $OPTARG | grep ncpus` != x ] ; then ncpus=`echo $OPTARG | sed -e 's/^.*ncpus=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | grep ngpus` != x ] ; then ngpus=`echo $OPTARG | sed -e 's/^.*ngpus=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | grep select` != x ] ; then vnodes=`echo $OPTARG | sed -e 's/^.*select=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | grep mpiprocs` != x ] ; then mpiprocs=`echo $OPTARG | sed -e 's/^.*mpiprocs=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | egrep -- "nodes=.*:ppn=.*:mem="`  != x ] ; then mem_v=`echo $OPTARG | sed -e 's/^.*mem=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | egrep 'select[^,]*mem'` != x ] ; then mem_v=`echo $OPTARG | sed -e 's/^.*mem=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | egrep 'select[^,]*model'` != x ] ; then model=`echo $OPTARG | sed -e 's/^.*model=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | egrep '(^|,)mem'` != x ] ; then mem_t=`echo $OPTARG |  sed -e 's/^.*mem=//' -e 's/[:,].*//'` ; fi ; \
      echo $OPTARG | egrep -- "place *=.*excl" 1>&2 && echo ERROR: $error_msg_place_excl 1>&2 && cleanup_exit 1 ; \
      echo $OPTARG | egrep -- "nodes=.*:ppn=" 1>&2 && echo INFO: $warning_ppn 1>&2 && echo 1>&2
      if [ x`echo $OPTARG | grep 'nodes'` != x ] ; then vnodes=`echo $OPTARG | sed -e 's/^.*nodes=//' -e 's/[:,].*//'` ; fi ; \
      if [ x`echo $OPTARG | egrep 'nodes=.*:ppn='` != x ] ; then ncpus=`echo $OPTARG | sed -e 's/^.*ppn=//' -e 's/[:,].*//'` ; fi ; \
      ;;
    I)
      interactive=1
      ;;
    J)
      job_is_array=1
      ;;
    P)
      job_has_project=1 ; projectid="$OPTARG"
      ;;
    W)
      if [ x"$OPTARG" = x"block=true" ] ; then interactive=1 ; fi
      ;;
  esac
done

# look for name of script file
# use getopt to get script name from command line arguments
script=`getopt -o "$flags" -- "$@" | grep -- " -- " | sed -e "s,^.* -- ',," -e "s,'$,,"`
# If not interactive and no script file on command line, capture commands from stdin
if [ $interactive -eq 0 ] ; then
 stdin=0
 if [ x"$script" = x ] ; then
  stdin=1
  script="$tmpfile"
  cat > "$tmpfile"
  chmod go-rwx "$tmpfile" > /dev/null 2>&1
 fi
 if [ -e "$script" ] ; then
  if [ -f /app/dos2unix/bin/dos2unix ] ; then
   /app/dos2unix/bin/dos2unix < "$script" > "$tmpscript" 2>/dev/null
   chmod go-rwx "$tmpscript" > /dev/null 2>&1
  fi
 fi
 # look for ncpus in job script
 if [ x"$ncpus" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- "^ *#PBS .*nodes=.*:ppn=" $tmpscript > /dev/null ; then
    ncpus=`egrep -- "^ *#PBS .*nodes=.*:ppn=" $tmpscript | tail -n 1 | sed -e 's/^.*ppn=//' -e 's/[:,].*//'`
   fi
   if egrep -- "^ *#PBS .*ncpus" $tmpscript > /dev/null ; then
    ncpus=`egrep -- "^ *#PBS .*ncpus" $tmpscript | tail -n 1 | sed -e 's/^.*ncpus=//' -e 's/[:,].*//'`
   fi
  fi
 fi
 # look for ngpus in job script
 if [ x"$ngpus" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- "^ *#PBS .*ngpus" $tmpscript > /dev/null ; then
    ngpus=`egrep -- "^ *#PBS .*ngpus" $tmpscript | tail -n 1 | sed -e 's/^.*ngpus=//' -e 's/[:,].*//'`
   fi
  fi
 fi
 # look for vnodes in job script
 if [ x"$vnodes" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- "^ *#PBS .*nodes" $tmpscript > /dev/null ; then
    vnodes=`egrep -- "^ *#PBS .*nodes" $tmpscript | tail -n 1 | sed -e 's/^.*nodes=//' -e 's/[:,].*//'`
   fi
   if egrep -- "^ *#PBS .*select" $tmpscript > /dev/null ; then
    vnodes=`egrep -- "^ *#PBS .*select" $tmpscript | tail -n 1 | sed -e 's/^.*select=//' -e 's/[:,].*//'`
   fi
  fi
 fi
 # look for mpiprocs in job script
 if [ x"$mpiprocs" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- "^ *#PBS .*mpiprocs" $tmpscript > /dev/null ; then
    mpiprocs=`egrep -- "^ *#PBS .*mpiprocs" $tmpscript | sed -e 's/^.*mpiprocs=//' -e 's/[:,].*//'`
   fi
  fi
 fi
 # look for memory in job script
 if [ x"$mem_v$mem_t" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- '^ *#PBS .*select[^,]*mem' $tmpscript > /dev/null ; then
    mem_v=`egrep -- '^ *#PBS .*select[^,]*mem *=' $tmpscript | tail -1 |  sed -e 's/^.*mem *=//' -e 's/[:,].*//'`
   fi
   if egrep -- '^ *#PBS .*-l([^,]*,| *)mem *=' $tmpscript > /dev/null ; then
    mem_t=`egrep -- '^ *#PBS .*-l([^,]*,| *)mem *=' $tmpscript  | tail -1 |  sed -e 's/^.*mem *=//' -e 's/[:,].*//'`
   fi
  fi
 fi
 # look for model in job script
 if [ x"$model" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- '^ *#PBS .*select[^,]*model' $tmpscript > /dev/null ; then
    model=`egrep -- '^ *#PBS .*select[^,]*model' $tmpscript | sed -e 's/^.*model *=//' -e 's/[:,].*//'`
   fi
  fi
 fi
 # look for queue in job script
 if [ x"$queue" = x ] ; then
  if [ -e "$tmpscript" ] ; then
   if egrep -- "^ *#PBS .*-q" $tmpscript > /dev/null ; then
    queue=`egrep -- "^ *#PBS .*-q" $tmpscript | sed -e 's,^.*-q,,' -e 's,^ *,,' -e 's/[, ].*//'`
   fi
  fi
 fi
 # look for PBS -l place=excl
 if [ -e "$script" ] ; then
  if egrep -- "^ *#PBS *-l.*place *=.*excl" $tmpscript 1>&2 ; then
    echo ERROR: $error_msg_place_excl 1>&2
    cleanup_exit 1
  fi
 fi
 # warn about -l nodes=X:ppn=Y
 if [ -e "$tmpscript" ] ; then
  if grep -- "^ *#PBS *-l.*nodes *=.*ppn *=" $tmpscript 1>&2 ; then
    echo INFO: $warning_ppn 1>&2
  fi
 fi
fi

# check job script for job array
if [ -e "$tmpscript" ] ; then
 if grep -- "^ *#PBS *-J" $tmpscript > /dev/null 2>&1 ; then job_is_array=1 ; fi
fi

# check project id
if [ $job_has_project -eq 0 ] ; then
 if [ -e "$tmpscript" ] ; then
  if [ $interactive -eq 0 ] ; then
   if egrep -- "^ *#PBS .*-P" "$tmpscript" > /dev/null ; then
    job_has_project=1
    projectid=`egrep -- "^ *#PBS .*-P" $tmpscript | tail -1 | sed -e 's,^.*-P *,,' -e 's, .*,,' -e 's,-.*,,'`
   fi
  fi
 fi
fi

projectarg=""
if [ $job_has_project -eq 0 ] ; then
 if [ x"$PROJECT" != x ] ; then
      job_has_project=1
      projectid="$PROJECT"
      projectarg="-P $PROJECT"
 elif [ x"$project" != x ] ; then
      job_has_project=1
      projectid="$project"
      projectarg="-P $project"
 fi
fi

if [ $job_has_project -eq 0 ] ; then
 warn_projectid=1
 if [ $warn_projectid -eq 1  ] ; then
  echo 1>&2
  echo 'INFO: As you have not specifed whether this job as a personal or project run,' 1>&2
  echo 'INFO: the system will count this as a personal run by default.' 1>&2
  echo 'INFO: ' 1>&2
  echo 'INFO: Please use -P Personal or -P <project_id> to properly account for your job.' 1>&2
  echo 'INFO: ' 1>&2
  echo 'INFO: Alternatively, in your job submission script please add' 1>&2
  echo 'INFO: #PBS -P Personal or #PBS -P <project_id>' 1>&2
  echo 'INFO: submitting job...' 1>&2
  echo 1>&2
 fi
else
 # Job has a project id. Check that it is either empty, Personal or an 8 digit number which is a valid group
 # Disable check which is now done by AMS
 check_projectid=0
 if [ $check_projectid -eq 1 ] ; then
  good_projectid=0
  if [ x"$projectid" == x ] ; then good_projectid=1 ; fi
  if [ x"$projectid" == xresv ] ; then good_projectid=1 ; fi
  if [ x"$projectid" == xPersonal ] ; then good_projectid=1 ; fi
  if [[ x"$projectid" =~ x[1-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9] ]] ; then
   getent group "${projectid}" > /dev/null && good_projectid=1
  fi
  if [ $good_projectid -eq 0 ] ; then
   echo "" 1>&2
   echo "ERROR: you have specified project id: $projectid" 1>&2
   echo "ERROR: " 1>&2
   echo "ERROR: This is not a valid project id" 1>&2
   if [ x"$projectid" == x19999999 ] ; then
    echo "ERROR: " 1>&2
    echo "ERROR: The project id 19999999 is the example id used in the documentation" 1>&2
   fi
   echo "ERROR: " 1>&2
   echo "ERROR: Job has been rejected" 1>&2
   echo "" 1>&2
   exit 1
  fi
 fi
fi

## MAINTENANCE: warn job array users of upcoming maintenance
warn_arrays=0
if [ $warn_arrays -eq 1 ] ; then
 # if job is array then print warning
 if [ $job_is_array -eq 1 ] ; then
  echo 1>&2
  echo INFO: This appears to be a PBS job array. 1>&2
  echo INFO: Please be aware that job arrays are automatically removed from the queue when the PBS scheduler is restarted. 1>&2
  echo INFO: This will occur during the upcoming system maintenance. 1>&2
  echo INFO: If this job has not fully completed by that time then it will be removed from the queue. 1>&2
  echo INFO: submitting job... 1>&2
  echo 1>&2
 fi
fi

# check sanity of ncpu resource
maxgpus=1
maxcpus=24
queue="${queue:-normal}"
if [ "$queue" == largemem ] ; then
 maxcpus=48
elif [ "${queue:0:2}" == fj ] ; then
 maxcpus=40
 maxgpus=8
 echo 1>&2
 echo 'ERROR: This queue has been removed.' 1>&2
 echo 'ERROR: Please use the queues dgx or dgx-dev instead.' 1>&2
 echo 1>&2
elif [ "$queue" == datathon ] ; then
 maxcpus=40
 maxgpus=8
elif [ "${queue:0:3}" == dgx ] ; then
 maxcpus=40
 maxgpus=8
fi
if [ x"$ncpus" != x ] ; then
 if [ $ncpus -gt $maxcpus ] ; then echo ERROR: ncpus set to be $ncpus which is greater than limit for selected node type which is $maxcpus 1>&2 ; cleanup_exit 1 ; fi
fi

# check sanity of ngpus resource
if [[ "$queue" == "normal" || "$queue" == "largemem" ]] ; then
 if [ x"$ngpus" != x ] ; then
  echo ERROR: GPU resource requested in non-GPU queue, please use "-q gpu" 1>&2
  echo ERROR: Job rejected 1>&2
  cleanup_exit 1
 fi
else
 if [ x"$ngpus" != x ] ; then
  if [ $ngpus -gt $maxgpus ] ; then
   echo ERROR: ngpus set to be $ngpus which is greater than limit for selected node type which is $maxgpus 1>&2
   echo ERROR: Perhaps you meant: -l select=${ngpus}:ncpus=${ncpus:-$maxcpus}:ngpus=1 1>&2
   echo ERROR: Job rejected 1>&2
   cleanup_exit 1
  fi
 fi
fi

# user should memory in largemem queue
if [[ "$queue" == "largemem" &&  x"$mem_v$mem_t" == x ]] ; then
 echo ERROR: You have requested the largemem queue but have not specified a memory requirement. 1>&2
 echo ERROR: Job rejected 1>&2
 cleanup_exit 1
fi

# more cores than are available in largemem queue
if [[ "$queue" == "largemem" ]] ; then
 if [[ $ncpus -gt 24 && $vnodes -gt 6 ]] ; then
  echo ERROR: You have requested more largemem nodes than are available. 1>&2
  echo ERROR: Requested: $vnodes 1>&2
  echo ERROR: Available: "3x(24cores+1TB), 1x(48cores+1TB), 4x(48cores+2TB), 1x(48cores+6TB)" 1>&2
  echo ERROR: Job rejected 1>&2
  cleanup_exit 1
 fi
 if [[ $ncpus -le 24 && $vnodes -gt 9 ]] ; then
  echo ERROR: You have requested more largemem nodes than are available. 1>&2
  echo ERROR: Requested: $vnodes 1>&2
  echo ERROR: Available: "3x(24cores+1TB), 1x(48cores+1TB), 4x(48cores+2TB), 1x(48cores+6TB)" 1>&2
  echo ERROR: Job rejected 1>&2
  cleanup_exit 1
 fi
 ((totalncpus=vnodes*ncpus))
  if [[ "$totalncpus" -gt 360 ]] ; then
  echo ERROR: You have requested more largemem cores than are available. 1>&2
  echo ERROR: Requested: $totalncpus 1>&2
  echo ERROR: Available: "3x(24cores+1TB), 1x(48cores+1TB), 4x(48cores+2TB), 1x(48cores+6TB)" 1>&2
  echo ERROR: Job rejected 1>&2
  cleanup_exit 1
 fi
fi

# more than 24 ncpus requires full nodes
if [[ "$queue" == "normal" && x"$vnodes" != x && x"$ncpus" != x ]] ; then
 ((totalncpus=vnodes*ncpus))
 if [[ $totalncpus -gt $maxcpus && $ncpus -ne $maxcpus ]] ; then
   echo "ERROR: You have requested more than $maxcpus cpus but have not requested to use full nodes" 1>&2
   echo "ERROR: If you wish to underpopulate compute nodes then you can use something like:" 1>&2
   echo "ERROR:    -l select=${vnodes}:ncpus=${maxcpus}:mpiprocs=${ncpus}:ompthreads=1" 1>&2
   echo "ERROR: Job rejected" 1>&2
   echo 1>&2
   cleanup_exit 1
 elif [[ $vnodes -gt 1 && $ncpus -ne $maxcpus ]] ; then
   echo "ERROR: You have requested more than 1 node but have not requested to use full nodes" 1>&2
   echo "ERROR: If you wish to underpopulate the compute node then you can use something like:" 1>&2
   echo "ERROR:    -l select=${vnodes}:ncpus=${maxcpus}:mpiprocs=${ncpus}:ompthreads=1" 1>&2
   echo "ERROR: Job rejected" 1>&2
   echo 1>&2
   cleanup_exit 1
 fi
fi

# user-specific block of GPU queue (currently turned off)
if [ 0 -eq 1 ]; then
if [[ "$queue" == "gpu" && x"$LOGNAME" = xXXXXX ]] ; then
  # copy script for accounting
  ofile=blocked.`date +%s`.$LOGNAME
  if [ -e "$script" ] ; then
   cp "$script" "$script_store/$ofile"
   chmod go-rwx "$script_store/$ofile" > /dev/null 2>&1
  else
   echo "$@" > "$script_store/$ofile"
   chmod go-rwx "$script_store/$ofile" > /dev/null 2>&1
  fi
 echo ERROR: submitting non-gpu jobs to gpu queue. 1>&2
 echo ERROR: please refer to ticket NSCC-INCXXXXX 1>&2
 echo ERROR: If this is a GPU-enabled job then please contact the helpdesk to re-enable access. 1>&2
 echo ERROR: Job submission failed. 1>&2
 exit 1
fi
fi

# check sanity of mpiprocs resource, if sharing a node then mpiprocs should be less than ncpus
if [ x"$ncpus" != x ] ; then
 if [ x"$mpiprocs" != x ] ; then
  if [ $mpiprocs -gt $ncpus ] ; then
   if [ $ncpus -lt $maxcpus ] ; then
    echo ERROR: mpiprocs is set to $mpiprocs which is greater than ncpus which is set to $ncpus 1>&2
    echo ERROR: This is only allowed if requesting whole nodes \(ncpus=$maxcpus\) 1>&2
    cleanup_exit 1
   else
    echo INFO: You have requested more MPI processes than physical CPU cores  1>&2
    echo INFO: mpiprocs is set to $mpiprocs which is greater than ncpus which is set to $ncpus 1>&2
    echo INFO: This is not generally recommended for compute-intensive applications  1>&2
    echo INFO: submitting job... 1>&2
    echo 1>&2
   fi
  fi
 fi
fi

# check sanity of vnodes
#echo DEBUG: vnodes=$vnodes

# check memory per vnode
if [ x"$mem_v" != x ] ; then
 if [ x"$model" != xrx4770m1 ] ; then
  mem_v_actual=$mem_v
  mem_v=`echo $mem_v | tr '[A-Z]' '[a-z]'`
  mem_v="${mem_v/tb/t}"
  mem_v="${mem_v/gb/g}"
  mem_v="${mem_v/mb/m}"
  mem_v="${mem_v/kb/k}"
  if [ x"${mem_v/t/}" != "x${mem_v}" ] ; then mem_v=`echo "${mem_v/t/}"\*1099511627776 | bc` ; fi
  if [ x"${mem_v/g/}" != "x${mem_v}" ] ; then mem_v=`echo "${mem_v/g/}"\*1073741824 | bc` ; fi
  if [ x"${mem_v/m/}" != "x${mem_v}" ] ; then mem_v=`echo "${mem_v/m/}"\*1048576 | bc` ; fi
  if [ x"${mem_v/k/}" != "x${mem_v}" ] ; then mem_v=`echo "${mem_v/k/}"\*1024 | bc` ; fi
  if [ x"${mem_v/[a-z]/}" = "x${mem_v}" ] ; then
   if [ x"$queue" = xnormal ] ; then
    # 96GB is 103079215104 but default is 105Gb which is 112742891520
    # if [ $mem_v -gt 103079215104 ] ; then
    if [ $mem_v -gt 112742891520 ] ; then
      echo ERROR: mem=$mem_v_actual 1>&2
      echo ERROR: You have requested more than 96GB of memory in a node in the normal queue 1>&2
      echo ERROR: There are no nodes where this job can run, job rejected 1>&2
      cleanup_exit 1
    elif [ $mem_v -gt 103079215104 ] ; then
     echo INFO: You have requested more than 96GB of memory in a node in the normal queue. 1>&2
     echo INFO: This is more than the value defined by NSCC policy of 4GB per core. 1>&2
     echo INFO: In some circumstances this can cause the node to run out of memory. 1>&2
     echo INFO: In future we highly recommend setting a memory limit of 96GB per standard node. 1>&2
     echo INFO: submitting job... 1>&2
     echo 1>&2
    fi # mem_v
    t_ncpus=${ncpus:-24}
    ((mem_c=mem_v/t_ncpus))
    ((core_req=mem_v/4294967296))
    ((mem_tst=core_req*4294967296))
    if [ $mem_tst -ne $mem_v ] ; then ((core_req=core_req+1)) ; fi
    # 4GB is 4294967296 but 105GB/24 is 4697620480
    # if [ $mem_c -gt 4294967296 ] ; then
    if [ $mem_c -gt 4697620480 ] ; then
     echo 1>&2
     echo INFO: It looks as if you have requested more than 4GB of memory per core in the normal queue. 1>&2
     echo INFO: ncpus=$ncpus mem=$mem_v_actual 1>&2
     echo INFO: This is against policy. 1>&2
     echo INFO: In future please either increase the ncpus value or decrease the mem value. 1>&2
     echo INFO: For example for an MPI application you could use something like: -l select=${vnodes:-1}:ncpus=${core_req}:mpiprocs=${ncpus}:ompthreads=1:mem=${mem_v_actual}  1>&2
     echo INFO: or for a threaded application you could use something like: -l select=${vnodes:-1}:ncpus=${core_req}:ompthreads=${ncpus}:mem=${mem_v_actual}  1>&2
     echo INFO: submitting job... 1>&2
     echo 1>&2
    fi # mem_c
   elif [ x"$queue" = xgpu ] ; then
    # 96GB is 103079215104 but default is 105Gb which is 112742891520
    # if [ $mem_v -gt 103079215104 ] ; then
    if [ $mem_v -gt 112742891520 ] ; then
      echo ERROR: mem=$mem_v_actual 1>&2
      echo ERROR: You have requested more than 96GB of memory in a node in the gpu queue 1>&2
      echo ERROR: There are no nodes where this job can run, job rejected 1>&2
      cleanup_exit 1
    elif [ $mem_v -gt 103079215104 ] ; then
     echo INFO: You have requested more than 96GB of memory in a node in the gpu queue. 1>&2
     echo INFO: This is more than the value defined by NSCC policy of 4GB per core. 1>&2
     echo INFO: In some circumstances this can cause the node to run out of memory. 1>&2
     echo INFO: In future we highly recommend setting a memory limit of 96GB per gpu node. 1>&2
     echo INFO: submitting job... 1>&2
     echo 1>&2
    fi # mem_v
   fi # queue
  fi # mem_v numeric
 fi # model
fi # mem_v



# check total memory
if [ x"$mem_t" != x ] ; then
 if [ x"$model" != xrx4770m1 ] ; then
  mem_t_actual=$mem_t
  mem_t=`echo $mem_t | tr '[A-Z]' '[a-z]'`
  mem_t="${mem_t/tb/t}"
  mem_t="${mem_t/gb/g}"
  mem_t="${mem_t/mb/m}"
  mem_t="${mem_t/kb/k}"
  if [ x"${mem_t/t/}" != "x${mem_t}" ] ; then mem_t=`echo "${mem_t/t/}"\*1099511627776 | bc` ; fi
  if [ x"${mem_t/g/}" != "x${mem_t}" ] ; then mem_t=`echo "${mem_t/g/}"\*1073741824 | bc` ; fi
  if [ x"${mem_t/m/}" != "x${mem_t}" ] ; then mem_t=`echo "${mem_t/m/}"\*1048576 | bc` ; fi
  if [ x"${mem_t/k/}" != "x${mem_t}" ] ; then mem_t=`echo "${mem_t/k/}"\*1024 | bc` ; fi
  if [ x"${mem_t/[a-z]/}" = "x${mem_t}" ] ; then
   if [ x"$queue" = xnormal ] ; then
    # 96GB is 103079215104 but default is 105Gb which is 112742891520
    # if [ $mem_t -gt 103079215104 ] ; then
    t_vnodes=${vnodes:-1}
    ((mem_lim=112742891520*t_vnodes))
    ((mem_warn=103079215104*t_vnodes))
    if [ $mem_t -gt $mem_lim ] ; then
      echo ERROR: mem=$mem_t 1>&2
      echo ERROR: You have requested more than 96GB of memory in a node in the normal queue 1>&2
      echo ERROR: There are no nodes where this job can run, job rejected 1>&2
      cleanup_exit 1
    elif [ $mem_t -gt $mem_warn ] ; then
     echo INFO: You have requested more than 96GB of memory in a node in the normal queue. 1>&2
     echo INFO: This is more than the value defined by NSCC policy of 4GB per core. 1>&2
     echo INFO: In some circumstances this can cause the node to run out of memory. 1>&2
     echo INFO: In future we highly recommend setting a memory limit of 96GB per standard node. 1>&2
     echo INFO: submitting job... 1>&2
     echo 1>&2
    fi # mem_t
    t_vnodes=${vnodes:-1}
    t_ncpus=${ncpus:-24}
    ((mem_c=mem_t/t_vnodes))
    ((mem_c=mem_c/t_ncpus))
    ((core_req=mem_t/4294967296))
    ((mem_tst=core_req*4294967296))
    if [ $mem_tst -ne $mem_t ] ; then ((core_req=core_req+1)) ; fi
    # 4GB is 4294967296 but 105GB/24 is 4697620480
    # if [ $mem_c -gt 4294967296 ] ; then
    if [ $mem_c -gt 4697620480 ] ; then
     echo 1>&2
     echo INFO: It looks as if you have requested more than 4GB of memory per core in the normal queue. 1>&2
     echo INFO: ncpus=$ncpus mem=$mem_t_actual 1>&2
     echo INFO: This is against policy. 1>&2
     echo INFO: In future please either increase the ncpus value or decrease the mem value. 1>&2
     echo INFO: For example for an MPI application you could use something like: -l select=${vnodes:-1}:ncpus=${core_req}:mpiprocs=${ncpus}:ompthreads=1:mem=${mem_t_actual}  1>&2
     echo INFO: or for a threaded application you could use something like: -l select=${vnodes:-1}:ncpus=${core_req}:ompthreads=${ncpus}:mem=${mem_t_actual}  1>&2
     echo INFO: submitting job... 1>&2
     echo 1>&2
    fi # mem_c
   fi # queue
  fi # mem_t numeric
 fi # model
fi # mem_t

# check for faulty stdout or stderr
if [ -e "$tmpscript" ] ; then
 _x=`egrep '^[[:space:]]*#PBS -(o|e) (~|\$)' "$tmpscript"`
 if [ x"$_x" != x ] ; then
  echo 1>&2
  echo 'WARNING: You appear to be using the syntax of either a ~ or $ with #PBS -o or #PBS -e' 1>&2
  echo 'WARNING: Please be aware that PBS does not expand ~ or variables in #PBS directives' 1>&2
  echo 'WARNING: These files will not be copied back correctly from the compute nodes' 1>&2
  echo 'WARNING: Please use either the full or relative path' 1>&2
  echo 'WARNING: submitting job...' 1>&2
  echo 1>&2
 fi
fi

((rc=0))
# run real qsub command
if [ $interactive -eq 1 ] ; then
 # if interactive replace current shell with qsub and don't try to capture job script
 exec $QSUB_COMMAND "$@" $projectarg
elif [ $stdin -eq 0 ] ; then
 # using job script from command line and capture id in jobid variable for logging job script
 jobid=`$QSUB_COMMAND "$@" $projectarg` 
 ((rc=$?))
else
 # get job script from stdin and capture id in jobid variable for logging job script
 jobid=`$QSUB_COMMAND "$@" $projectarg < "$tmpfile"`
 ((rc=$?))
fi
if [ $rc -ne 0 ] ; then echo $jobid ; if [ $stdin -eq 1 ] ; then rm -f "$tmpfile" ; fi ; cleanup_exit $rc ; fi

# print jobid as expected by qsub
echo $jobid

## MAINTENANCE: Put job arrays on hold
hold_arrays=0
if [ $hold_arrays -eq 1 ] ; then
if [ $job_is_array -eq 1 ] ; then
 echo 1>&2
 echo WARNING: This appears to be a PBS job array. 1>&2
 echo WARNING: Job arrays are being put into hold status until after the maintenance period. 1>&2
 echo WARNING: Once the system is available the job arrays will be released. 1>&2
 echo 1>&2
 qhold $jobid 1>&2
fi
fi

if [ x"$jobid" == x ] ; then jobid=noid.`date +%s` ; fi

# copy script for accounting
if [ -e "$script" ] ; then
 cp "$script" "$script_store/$jobid.$LOGNAME" > /dev/null 2>&1
 chmod go-rwx "$script_store/$jobid.$LOGNAME" > /dev/null 2>&1
else
 echo "$@" > "$script_store/$jobid.$LOGNAME" > /dev/null 2>&1
 chmod go-rwx "$script_store/$jobid.$LOGNAME" > /dev/null 2>&1
fi

# delete temporary file (if necessary) and exit
cleanup_exit
