#!/bin/bash

## When run on your local machine, this script starts
## a apptainer container on ozstar running jupyter, creates an 
## ssh tunnel, and starts a browser to a jupyter notebook.

OTHER_ARGUMENTS=()
oz=nt.swin.edu.au

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"/ozjup

STATUS_FILE=$HOME/.jup_status
TOKEN_FILE=$HOME/.jup_token
JOB_FILE=$HOME/.jup_jobid
IMAGE_NAME=ozjup.sif
# SIF_FILE=$HOME/singularity/tensorflow2-gpu-py3-r-jupyter_latest.sif
SIF_HOME=/fred/oz012/$IMAGE_NAME
# SIF_URL=https://swift.rc.nectar.org.au/v1/AUTH_3a1bdd754ad742cba1ae637bd17e0a6a/ozjup/$IMAGE_NAME
SIF_URL=https://object-store.rc.nectar.org.au/v1/AUTH_387249ff1d9c47c7beb56c12c712f9d4/ozjup/$IMAGE_NAME
IMAGENAME=library://ajgreen/default/tensorflow2-gpu-py3-r-jupyter:latest
RUN_ON_NODE=
WORK_DIR="$WORK_DIR"
USERNAME="$OZSTAR_USERNAME"
PORT=9999
DPORT=9999
MEM=10
HOURS=4
GPU=yes
TOKEN_CHECK=

printhelp() {
    echo "Usage: Run on local machine; default action is to start jupyter container on a head node. 
    ./ozjup [-u username] [-n] [-m memory] [-h hours] command

Where <command> is one of:
start: Start notebook and connect.
status: Check if running and if so where
kill: Stop running instance
token: Run jupyter notebook list to view the connection token
tunnel: Start ssh tunnel to existing session
help: Print this help and exit

With optional flags:
-u, --user: OzSTAR username (If left out, will work it out automatically, can be set with OZSTAR_USERNAME environment variable.)
-w, --workdir: The notebook root, i.e. the root directory where data/  notebooks are stored; a path on OzSTAR. Default = home dir.
-n, --node: Run on a dedicated node using a slurm job
-m, --memory: Memory required in GB (with -n only)
-h, --hours: Hours in walltime to request (with -n only)
-ng, --no-gpu: Don't ask for a GPU (with -n only)
-i, --image: Use a custom image (e.g. ozjup_for_me.sif)

## Examples
./ozjup -u username start # run jupyter on head node and start SSH tunnel.
./ozjup -u username -n -m 10 -h 2 start # run jupyter on job node with 10G for 2 hours, wait for job to start, then start SSH tunnel.
./ozjup check # check if it's running
./ozjup kill # kill container if running
"
}

waitjob() {
    echo Waiting for job to start...
    while [[ ! -e $STATUS_FILE ]]; do
        sleep 1
    done
    echo "Job started - firing up notebook now."
    sleep 5
}

killjob() {
    if [[ ! -e "$STATUS_FILE" ]]; then
        echo Nothing to kill.
    # elif [[ -e $JOB_FILE ]]; then
        # jobid=$(cat $JOB_FILE)
        # scancel $jobid
        # echo "Job $jobid Killed."
        # rm $JOB_FILE
        # rm $STATUS_FILE
    else
        jobhost=$(cat $STATUS_FILE)
        if [[ "$(hostname)" != "$jobhost" ]]; then
            ssh ${jobhost} -q -t "exec bash -l ozjup kill"
        else
            running=isrunning
            if [[ "0" != running ]]; then
                apptainer instance stop ozjup
            fi
            rm $STATUS_FILE
        fi
    fi
}

isrunning() {
    running=$(apptainer instance list|grep ozjup)
    if [[ -n "$running" ]]; then
        echo 1
    else
        echo 0
    fi
}


isrunningold() {
    if [[ -n $(ps ux | grep 'apptainer runtime') ]]; then
        ps ux | grep 'apptainer runtime' | awk '{print $2}'
    else
        echo 0
    fi
}

runjup() {
    checksif
    workdir=$1
    # apptainer instance start --nv -B /apps/skylake/software/CUDA/10.1.243/lib64:/usr/local/cuda/lib64 -B "$workdir":/work $SIF_FILE ozjup
    apptainer instance start --nv -B "$workdir":/work $SIF_FILE ozjup
    echo $(hostname) > $STATUS_FILE
}

checkstatus() {
    localhost=$1
    if [[ -n "$1" ]]; then
        status=$(cat $STATUS_FILE)
    else
        # status=$(ssh $ozu "cat $STATUS_FILE 2>/dev/null")
        status=$(ssh $ozu "cat .jup_status 2>/dev/null")
    fi
    if [[ -n "$status" ]]; then
        echo $status
    fi
}

gettoken() {
    tokcommand="apptainer exec instance://ozjup jupyter notebook list"
    jobhost=$(cat $STATUS_FILE)
    if [[ "$(hostname)" != "$jobhost" ]]; then
        result=$(ssh ${jobhost} -q -t "exec bash -l ozjup token")
    else
        result=$( $tokcommand | grep http | sed -E 's#http://[^:]+:#http://localhost:#')
    fi
    echo $result
}


sshtunnel() {
    NODENAME=$1

    echo "Starting SSH tunnel to $NODENAME..."
    # set -m
    # echo "Now connect to http://localhost:$PORT"
    token=$(ssh ${ozu} "exec bash -l ozjup token")
    echo $token
    # echo COMMAND: ssh -o "StrictHostKeyChecking no" -N -L  $PORT:localhost:$DPORT -J ${ozu} ${USERNAME}${NODENAME} 
    ssh -o "StrictHostKeyChecking no" -N -L  $PORT:localhost:$DPORT -J ${ozu} ${USERNAME}${NODENAME} 
    # echo $OPENCMD http://localhost:${PORT}/ &
    # fg
}

checksif() {
    if [ ! -e $SIF_FILE ]
    then
        echo "Fetching image: this will happen only once"
        mkdir -p $HOME/apptainer
        cd $HOME/apptainer
        if [[ -e "$SIF_HOME" ]]; then
            ln -s $SIF_HOME $IMAGE_NAME
        else
            wget --quiet $SIF_URL
        fi
        # apptainer pull $IMAGENAME
    # else
        # echo "Image present."
    fi
}

makebatch() {
    MEM=$1
    HOURS=$2
    GPU=$3
    gpuline=""
    if [[ -n "$GPU" ]]; then
        gpuline="#SBATCH --gres=gpu"
    fi
    echo "#!/bin/bash
#SBATCH --job-name=ozjup
#SBATCH --output=$HOME/ozjup.log
#SBATCH --cpus-per-task=1
#SBATCH --time=${HOURS}:00:00
#SBATCH --mem=${MEM}G
#SBATCH --tmp=24G
$gpuline

module load apptainer
cd $HOME

apptainer instance start --nv -B "$WORK_DIR":/work $SIF_FILE ozjup
"'echo $(hostname) > '"$STATUS_FILE

jobup=1"'
while [[ "1" == "$jobup" ]]; do
    sleep 10
    running=$(apptainer instance list|grep ozjup)
    if [[ -z "$running" ]]; then
        jobup=0
    fi
done'"
rm $STATUS_FILE

" > $HOME/runjup.sh 
    jobid=$(sbatch $HOME/runjup.sh|awk '{print $4'})
    echo $jobid > $JOB_FILE
}

allargs=$@
for arg in "$@"
do
    case $arg in
        help)
        printhelp
        exit
        shift
        ;;
        start)
        START_JOB=1
        shift
        ;;
        tunnel)
        TUNNEL_ONLY=1
        shift
        ;;
        kill)
        KILL_JOB=1
        shift
        ;;
        status)
        CHECK_JOB=1
        shift
        ;;
        check)
        echo "Deprecated: Use ozjup status instead."
        CHECK_JOB=1
        shift
        ;;
        -n|--node)
        RUN_ON_NODE=1
        shift 
        ;;
        token)
        TOKEN_CHECK=1
        shift
        ;;
        -ng|--no-gpu)
        GPU=
        shift
        ;;
        -c=*|--cache=*)
        CACHE_DIRECTORY="${arg#*=}"
        shift # Remove --cache= from processing
        ;;
        -u|--user)
        USERNAME="$2"
        shift
        # shift
        ;;
        -t|--tmpdir)
        apptainer_TMPDIR="$2"
        shift
        ;;
        -w|--workdir)
        WORK_DIR="$2"
        shift
        # shift
        ;;
        -i|--image)
        IMAGE_NAME="$2"
        shift
        ;;
        -m|--memory)
        MEM="$2"
        shift
        # shift
        ;;
        -h|--hours)
        HOURS="$2"
        shift
        # shift
        ;;
        *)
        OTHER_ARGUMENTS+=("$1")
        shift # Remove generic argument from processing
        ;;
    esac
done
# echo "Running script with these args:" $allargs

environment=home

if [ "1" -eq $(grep -i "Rocky Linux" /etc/redhat-release 2>/dev/null |wc -l) ]
then
    if [ -n "$(hostname | grep tooarrana)" ]
    then
        environment=head
    else
        environment=node
    fi
    USERNAME=$USER
else
    if [[ -z "$USERNAME" ]]; then
        USERNAME=$(ssh $oz 'echo $USER')
    fi

    if [[ -n "$USERNAME" ]]; then
        USERNAME="$USERNAME"@
    fi
fi

ozu=${USERNAME}$oz


# echo XX Running on $(hostname) as user $USERNAME Node=$RUN_ON_NODE env=$environment workdir=${WORK_DIR}

# if [[ ! -e "$IMAGE_NAME" ]]; then
    # SIF_FILE="$IMAGE_NAME"
# else
    SIF_FILE="$HOME/apptainer/$IMAGE_NAME"
# fi

if [ "$environment" = "home" ]
then
    if [[ -z "${TOKEN_CHECK}${START_JOB}${KILL_JOB}${CHECK_JOB}${TUNNEL_ONLY}" ]]; then
        printhelp
        exit
    fi
    # scp -q $(realpath $0) ${ozu}:.
    scp -q $SCRIPTPATH ${ozu}:.
    if [[ -n "$CHECK_JOB" ]]; then
        status=$(checkstatus)
        if [[ -n "$status" ]]; then
            echo "Running on" $status
        else
            echo "Not running."
        fi
        exit
    fi
    if [[ -n "$TUNNEL_ONLY" ]]; then
        status=$(checkstatus)
        if [[ -n "$status" ]]; then
            sshtunnel $status
        else
            echo "No notebook running."
        fi
        exit
    fi

    # Re-execute the script, but on ozstar.
    ssh ${ozu} -q -t "exec bash -l ozjup $allargs"

    if [[ -n "$KILL_JOB$TOKEN_CHECK$TUNNEL_ONLY" ]]; then
        exit
    fi

    sleep 2
    status=$(checkstatus)
    if [[ -n "$status" ]];
    then
        sshtunnel $status
        # echo "Now connect to http://localhost:$PORT"
    else
        echo 
    fi
fi

if [ "$environment" = "head" ]
then
    module load apptainer
    if [[ -n "$KILL_JOB" ]]; 
    then
        killjob 
        exit
    fi
    if [[ -n "$CHECK_JOB" ]]; 
    then
        checkstatus
        exit
    fi
    if [[ -n "$TOKEN_CHECK" ]]; 
    then
        gettoken 
        exit
    fi

    if [[ -z "$WORK_DIR" ]]; then
        WORK_DIR="$HOME"
    fi 

    # OK, we're running it
    rm -f $STATUS_FILE
    rm -f $JOB_FILE

    if [ -z "$RUN_ON_NODE" ]
    then
        runjup $WORK_DIR
    else
        makebatch $MEM $HOURS $GPU
        # jobid=$(makebatch $MEM $HOURS $GPU)
        waitjob
        # waitjob $jobid
        # reportback
    fi
fi

if [ "$environment" = "node" ]
then
    module load apptainer
    if [[ -n "$KILL_JOB" ]]; 
    then
        killjob 
        exit
    fi
    if [[ -n "$CHECK_JOB" ]]; 
    then
        checkstatus
        exit
    fi
    if [[ -n "$TOKEN_CHECK" ]]; 
    then
        gettoken 
        exit
    fi
fi
