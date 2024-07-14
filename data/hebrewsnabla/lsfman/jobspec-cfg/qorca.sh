#!/usr/bin/env bash

# Modified  2018-09-25
# Author    zyzhu
# 2019-03-09: Set span as nproc by default; change log file generation path
# 2019-03-11: Many changes: change process, machine numbers; default scratch path of multi-nodes; bsub file structure
# 2021-05-26: Reset argument -m to specify node name that job would be submitted
# 2021-07-02: Added ORCA 5.0.0; warning message not printed to log file
#
# 2021-Dec:   Avoid adding nprocs to inp; Add 5.0.2
# 2022-Feb:   Add 5.0.3
# 2023-Mar:   Add 5.0.4
#
# Modified from qg09 script

#---- Basic variables
defver=504

#---- qsub_usage Help Information
qsub_usage()
{
    echo ""
    echo "------"
    echo " qorca.sh"
    echo "  Modified from Zhenyu Zhu's qorca.z Script"
#    echo "  Modified from qg09 script"
    echo "  Use this script to make an ORCA Calculation"
    echo "  Modified by SR Wang for qcusers, Dec 2021"
    echo "------"
    echo ""
    echo "Usage:"
    echo "  -v     Version      [default: ${defver}]"
    echo "  -q     Queue        [default: single]"
    echo "  -p     Processes    [default: 1]"
    echo "  -m     Node to be submitted  [default: None]"
    echo "  -P     Do not submit job if set to 1, i.e., generate .bsub file but not submit that file.  [default: 0]"
    echo "  -c     Extra files to be copied to scratch  [default: Nothing]"
    echo ""
    echo "Example:"
    echo "  qorca.sh -q small -p 4 xxx.inp"
    echo ""
    echo "Attention!"
    echo "  - Unlike qorca.z, number of cores is NOT appended to the input card automatically."
    echo "    You NEED to specify 'PAL*' or '%pal nprocs ** end' in input card"
    echo ""
    echo " By default, submitting xxx.inp will copy xxx.* (like xxx.gbw) to scratch. But if you want to read other files,"
    echo " like yyy.gbw, zzz.uno, you need to specify that by -c"
    echo "  qorca.sh -c yyy.gbw xxx.inp"
    echo "  qorca.sh -c yyy.gbw,zzz.uno xxx.inp"
    echo ""
    echo "Version:"
#    echo "  -v 303    3.0.3 version with openmpi-1.6.5"
#    echo "  -v 401    4.0.1 version with openmpi-2.0.2"
    echo "  -v 411    4.1.1 version with modified openmpi-3.1.3"
    echo "  -v 421    4.2.1 version with modified openmpi-3.1.3"
#    echo "  -v 500    5.0.0 version with openmpi-4.1.1"
    echo "  -v 502    5.0.2 version with openmpi-4.1.1"
    echo "  -v 503    5.0.3 version with openmpi-4.1.1"
    echo "  -v 504    5.0.4 version with openmpi-4.1.1"
    echo ""
    echo "Queue and default number of processes:"
    echo "  -q single   (1 proc)"
    echo "  -q small    (4 proc)"
    echo "  -q Gaussian (28 proc)"
#    echo "  -q xp24mc4  24 threads"
#    echo "  -q xp24mc2  24 threads = xppn"
    echo "  -q xp40mc12 (40 proc)"
    echo "  -q xp48mc8  (48 proc)"
    echo "  -q xp36mc10 (36 proc)"
    echo ""
    echo ""
}

#---- Parse terminal arguments for queue submitting
qsub_parse_args()
{
    while getopts "v:q:p:hm:P:c:" o; do
        case $o in
#** ORCA_VER
#** QUEUE
#** NPROC
#** MACH
            v) ORCA_VER=$OPTARG ;;
            q) QUEUE=$OPTARG ;;
            p) NPROC=$OPTARG ;;
            m) MACH=$OPTARG ;;
            P) PAUSE=$OPTARG ;;
            c) IFS=',' COPYarray=($OPTARG) ;;            
            h) qsub_usage
               exit 0 ;;
            \?) qsub_usage ;;
        esac
    done
    shift $(($OPTIND - 1))

    # check job file
    if [[ $# -lt 1 ]]; then
        qsub_usage
    fi
    if [[ ! -e "$1" ]]; then
        echo "Error: File $1 does not exist!"
        exit 1
    fi
    COPY=""
    for i in "${COPYarray[@]}"; do
      if [[ ! -e $i ]]; then
        echo "Warning: file $i does not exist in -c"
      fi
      COPY+="${i} "
    done
    if [[ ! -z $COPY ]]; then
      echo "copy $COPY to workdir"
    fi
#    if [[ ! -e "$COPY" ]]; then
#        echo "Error: File $COPY does not exist!"
#        exit 1
#    fi
#** ORCA_FILE: name of this current job
#** ORCA_WRKDIR: current work directory
    ORCA_FILE_ARG=$1
    ORCA_FILE=$(basename "$ORCA_FILE_ARG")
    ORCA_WRKDIR=$(dirname $(realpath "$ORCA_FILE_ARG"))
    ORCA_FILE_PREFIX="${ORCA_FILE%.*}"
    if [[ -n "$2" ]]; then
        PBS_FILENAME="$2"
    else
        PBS_FILENAME="${ORCA_FILE%.*}"
    fi

    # pbs script name should be less than 14
    PBS_FILENAME="${PBS_FILENAME:0:10}"
    echo "set pbs job name as: $PBS_FILENAME"

#** BSUB_FILE
    BSUB_FILE="$ORCA_WRKDIR/$PBS_FILENAME.bsub"

#--   Define ORCA Version
#** ORCA_EXEC
#** OPENMPI_PATH
    if [[ -z "$ORCA_VER" ]]; then
        echo "Try to use the default version -- ${defver}"
        ORCA_VER=${defver}
    fi
    if [[ "$ORCA_VER" == "303" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_3_0_3_linux_x86-64/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-1.6.5
    elif [[ "$ORCA_VER" == "401" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_4_0_1_2_linux_x86-64_openmpi202/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-2.0.2
    elif [[ "$ORCA_VER" == "411" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_4_1_1_linux_x86-64_openmpi313/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-3.1.3
    elif [[ "$ORCA_VER" == "421" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_4_2_1_linux_x86-64_openmpi314/orca
        OPENMPI_PATH=/share/apps/OpenMPI/openmpi-3.1.4
    elif [[ "$ORCA_VER" == "500" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_5_0_0_linux_x86-64_shared_openmpi411/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-4.1.1
        ORCA_PATH=/share/apps/ORCA/orca_5_0_0_linux_x86-64_shared_openmpi411
    elif [[ "$ORCA_VER" == "502" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_5_0_2_linux_x86-64_shared_openmpi411/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-4.1.1
        ORCA_PATH=/share/apps/ORCA/orca_5_0_2_linux_x86-64_shared_openmpi411
    elif [[ "$ORCA_VER" == "503" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_5_0_3_linux_x86-64_shared_openmpi411/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-4.1.1
        ORCA_PATH=/share/apps/ORCA/orca_5_0_3_linux_x86-64_shared_openmpi411
    elif [[ "$ORCA_VER" == "504" ]]; then
        ORCA_EXEC=/share/apps/ORCA/orca_5_0_4_linux_x86-64_shared_openmpi411/orca
        OPENMPI_PATH=/share/apps/ORCA/openmpi-4.1.1
        ORCA_PATH=/share/apps/ORCA/orca_5_0_4_linux_x86-64_shared_openmpi411
    else
        echo "No such version"
        qsub_usage
        exit 1
    fi
}

qcheck()
{
    if [[ -z "$QUEUE" ]]; then
        echo "Try to use the default queue -- single"
        QUEUE="single"
    fi
    if [[ -z "$NPROC" ]]; then
        #if [[ $QUEUE == "xppn" || $QUEUE == "xp24mc4" || $QUEUE == "xp24mc2" ]]; then
        #    echo "Try to use the default nproc for xppn -- 24"
        #    NPROC=24
        if [[ $QUEUE == "Gaussian" ]]; then
            echo "Try to use the default nproc for Gaussian -- 28"
            NPROC=28
        elif [[ $QUEUE == "xxchem" || $QUEUE == "xp40mc12" ]]; then
            echo "Try to use the default nproc for xxchem -- 40"
            NPROC=40
        elif [[ $QUEUE == "xp48mc8" ]]; then
            echo "Try to use the default nproc for xp48mc8 -- 48"
            NPROC=48
        elif [[ $QUEUE == "xp36mc10" ]]; then
            echo "Try to use the default nproc for xp36mc10 -- 36"
            NPROC=36
        elif [[ $QUEUE == "small" ]]; then
            echo "Try to use the default nproc for small -- 4"
            NPROC=4
        elif [[ $QUEUE == "single" ]]; then
            echo "Try to use the default nproc for single -- 1"
            NPROC=1
        else
            echo "Try to use the default nproc -- 1"
            NPROC=1
        fi
    fi

#** NPROC
#   Read number or threads from input file
#   parallel ORCA: https://sites.google.com/site/orcainputlibrary/setting-up-orca
#   regex to read PAL: https://unix.stackexchange.com/questions/131785/regex-that-would-grep-numbers-after-specific-string
#   or page 6 of ORCA Manual 4.0.1
#   However, I decided to remove the current lines, adding something to temporary file
#    NPROC=$(grep -e '^!' $ORCA_FILE | grep -oP '(?<=PAL)[0-9]+')
#    if [[ -z $NPROC ]]; then
#        echo "No parallel. 1 thread submitted."
#        NPROC=1
#    fi
#** NPROC end
    NPINP1=$(grep -e '^!' $ORCA_FILE | grep -oP '(?<=PAL)[0-9]+')
    NPINP2=$(grep -i -e 'nproc' $ORCA_FILE | grep -oP '[0-9]+')
    echo "Detected PAL: $NPINP1 Nproc: $NPINP2 in inp file"
    if [ $NPROC -gt 1 ]; then
      if [[ -z $NPINP1  && -z $NPINP2 ]]; then
        echo "Warning: No pal or nproc detected in inp file. This job may run in single process!"
        echo "You can specify"
        echo "    %pal nprocs NPROC end"
        echo "in inp file"
      fi
    fi
    if [[ $NPINP1 -gt $NPROC || $NPINP2 -gt $NPROC ]]; then
        echo "Warning: the pal or nproc number in inp file larger than '-p' value!"
    fi
}

qsubmit()
{
#   Header information
    echo "#BSUB -J $PBS_FILENAME "
    echo "#BSUB -q $QUEUE "
    echo "#BSUB -R span[ptile=$NPROC]"
    echo "#BSUB -n $NPROC"
    echo "#BSUB -o $ORCA_FILE_PREFIX-%J.out"
    if [[ -z "$MACH" ]]; then
        echo ""
    else
        echo "#BSUB -m $MACH"
    fi
#   Job-specific information
    echo ""
    echo "USER=$USER"
    echo "ORCA_EXEC=$ORCA_EXEC"
    echo "PATH=$OPENMPI_PATH/bin:"'$PATH'
    echo "LD_LIBRARY_PATH=$OPENMPI_PATH/lib:"'$LD_LIBRARY_PATH'
    if [[ ! -z "$ORCA_PATH" ]]; then
        echo "LD_LIBRARY_PATH=$ORCA_PATH:"'$LD_LIBRARY_PATH'
    fi
    echo "ORCA_WRKDIR=$ORCA_WRKDIR"
    echo "ORCA_FILE=$ORCA_FILE"
    echo "ORCA_FILE_PREFIX=$ORCA_FILE_PREFIX"
    #if [[  ! -z "$COPY" ]]; then
    #  echo "COPY=$COPY"
    #fi
    echo "NPROC=$NPROC"
    echo 'cd $ORCA_WRKDIR'
#   Make scratch directory
    echo ""
    # if multi-machine, generate temporary files in current directory
#   if [[ "$MACH" == 1 ]]; then
    echo 'mkdir -p /scratch/scr/$USER/orca/'
    echo 'SCR_DIR=$(mktemp -d /scratch/scr/$USER/orca/"$ORCA_FILE"__XXXXXX)'
#   else
#       echo 'mkdir -p /tmp/$USER/orca/'
#       echo 'SCR_DIR=$(mktemp -d /tmp/$USER/orca/"$ORCA_FILE"__XXXXXX)'
#   fi
    echo 'echo \"From BSUB script: Scratch file directory: $SCR_DIR\" > $ORCA_WRKDIR/$ORCA_FILE_PREFIX.oout 2>&1'
    echo 'cp $ORCA_FILE $SCR_DIR'
    echo 'cp $ORCA_FILE_PREFIX.gbw $SCR_DIR'
    echo 'cp $ORCA_FILE_PREFIX.pot $SCR_DIR'
    if [[  ! -z "$COPY" ]]; then
      echo "cp $COPY "'$SCR_DIR'
    fi
#    echo 'cp *.uno *.qro $SCR_DIR'
    echo 'cp *.cmp $SCR_DIR'
#    echo 'cp *.xyz $SCR_DIR'
    echo 'cd $SCR_DIR'
#   Make processes
#    echo ""
#    echo 'echo "" >> $ORCA_FILE'
#    echo 'echo "%pal nprocs $NPP end" >> $ORCA_FILE'
#    echo 'echo "" >> $ORCA_FILE'
#   Cleanup setup
    echo ""
    echo 'cleanup()'
    echo '{'
    echo '    echo Job terminated from outer space! >> $ORCA_WRKDIR/$ORCA_FILE_PREFIX.oout 2>&1'
    echo '    rm *.tmp $ORCA_FILE'
    echo '    mv $SCR_DIR $ORCA_WRKDIR'
    echo '    echo From BSUB script: Move all scratch directory to the job file directory! Note to remove that if you do not need that! >> $ORCA_WRKDIR/$ORCA_FILE_PREFIX.oout 2>&1'
    echo '    exit'
    echo '}'
    echo 'trap cleanup SIGTERM SIGINT SIGHUP SIGKILL SIGSEGV SIGSTOP SIGPIPE'
#   Make calculation
    echo ""
    echo '$ORCA_EXEC $ORCA_FILE >> $ORCA_WRKDIR/$ORCA_FILE_PREFIX.oout'
    echo 'rm *.tmp'
    echo 'mv * $ORCA_WRKDIR' 
    echo 'rmdir $SCR_DIR'
}

qsub_parse_args $*
qcheck
qsubmit > $BSUB_FILE
if [[ "$PAUSE" != 1 ]]; then
    bsub < $BSUB_FILE
fi
