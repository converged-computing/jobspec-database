#!/usr/bin/env bash
# author oleg.shpynov@jetbrains.com

# MOCK for module command
type module &>/dev/null || module() { echo "[mock] module $@"; }

# CHPC (qsub) mock replacement
if which qsub &>/dev/null; then
    # Use function to get rid of command substitution.
    # Command substitution doesn't work well with parallel execution.
    run_parallel()
    {
        # LOAD args to $CMD
        CMD=""
        while read -r line; do CMD+=$line; CMD+=$'\n'; done;
        # Return through global variable here, because we can't use command substitution.
        QSUB_ID=$(qsub <<< "$CMD")
    }

    # Small procedure to wait until all the tasks are finished on the qsub cluster
    # Example of usage: wait_complete $TASKS, where $TASKS is a task ids returned by qsub.
    wait_complete()
    {
        echo "Waiting for tasks..."
        for TASK in $@
        do :
            echo -n "TASK: $TASK"
            # The task id is actually the first numbers in the string
            TASK_ID=$(echo ${TASK} | sed -e "s/\([0-9]*\).*/\1/")
            if [[ ! -z "$TASK_ID" ]]; then
                while qstat ${TASK_ID} &> /dev/null; do
                    echo -n "."
                    sleep 100
                done;
            fi
            echo
        done
        echo "Done. Waiting for tasks"
    }
else
    if [[ -z $WASHU_PARALLELISM ]]; then
        WASHU_PARALLELISM=8
    fi
    >&2 echo "Local tasks WASHU_PARALLELISM=$WASHU_PARALLELISM"

    # Local qsub emulation
    qsub()
    {
        # LOAD args to $CMD
        CMD=""
        while read -r line; do CMD+=$line; CMD+=$'\n'; done;

        # MacOS cannot handle XXXX template with ".sh" suffix, also --suffix
        # option not available in BSD mktemp, so let's do some hack
        QSUB_FILE_PREFIX=$(mktemp "${TMPDIR:-/tmp/}qsub.XXXXXXXXXXXX")
        QSUB_FILE="${QSUB_FILE_PREFIX}.sh"
        rm ${QSUB_FILE_PREFIX}

        echo "#This file was generated as QSUB MOCK" > $QSUB_FILE
        echo 'type module &>/dev/null || module() { echo "[mock] module $@"; }' >> $QSUB_FILE
        echo "$CMD" >> $QSUB_FILE
        LOG=$(echo "$CMD" | grep "#PBS -o" | sed 's/#PBS -o //g')
        >&2 echo "LOCAL running TASK: ${QSUB_FILE} LOG: $LOG"
        # Redirect both stderr and stdout to LOG file, don't use output, since we use [run_parallel]
        bash $QSUB_FILE &> "$LOG" &
    }

    run_parallel()
    {
        # Wait until less then $WASHU_PARALLELISM tasks running
        while [[ $(jobs | wc -l) -ge $WASHU_PARALLELISM ]] ; do sleep 1 ; done

        # LOAD args to $CMD
        CMD=""
        while read -r line; do CMD+=$line; CMD+=$'\n'; done;
        qsub <<< "$CMD"
    }

    wait_complete()
    {
        echo "LOCAL waiting for tasks..."
        wait
        echo "Done. LOCAL waiting for tasks"
    }
fi

# Checks for errors in logs, stops the world
check_logs()
{
    # IGNORE MACS2 ValueError
    # See for details: https://github.com/JetBrains-Research/washu/issues/14
    # Also ignore SPP failure
    ERRORS=$(find . -name "*.log" | xargs grep -i -E "error|exception|No such file or directory" |\
        grep -v -E "ValueError|WARNING" | grep -v -E "Error in runmean")

    if [[ ! -z "$ERRORS" ]]; then
        echo "ERRORS found"
        echo "$ERRORS"
        exit 1
    fi
}

job_tmp_dir() {
    if [[ -z "${PBS_JOBID}" ]]; then
      TMP_DIR=~/tmp/job$$/;
    else
      TMP_DIR="/tmp/$PBS_JOBID/";
    fi
    mkdir -p "${TMP_DIR}"

    echo "${TMP_DIR}"
}

clean_job_tmp_dir() {
    if [[ -z "${PBS_JOBID}" ]]; then
      rm -rf "$(job_tmp_dir)"
    fi
}

# Convert path to absolute path and expand all symlinks
function expand_path() {
    # Default case:
    if [[ $1 == /* ]]; then
        TARGET_FILE=$1
    else
        TARGET_FILE="$(pwd)/$1"
    fi

    # If file or directory exists: expand it
    # expand ".." and "." including trailing case
    # based on https://stackoverflow.com/questions/3915040/bash-fish-command-to-print-absolute-path-to-a-file
    if [[ -d "$1" ]]; then
        # dir
        TARGET_FILE="$(cd "$1"; pwd)"
    elif [[ -f "$1" ]]; then
        # file
        if [[ $1 == */* ]]; then
            TARGET_FILE="$(cd "${1%/*}"; pwd)/${1##*/}"
        fi
    fi

    # resolve symlinks:
    # replacement for `readlink -f path` which isn't available in MacOS
    # http://stackoverflow.com/questions/1055671/how-can-i-get-the-behavior-of-gnus-readlink-f-on-a-mac
    PARENT_DIR=$(dirname ${TARGET_FILE})
    if [[ ! -d ${PARENT_DIR} ]]; then
        echo "${TARGET_FILE}"
    else
        cd "$PARENT_DIR"
        TARGET_FILE=`basename ${TARGET_FILE}`

        # Iterate down a (possible) chain of symlinks
        while [[ -L "$TARGET_FILE" ]]
        do
            TARGET_FILE="$(readlink ${TARGET_FILE})"
            cd "$(dirname ${TARGET_FILE})"
            TARGET_FILE="$(basename ${TARGET_FILE})"
        done

        # Compute the canonicalized name by finding the physical path
        # for the directory we're in and appending the target file.
        PHYS_DIR="$(pwd -P)"
        echo "${PHYS_DIR}/${TARGET_FILE}"
    fi
}


# Computes and returns pileup file for given BAM file
function pileup(){
    if [[ ! $# -eq 1 ]]; then
        echo "Need 1 argument! <bam_file>"
        exit 1
    fi
    BAM=$1
    PILEUP_DIR=$(dirname $(expand_path ${BAM}))/pileup
    if [[ ! -d ${PILEUP_DIR} ]]; then
        >&2 echo "Create pileup dir ${PILEUP_DIR}"
        mkdir -p ${PILEUP_DIR}
    fi
    NAME=$(basename ${BAM/.bam/_pileup.bed})
    RESULT=${PILEUP_DIR}/${NAME}
    if [[ ! -f ${RESULT} ]]; then
        export TMPDIR=$(type job_tmp_dir &>/dev/null && echo "$(job_tmp_dir)" || echo "/tmp")
        mkdir -p "${TMPDIR}"
        PILEUP_TMP=$(mktemp $TMPDIR/pileup.XXXXXX.bed)
        >&2 echo "Calculate ${BAM} pileup file in tmp file: ${PILEUP_TMP}"
        bedtools bamtobed -i ${BAM} > ${PILEUP_TMP}
        # Check that we are the first in async calls, not 100% safe
        if [[ ! -f ${RESULT} ]]; then
            mv ${PILEUP_TMP} ${RESULT}
        else
            >&2 echo "Ignore result, file has been already calculated: ${RESULT}"
        fi
    fi
    echo "${RESULT}"
}