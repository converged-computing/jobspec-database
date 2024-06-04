#!/bin/bash

action=${1:-bao-login}


shift
args=$@

me=$(readlink -f $0)


function bao-help() {
    echo "help for $me"
    < bao.sh awk 'BEGIN {fstart=0} /^function bao/ {fstart=NR; print $2} fstart>0 { if ( $0 ~ /.+=\$\{[1-9].*/ ) {print $0} else {fstart=0} }'
}

function bao-self-test() { 
    # perform self-test
    echo TODO
}

function bao-get-config() {
    path=${1:?}
    v=$(< $HOME/.local/etc/bao/bao.yaml yq -r $path)
    eval echo $v
}


function bao-store-fact() {
    fact=${1:?}

    fn=$HOME/.local/state/$BAOBAB_LOGIN_NODE/$fact

    mkdir -pv $(dirname $fn)

    date > $fn
}

function bao-ask-fact() {
    fact=${1:?}
    
    echo "asking fact $fact" >&2

    fn=$HOME/.local/state/$BAOBAB_LOGIN_NODE/$fact


    [ -s $fn ]
}



function bao-mount() {
    mountpoint=$(bao-get-config .sshfs_mountpoint)
    echo "mounting as $mountpoint"
    fusermount -u $mountpoint || echo 'can not unmount'
    mkdir -pv $mountpoint

    if  [ ${bao_ssh_mode:-key} == "sshpass" ]; then
        sshfs ${BAOBAB_LOGIN_NODE}:/ $mountpoint -o ssh_command='sshpass -f '<(keyring get unige $(whoami))' ssh';
    else
        sshfs ${BAOBAB_LOGIN_NODE}:/ $mountpoint;
    fi
}

function bao-login() {
    if  [ ${bao_ssh_mode:-key} == "sshpass" ]; then
        sshpass -f <(keyring get unige $(whoami)) ssh -Y ${BAOBAB_LOGIN_NODE} $@;
    else
        set -x
        ssh -Y ${BAOBAB_LOGIN_NODE} $@;
    fi
}


function bao-upload-dir() {
    local_dir=${1:?}
    remote_dir=${2:?}

    bao mkdir -pv $remote_dir
    (cd $local_dir; ls -ltor; tar cvf - * | bao "tar xvf - -C $remote_dir")
}

function bao-download-dir() {
    remote_dir=${1:?}
    local_dir=${2:?}

    bao tar cvzf - $remote_dir | tar xvzf - -C $local_dir --strip-components=4
}

function bao-update-env() { 

    echo "calling ${FUNCNAME[0]} $1" >&2

    bao_env_version=$(find $HOME/.local/share/bao -type f -exec md5sum {} \; | md5sum | cut -c1-8)

    fact=remote-env-installed/$bao_env_version

    echo "fact: $fact" >&2

    if bao-ask-fact $fact; then
        echo "bao remotely available $fact" >&2
    else
        echo "bao not remotely available $fact, installing" >&2
        bao mkdir -pv env
        bao-upload-dir $HOME/.local/share/bao env
        bao 'mv env/ssh-config ~/.ssh/config'
        bao-store-fact $fact
    fi
    echo "done ${FUNCNAME[0]} $1" >&2
}

function bao-upload-image() {
    bao-upload-image.sh
}

function bao-sync-data-rev() {
    bao bash env/sync.sh sync-rev ${1:?} ${2:-cons}
}

function bao-sync-cat() {
    bao bash env/sync.sh sync-cat
}

function bao-sync-data-revs() {
    bao bash env/sync.sh sync-revs ${1:?from} ${2:?to} ${3:-cons}
}


function bao-sync-ic() {
    bao bash env/sync.sh sync-ic 
}

function bao-sync-idx() {
    bao bash env/sync.sh sync-idx
}


function bao-pull() {
    pattern=${1:?}

    bao-download-dir scratch/data/reduced/ddcache/$pattern  $(bao-get-config .local_ddcache)
}

function bao-list-functions() {
    ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

    < $ABSOLUTE_PATH awk -F'[ (]' '/function bao-/ {print $2}'
}

function bao-upload-workflow() {
    echo "calling ${FUNCNAME[0]} $1" >&2

    workflow_dir=${1:?}

    bao-update-env

    workflow_dir=$(realpath $workflow_dir)

    workflow_version=$(find $workflow_dir -type f -exec md5sum {} \; | md5sum | cut -c1-8)

    workflow_remote_path=workflows/$(basename $workflow_dir)/$workflow_version

    echo "workflow_version $workflow_version" >&2
    echo "workflow_remote_path $workflow_remote_path" >&2

    fact=uploaded/$workflow_remote_path

    if bao-ask-fact $fact; then
        echo "already uploaded" >&2
    else
        echo "workflow $workflow_dir version/hash $workflow_version to $workflow_remote_path" >&2

        bao-upload-dir $workflow_dir $workflow_remote_path
        echo $workflow_version | bao "cat -> $workflow_remote_path/workflow-version"
        bao-store-fact $fact
    fi
        
    echo "$ workflow_remote_path: $workflow_remote_path" 


    #for t0 in $(cat utc-gbm.json | shuf -n 10); do t0=$t0 bash submit.sh ; done
}

function bao-workflow-remote-path() {
    echo "calling ${FUNCNAME[0]} $1" >&2
    echo '$HOME/'$(bao-upload-workflow $workflow_dir | awk '/^\$/ {print substr($0,2,length($0))}' | yq -r .workflow_remote_path)
    echo "done ${FUNCNAME[0]} $1" >&2
}

function bao-workflow-prep-entrypoint() {
    workflow_dir=${1:?}

    echo "calling ${FUNCNAME[0]} $1"

    workflow_remote_path=$(bao-workflow-remote-path)


    echo "
        #!/bin/bash
        #SBATCH --ntasks 1
        #SBATCH --time=${batch_time:-02:00:00}

        source \$HOME/env/init.sh

        export ODA_WORKFLOW_PATH=$workflow_remote_path
        export ODA_WORKFLOW_VERSION=\$(cat $workflow_remote_path/workflow-version)
        export ODA_WORKFLOW_OUTPUT_PATH=\$DDCACHE_ROOT/\$(bash $workflow_remote_path/oda-cache.sh)
        export ODA_JOB_DIR=$workflow_remote_path

        workflow run bash $workflow_remote_path/entrypoint.sh
    " | awk 'NR>1' | cut -c9-1000 | bao "cat -> $workflow_remote_path/auto-entrypoint.sh"

    echo "--> $workflow_remote_path/auto-entrypoint.sh"
    bao cat $workflow_remote_path/auto-entrypoint.sh
}

function bao-run() {
    bao-run-workflow $@
}

function bao-run-workflow() {
    workflow_dir=${1:?}
    setenv=${2:-}

    workflow_remote_path=$(bao-workflow-remote-path)
    bao-workflow-prep-entrypoint $workflow_dir

    setenv=${setenv//,/ }

    echo "setenv: $setenv"
    
    bao "set -x; cat $workflow_remote_path/auto-entrypoint.sh; echo $setenv; cd $workflow_remote_path; export $setenv TMPDIR=\$HOME/scratch/tmp-run ; bash auto-entrypoint.sh" 
}

function bao-submit-array() {
    workflow_dir=${1:?workflow dir}
    argvar=${2:?argvar}
    arglist=${3:?arglist}

    echo "submitting to ${partition:=mono-shared-EL7}"

    echo "will find remote path..."
    workflow_remote_path=$(bao-workflow-remote-path)
    bao-workflow-prep-entrypoint $workflow_dir

    taskid=${taskid:-$(date +%s)}
    taskdir=$workflow_remote_path/arrays/$taskid

    bao mkdir -pv $taskdir/logs

    < $arglist awk '{print "export '$argvar'="$1}' | bao "cat -> $taskdir/envlist.sh"

    
    echo "
        export ODA_JOB_DIR=$workflow_remote_path
        cd $taskdir
        i=0
        while IFS="" read -r p;  do (
            echo \$p
            sp=\$(echo \$p | awk '{print \$2}' | sed 's/[^a-zA-Z0-9 -]/_/g')
            dgst=\$(echo \$p | md5sum | cut -c 1-8)
            echo "job \$sp, \$dgst"

            eval \$p

            export logfile=$taskdir/logs/\${dgst}-\${sp}

            echo -n 'submitted at ' > \$logfile
            date >>  \$logfile

            sbatch --profile=all --mem-per-cpu 4000  --partition ${partition} --export=ALL --output \$logfile $workflow_remote_path/auto-entrypoint.sh
        ); let 'i++'; done < envlist.sh 
    " | bao "cat -> $taskdir/submit-array.sh"

    bao cat $taskdir/submit-array.sh
    bao cat $taskdir/envlist.sh

    bao bash $taskdir/submit-array.sh
}


function bao-upload-images() {
    pattern=${1:?}

    bao mkdir -pv scratch/singularity

    for fn in /dev/shm/singularity/$pattern; do
        cat $fn | pv  | bao "cat - > scratch/singularity/$(basename $fn)"
    done
}

function bao-tail-last-job() {
    n=${1:-1000}
    bao 'fn=$(ls -tr workflows/*/*/*/*/*/* | tail -1); echo "last job log:"; ls -ltor $fn; tail -n '${n}' -f $fn '
}

function bao-less-last-job() {
    bao 'cat  $(ls -tr workflows/*/*/*/*/*/* | tail -1) ' | less -R
}

function bao-logs() {
    patt=${1:-\\*}
    cmin=${2:--60}

    logs=$(bao 'find workflows -type f -wholename \*logs\* -wholename '"${patt}"' -cmin '${cmin})
        
    nlogs=$(echo $logs | wc -w)
    echo "found $nlogs logs"

    if [ $nlogs == 1 ]; then
        bao "cat $logs" 
    else
        echo "found many logs, please choose:"
        bao "ls -ltor $logs"
    fi
    
}

function bao-squeue() {
    bao squeue -u $(whoami)
}


function bao-upload-token() {
    bao-gentoken -o jwt 
    cat jwt | bao 'cat -> .dataapi-jwt' 
    rm -fv jwt
}

function bao-upload-dda-token() {
    cat $HOME/.dda-token | bao 'chmod u+w .dda-token;umask 077; cat -> .dda-token' 
    rm -fv jwt
}

# exceptions

function bao-find-exceptions() {
    echo
}

function bao-sync-ic-version() {
    version=${1:?}
    eval $(bao cat \$HOME/env/init.sh | grep DATA_ROOT=)
    bao rsync -avuL login01.astro.unige.ch:/unsaved/astro/savchenk/osa11/ic-collection/$version/ $DATA_ROOT/ic-collection/$version/
}

function bao-sync-resources() {
    eval $(bao cat \$HOME/env/init.sh | grep DATA_ROOT=)
    bao rsync -avu login01.astro.unige.ch:/unsaved/astro/savchenk/data/resources/ $DATA_ROOT/resources/
}

function bao-sacct() {
    since=${1:-$(date +%Y-%m-%d)}
    bao sacct --starttime 2020-03-06 --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,elapsedraw,timelimitraw,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

    # summarize quality!
}

function bao-inspect-archives() {
    bao 'source env/init.sh; echo "REP_BASE_PROD: $REP_BASE_PROD"; echo "found revolutions in REP_BASE_PROD: "$(ls  $REP_BASE_PROD/scw | wc -l)'
}

export BAOBAB_LOGIN_NODE=${BAOBAB_LOGIN_NODE:-baobab2.hpc.unige.ch}


if echo $action | grep ^bao-; then
    $action $args
    #${action//bao-/} $args
else
    bao-login $action $args
fi
