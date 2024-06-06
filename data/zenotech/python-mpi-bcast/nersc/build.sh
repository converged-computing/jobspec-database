#! /bin/bash

#SBATCH -N 1
#SBATCH -p debug
#SBATCH -t 00:20:00
#SBATCH -o build.log

# kill children as we die
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

source ../activate.sh 

PREFIX=${NERSC_HOST}
mkdir -p $PREFIX

function rotate {
    if [ -f $1 ]; then
        local postfix=`stat -c %y $1 | cut -d ' ' --output-delimiter='_' -f1,2`
        mv $1 $1-$postfix
    fi
    mkdir -p `dirname $1`
    mv $2 $1
    echo "Created file at $1"
}

function build {
    if module swap python python/$1 2>&1 | grep ERROR ; then 
        return 1
    fi
    # the first call won't set the env because it is in a child process.
    module swap python python/$1

    local python=`which python`
    local anaconda=`dirname $python`/..
    anaconda=`readlink -f $anaconda`
    mkdir -p $PREFIX/$1
    echo "Building bundle for Python at $anaconda"

    pushd $PREFIX/$1

    ( bundle-anaconda _python.tar.gz $anaconda
      rotate python.tar.gz _python.tar.gz
    ) &

    ( MPICC=cc bundle-pip _mpi4py.tar.gz mpi4py
    rotate mpi4py.tar.gz _mpi4py.tar.gz
    ) &

    ( bundle-pip _fitsio.tar.gz https://github.com/esheldon/fitsio/archive/v0.9.8rc2.tar.gz
    rotate fitsio.tar.gz _fitsio.tar.gz
    ) &

    wait
    popd
}

function system {
    pushd $PREFIX

    rm -rf lib
    mkdir -p lib

    local filelist
    local EXEC
    if which python-mpi; then
        EXEC=python-mpi
    else
        EXEC='python -m mpi4py'
    fi
    filelist=`srun -n 1 strace $EXEC -c 'from mpi4py import MPI' 2>&1 \
    | grep "= 3$" | grep "\\.so" | sed -s 's;open(";;' | sed -s 's;".*;;' \
    | sort | uniq | grep "\\.so" |grep -v "ld.so.conf"`

    srun -n 1 cp -aL $filelist lib/
    srun -n 1 cp -aL /lib64/* lib/
    echo $filelist > filelist
    tar -czf _system-libraries.tar.gz lib/

    rotate system-libraries.tar.gz _system-libraries.tar.gz
    popd
}

echo "-------------------------------------------------"
echo "Working on host $NERSC_HOST, output to $PREFIX"

echo "First build the system-libraries bundle"

( system )

echo "Second build the python environments; output is messy"

( build 2.7-anaconda ) &
( build 3.4-anaconda ) &
( build 3.5-anaconda ) &

wait

echo "-------------------------------------------------"

