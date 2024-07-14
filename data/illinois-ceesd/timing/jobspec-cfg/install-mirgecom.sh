#!/bin/bash -l

set -e
set -x
set -o pipefail

date

MIRGE_BRANCH="main"
TIMING_ENV_NAME="mirge.timing.env"

if [ ! -z "$1" ]; then
    MIRGE_BRANCH="$1"
fi

if [ ! -z "$2" ]; then
    TIMING_ENV_NAME="$2"
fi

# -- Install conda env, dependencies and MIRGE-Com via *emirge*
# --- remove old run if it exists
if [ -d "emirge" ]
then
    echo "Removing old timing run."
    mv -f emirge emirge.old
    rm -rf emirge.old &
fi
# --- grab emirge and install MIRGE-Com 
git clone https://github.com/illinois-ceesd/emirge.git
cd emirge
./install.sh --branch=${MIRGE_BRANCH} --env-name=${TIMING_ENV_NAME}

