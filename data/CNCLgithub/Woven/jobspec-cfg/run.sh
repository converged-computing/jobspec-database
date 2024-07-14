#!/bin/bash

############################################################################
# @ Filename      : run.sh
# @ Description   : 
# @ Arguments     : 
# @ Date          : 
############################################################################

cd "$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

. load_config.sh

CONT="${ENV['cont']}"
COMMAND="$@"
SING="${ENV['exec']} exec --nv"
mounts=(${ENV['mounts']})
BS=""
for i in "${mounts[@]}";do
    if [[ $i ]]; then
        BS="${BS} -B $i:$i"
    fi
done

BS=""

printf "=%.0s"  $(seq 1 79)
printf "\nExecuting: %s\n" "${COMMAND}"
printf "=%.0s"  $(seq 1 79)
printf "\n"

if [ ! -d "out" ]; then
    mkdir out
fi

${SING} ${BS} ${CONT} bash -c "source ${ENV['env']}/bin/activate \
                                && exec $COMMAND \
                                && deactivate"