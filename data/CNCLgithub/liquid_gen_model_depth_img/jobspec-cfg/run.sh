#!/bin/bash

cd "$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

. load_config.sh


# Define the path to the container and conda env
CONT="${ENV['cont']}"

# Parse the incoming command
COMMAND="$@"

# Enter the container and run the command
SING="${ENV['exec']} exec"
mounts=(${ENV['mounts']})
BS=""
for i in "${mounts[@]}";do
    if [[ $i ]]; then
        BS="${BS} -B $i:$i"
    fi
done

# add the repo path to "/project"
BS="${BS} -B ${PWD}:/project"

printf "=%.0s"  $(seq 1 79)
printf "\nExecuting: %s\n" "${COMMAND}"
printf "=%.0s"  $(seq 1 79)
printf "\n"
${SING} ${BS} ${CONT} bash -c "conda init bash \
	&& source activate $PWD/${ENV['env']} \
        && cd $PWD \
        && exec $COMMAND \
        && cd /project \
        && deactivate"
