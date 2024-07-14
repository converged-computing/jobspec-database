#!/bin/bash


# You've created a population average in *population_avergae_tutorial*
# It's now we posssible to spatially normalise the mutant and wild type data to into the same space as this population average.
# We will however use one that's previously been made, as we already have a mask and label for it.

#### Some preliminary stuff to get it working ####
# Get the current working directory of this script
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# printf this before each command to get indentation and dollar prompt
c="\n\t$ "

# Print out the commands except echos and reads
trap '! [[ "$BASH_COMMAND" =~ ^(echo|read|printf) ]] && \
cmd=`eval echo "$BASH_COMMAND" 2>/dev/null` && echo "$cmd"' DEBUG



#### Start the walk-through ####
printf $c
cd "$parent_path"/data/wild_type_and_mutant_data

# Run the job runner script.

# First thing is to generate jobs lists. These keep track of which specimens are being processed so that the
# jobs can be farmed off to other various machines.
printf "\n"
read -p  "First we create CSV files that can be used to used to schedule registration across different computers. Press enter to start:"
printf $c
lama_job_runner -c generate_data.toml -r baseline --make_job_file && lama_job_runner -c generate_data.toml -r mutants --make_job_file
printf "\n"

read -p "Press enter to spatially normalise the wild type control and mutant specimens. To speed this up, the same command could be run on different machines that have access to the data:"
printf $c
lama_job_runner -c generate_data.toml -r baseline && lama_job_runner -c generate_data.toml -r mutants

printf "\n"
status=$?
[ $status -eq 0 ] && echo "All the data is now generated. Now go to step 3 to run the stats pipeline." || echo "The walkthrough failed. Please get in touch bit@har.mrc.ac.uk"






