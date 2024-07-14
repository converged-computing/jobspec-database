#!/usr/bin/env bash
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH -N 1
#SBATCH -p short
#SBATCH --mem 32G
#SBATCH -C E5-2695

# Stop execution after any error
set -e

# Cleanup function to be executed upon exit, for any reason
function cleanup() {
    rm -rf $WORKDIR
}



########################################
#
# Useful variables
#
########################################

# Your user name
# (Don't change this)
MYUSER=$(whoami)

# Path of the local storage on a node
# Use this to avoid sending data streams over the network
# (Don't change this)
#LOCALDIR=/local
# To be changed as per experiment (my cluster environment)
MYDIR='/home/aakhter/work/TOData'

# Folder where you want your data to be stored (my cluster environment)
DATADIR=$MYDIR/data
SOLDIR=$MYDIR/sol

# Change as per experiment
THISJOB=$MYDIR

########################################
#
# Job-related variables
#
########################################

# Job working directory
# (Don't change this)
WORKDIR=$LOCALDIR/$MYUSER/$THISJOB



########################################
#
# Job directory
#
########################################

# Create work dir from scratch, enter it
# (Don't change this)
#rm -rf $WORKDIR && mkdir -p $WORKDIR && cd $WORKDIR

# Make sure you cleanup upon exit
# (Don't change this)
trap cleanup EXIT SIGINT SIGTERM



########################################
#
# Actual job logic
#
########################################

# Execute job
# Commands
#module load gurobi # Comment this if running heuristic
#python3 -m pip install plotly --user
#mkdir C-mdvrp # comment this if running random input
#cp $MYDIR/C-mdvrp/* $WORKDIR/C-mdvrp/  # comment this if running random input
for FUEL in 50 75 100 125 150
do
    for T_MAX in 50 75 150 300 450 600
    do
        for ITER in {0..9}
        do
            for SOLVER_TYPE in F1 F2 F3 F4 F5 F6 F7 F8
            do
                INSTANCE_STRING=$1
                ITERATION_STRING="${INSTANCE_STRING}F${FUEL}Tmax${T_MAX}Iter${ITER}"
                echo $ITERATION_STRING $SOLVER_TYPE
                python $MYDIR/main.py ${ITERATION_STRING} ${SOLVER_TYPE}
            done
        done
    done
done
#rm -rf $DATADIR
#mkdir $DATADIR
#mkdir $DATADIR/csv && mkdir $DATADIR/img
#mv *.csv $DATADIR/csv
#mv *.html $DATADIR/img
