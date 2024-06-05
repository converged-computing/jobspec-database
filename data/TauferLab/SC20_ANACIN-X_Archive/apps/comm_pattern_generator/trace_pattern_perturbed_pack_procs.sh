#!/usr/bin/env bash

#SBATCH -o comm_pattern-%j.out
#SBATCH -e comm_pattern-%j.err

n_procs=$1
app=$2
config=$3

# Determine number of nodes we need to run on
system=$(hostname | sed 's/[0-9]*//g')
if [ ${system} == "quartz" ]; then
    n_procs_per_node=36
elif [ ${system} == "catalyst" ]; then
    n_procs_per_node=24
fi
n_nodes=$(echo "(${n_procs} + ${n_procs_per_node} - 1)/${n_procs_per_node}" | bc)

# Define tool stack
anacin_x_root=$HOME/ANACIN-X
pnmpi=${anacin_x_root}/submodules/PnMPI/build_${system}/lib/libpnmpi.so
pnmpi_lib_path=${anacin_x_root}/anacin-x//pnmpi/patched_libs/
pnmpi_conf=${anacin_x_root}/anacin-x/pnmpi/configs/dumpi_mini_ninja.conf
mini_ninja_cfg=$HOME/Src_mini_ninja/config/uniform.json

LD_PRELOAD=${pnmpi} PNMPI_LIB_PATH=${pnmpi_lib_path} PNMPI_CONF=${pnmpi_conf} MINI_NINJA_CONFIG=${mini_ninja_cfg} srun -N${n_nodes} -n${n_procs} ${app} ${config}
