#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   *** Serial Job on SKX Normal Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J myjob                        # Job name
#SBATCH -o myjob.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e myjob.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                   # Queue (partition) name
#SBATCH -N 1                            # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 03:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=siyuan.ma@utexas.edu
#SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A Hardware-Acceleratio         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...
module load intel/18.0.2 python3/3.7.0
module load cuda/10.1 cudnn/7.6.5 nccl/2.5.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda10_1/lib64
source $WORK/Lab1B_virtualenv/bin/activate
mkdir -p $WORK/Lab1B/output


home_path=$WORK/HRDWR-ARCH-4-ML/Lab1
output_path=${home_path}/output_nvprof
mkdir -p ${output_path}

# Launch serial code...
for code in 1 10 1101
do
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed --openacc-profiling off --metrics inst_executed python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_ipc --openacc-profiling off --metrics ipc python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_control --openacc-profiling off --metrics inst_control python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_integer --openacc-profiling off --metrics inst_integer python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_fp_64 --openacc-profiling off --metrics inst_fp_64 python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_fp_32 --openacc-profiling off --metrics inst_fp_32 python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_cf_fu_utilization --openacc-profiling off --metrics cf_fu_utilization python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_double_precision_fu_utilization --openacc-profiling off --metrics double_precision_fu_utilization python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_special_fu_utilization --openacc-profiling off --metrics special_fu_utilization python3 ${home_path}/mlp_keras.py ${code}
    #nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_single_precision_fu_utilization --openacc-profiling off --metrics single_precision_fu_utilization python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_dram_read_transactions --metrics dram_read_transactions python ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_dram_write_transaction --metrics dram_write_transaction python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed_global_loads --metrics inst_executed_global_loads python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed_local_loads --metrics inst_executed_local_loads python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed_shared_loads --metrics inst_executed_shared_loads python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed_surface_loads --metrics inst_executed_surface_loads python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed_global_stores --metrics inst_executed_global_stores python3 ${home_path}/mlp_keras.py ${code}
    # nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_inst_executed_local_stores --metrics inst_executed_local_stores python3 ${home_path}/mlp_keras.py ${code}
    nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_dram_read_throughput --openacc-profiling off --metrics dram_read_throughput python3 ${home_path}/mlp_keras.py ${code}
    nvprof --csv --log-file ${output_path}/mlp_nvprof_${code}_stall_memory_throttle --openacc-profiling off --metrics stall_memory_throttle python3 ${home_path}/mlp_keras.py ${code}
done
# ---------------------------------------------------
