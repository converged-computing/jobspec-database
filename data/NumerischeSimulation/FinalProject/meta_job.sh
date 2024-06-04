#!/bin/bash
####################################
#
# Executing a number of scenarios with different settings on the cluster
#
####################################

# load modules
module use /usr/local.nfs/sgs/modulefiles
module load gcc/10.2
module load openmpi/3.1.6-gcc-10.2
module load vtk/9.0.1
module load cmake/3.18.2

# compile code
# rm -rf build/
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make install

# iterate through list of scenarios
declare -a StringArray=( "tesla_140-84_left_lowRe_underRelax" "tesla_140-84_left_lowRe_noRelax" "tesla_140-84_right_lowRe_underRelax" "tesla_140-84_right_lowRe_noRelax" "tesla_140-84_left_highRe_noRelax" "tesla_140-84_left_highRe_underRelax" "tesla_140-84_right_highRe_underRelax" "tesla_140-84_right_highRe_noRelax")

for val in ${StringArray[@]}; do
   # request resources
   #SBATCH --job-name=submission_${val}
   #SBATCH --output=results_${val}.txt
   #
   #SBATCH --ntasks=1
   #SBATCH --time=100:00
   echo $val
   ./finalproject ../ini/${val}.txt > logs_${val}.txt &
done

# show job number
jobs
# fg <job_number>