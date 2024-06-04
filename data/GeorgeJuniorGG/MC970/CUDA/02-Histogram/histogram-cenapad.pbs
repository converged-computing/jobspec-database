#!/bin/bash
#PBS -N histogram 
#PBS -q testegpu
#PBS -e job_output.out
#PBS -o job_output.err
#PBS -l walltime=00:25:00


# load our environment
module purge
module load gcc/9.4.0
module load cmake/3.21.3-gcc-9.4.0
module load cuda/11.5.0-gcc-9.4.0

# cd to job directory
cd $PBS_O_WORKDIR

cmake -E remove -f build
cmake -E make_directory build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/opt/pub/spack/cuda/11.5.0/gcc/9.4.0/bin/nvcc
cd ..

###
### @swineone's script
###
cd build
make
cd ..
for a in 1 2 3 4 5
do
        build/parallel tests/$a.in \
              1>build/parallel.$a.out \
              2>build/parallel.$a.time
        cat build/parallel.$a.out

        build/serial tests/$a.in \
              1>build/serial.$a.out \
              2>build/serial.$a.time
        cat build/serial.$a.out

        diff -u tests/$a.out build/parallel.$a.out >&2
        diff_status=$?

        ser=$(<build/serial.$a.time)
        par=$(<build/parallel.$a.time)
        speedup=$(bc -l <<< "scale=4; $ser/$par")
        echo "  Serial runtime: ${ser}s"
        echo "Parallel runtime: ${par}s"
        echo "         Speedup: ${speedup}x"

        if [[ ! -f runtime.csv ]]; then
                echo "# Input,Serial time,Parallel time,Speedup" > runtime.csv
        fi

        if [ $diff_status -eq 0 ]; then
                echo "$a,$ser,$par,$speedup" >> runtime.csv
        fi
done
