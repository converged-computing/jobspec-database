#!/bin/sh

#This script requires at lease a singular parameter, an executable. 
#This should be the first parameter. 
#Any further parameters should be the ones required by the executable itself,
#  as these will get passed to the program when ran by the compute node.

if [ $# -lt 1 ]; then
	echo "Please provide a path to the executable you want to run"
	exit 1
fi

if [ -x "$1" ]; then
	echo "Program: $@"
else
	echo "The file provided is not an executable. Please provide an executable"
	exit 1
fi

if [ $( hostname ) != "kudu" ]; then
	echo "This is not being ran on kudu. Please run the following to connect to kudu from any DCS machine."
	echo "    ssh kudu"
	exit 1
fi

echo "#!/bin/sh

#SBATCH --job-name=$( whoami )-$1
#SBATCH --partition=gecko  
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2500M
#SBATCH --gres=gpu:0
#SBATCH --time=60:00
#SBATCH --output=%j/cs310_output_%j.out
#SBATCH --error=%j/cs310_error_%j.err

echo "===== ENVIRONMENT ====="

. /etc/profile.d/modules.sh
echo "===== CUDA INFO ====="

echo "CUDA 12.3"


echo "===== COMPILER INFO ====="

echo "GCC 12.2.0"

echo "===== CPU INFO ====="

lscpu

echo "===== GPU INFO ====="

lshw -C display

echo "===== GPU INFO NVIDIA-SMI ====="

nvidia-smi

echo "===== RUNNING ON $( hostname ) ====="

#############################################################################
# Want to add your own environment variables to the program? Add them here! #

#############################################################################
echo
" > tmp

if [ -f "Makefile" ]; then
	echo "echo ===== COMPILING Makefile IN $( pwd ) =====

make clean
make -j KOKKOS_DEVICES=OpenMP

export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads



echo
" >> tmp
fi

echo "echo ===== RUNNING $@ =====
srun $@" >> tmp

BATCH=$( sbatch tmp )
BATCHNO=$( echo $BATCH | sed 's/[^0-9]//g' )

rm tmp

mkdir $BATCHNO

echo "===== Job $BATCHNO has been submitted! ====="

echo "===== My jobs ====="
echo "Note: Don't worry if the job is PENDING, the job will be ran as soon as possible."
squeue -u $( whoami ) -o "%.8i %.20j %.10T %.5M %.20R %.20e"

#ENDTIME=$( squeue -o "%i %e" | grep $BATCHNO | cut -d " " -f 2 )

#echo "===== Job $BATCHNO has been submitted, should be finished by $ENDTIME ====="