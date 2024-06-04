#! /bin/bash
#SBATCH --job-name eagle_profile
#SBATCH --nodes 1
#SBATCH -o submit_eagle_%A_%a.out
#SBATCH -e submit_eagle_%A_%a.err
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --mem=4000

# Create the basic profiling information using "eagle_profile.submit" (this script) and "eagle_profile.slurm.template" scripts
# e.g. sbatch eagle_profile.submit <ngpu> 
# Now use sort and tail to remove the extra header strings "profile,itnum,ncpu,ngpu,function,time_ms"
# cat eagle_11710257_*.res | grep profile, | sort -h -r | tail -n +9
# cat eagle_compact_*_gpu_*_ncpu_01.res | grep profile, | sort -h -r | tail -n +2
# cat eagle_compact*_gpu_[1,2]_ncpu_14.res | grep profile, | sort -h -r | tail -n +2 > eagle_compact_gpu_1_2_ncpu_14.txt
# cat eagle_REPEATS_ss*.res | grep profile, | sort -h -r | tail -n +90 > eagle_REPEATS_ss.txt

if [ -z $1 ] ; then
echo "Error: You must specify the number of GPUs to use on the sbatch <scriptname> <ngpu> command line"
exit -1
fi


cd /flush1/bow355/AMplus_new_code/Large
# export SING_CUDA_ACC=/flush1/bow355/AMplus_new_code/Mid_docker_tests/mro_cuda8_eagle_acc2.img
# imtsc-cont-reg.it.csiro.au/eagle/mro_cuda8_eagle_acc2_hdf
export SING_CUDA_ACC=/flush1/bow355/AMplus_new_code/Mid_docker_tests/mro_cuda8_eagle_acc2_hdf-latest.img


MAXRAM=128000
MAXTHREADS=28
NGPU=$1
if [ "$NGPU" -gt "0" ] ; then
THREADLIST=("01" "02" "04" "08" "10" "12" "14")
GPUSTRING="#SBATCH --gres=gpu:$1"
else
THREADLIST=("01" "02" "04" "08" "10" "12" "14" "21" "28")
# THREADLIST=("10" "12" "14" "21" "28")
GPUSTRING=""
fi

for NTHREADS in "${THREADLIST[@]}"; do
    
    sed  -e "s/_OMP_CPU_TEMPLATE_/$NTHREADS/g" -e "s/_NUM_GPU_TEMPLATE_/$NGPU/g" am_big.R.template  > am_big_${SLURM_JOBID}_${NTHREADS}_${NGPU}.R
    RFILE=${SLURM_JOBID}_${NTHREADS}_${NGPU}
    sed -e "s/_RFILE_TEMPLATE_/$RFILE/g" -e "s/_SLURM_JOBID_TEMPLATE_/$SLURM_JOBID/g" -e "s/_NUM_CPU_TEMPLATE_/$MAXTHREADS/g" -e "s/_NUM_RAMMB_TEMPLATE_/$MAXRAM/g" -e "s/_OMP_CPU_TEMPLATE_/$NTHREADS/g" -e "s/_GRES_GPU_TEMPLATE_/$GPUSTRING/g" -e "s/_NUM_GPU_TEMPLATE_/$NGPU/g" eagle_profile.slurm.template > eagle_REPEAT_cs_ROWMAJOR_MASTER_${SLURM_JOBID}_${NTHREADS}_${NGPU}.slurm
    
    sbatch eagle_REPEAT_cs_ROWMAJOR_MASTER_${SLURM_JOBID}_${NTHREADS}_${NGPU}.slurm
done
