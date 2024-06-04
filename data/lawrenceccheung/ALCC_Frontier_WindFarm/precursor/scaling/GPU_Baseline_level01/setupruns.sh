#!/bin/sh

RUNDIR=/lustre/orion/cfd162/scratch/lcheung/ALCC_Scaling/GPU_Baseline_level01
BASENAME=Nodes
NODES='1 2'
#NODES='4 8 16 32 64 128 256 512'
EMAIL=lcheung@sandia.gov
CONFFILE=MedWS_LowTI_precursor01.inp
SUBMITFILE=frontier.sh

for NNODES in $NODES; do
    NGPU=$((NNODES*8))
    NNODES_FORMAT=$(printf "%04d" $NNODES)
    echo "Creating case ${NNODES_FORMAT}"
    # Make the directories
    NODEDIR=$RUNDIR/${BASENAME}_${NNODES_FORMAT}
    mkdir -p $NODEDIR
    cp -av $CONFFILE $NODEDIR/
    cat <<EOF > $NODEDIR/$SUBMITFILE
#!/bin/bash

#SBATCH --job-name=${BASENAME}_${NNODES_FORMAT}
#SBATCH --account=CFD162
#SBATCH --time=2:00:00
#SBATCH --nodes=$NNODES
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive
# #SBATCH -q debug

export rocm_version=5.4.3
export MPICH_GPU_SUPPORT_ENABLED=1

module purge
module load amd/\${rocm_version}
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cray-mpich

source /lustre/orion/cfd162/proj-shared/lcheung/spackbuilds/spack-manager.2/loadexawind.sh

export LD_LIBRARY_PATH=/opt/rocm-\${rocm_version}/llvm/lib/:$LD_LIBRARY_PATH
export HIP_LAUNCH_BLOCKING=1

export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software
export CONFFILE=$CONFFILE

srun -N $NNODES -n $NGPU --gpus-per-node=8 --gpu-bind=closest amr_wind \$CONFFILE

EOF
    
done
