#!/bin/bash            
#OAR -n HL-31
#OAR -l /nodes=14,walltime=8:00:00
#OAR -O HL-31_%jobid%.out
#OAR -E HL-31_%jobid%.err
#OAR --project hifimagnet
#OAR --notify mail:user_email

# Setup environment
source /applis/site/nix.sh
export NIX_PATH="nixpkgs=$HOME/.nix-defexpr/channels/nixpkgs"
#nix-env --switch-profile $NIX_USER_PROFILE_DIR/feelpp-singularity
nix-env --switch-profile $NIX_USER_PROFILE_DIR/nur-openmpi4
echo "Load nix-env:" $(nix-env -q)

echo "Singularity:" $(which singularity)
SVERSION=$(singularity --version)
echo "Singularity Version:" ${SVERSION}

nbcores=$(cat $OAR_NODE_FILE|wc -l)
nbnodes=$(cat $OAR_NODE_FILE|sort|uniq|wc -l)
njobs=$(cat $OAR_FILE_NODES | wc -l)

echo "nbcores=" $nbcores
echo "nbnodes=" $nbnodes
echo "np=" $njobs
echo "OAR_WORKDIR=" ${OAR_WORKDIR}

# Exec env (TODO param SVERSION)
IMGS_DIR=/bettik/$USER/singularity
#HIFIMAGNET=${IMGS_DIR}/hifimagnet-thermobox-P2-v0.108-singul2.6.simg
HIFIMAGNET=${IMGS_DIR}/hifimagnet-thermobox-debianpack.sif
if [ ! -f $HIFIMAGNET ]; then
	echo "Cannot find ${HIFIMAGNET}"
fi

# IO env
# # Get the name of the scratch temporary directory
# TMPDIR=/bettik/$USER/dahu/oar.$OAR_JOB_ID
# # Go into the scratch tmp directory
# cd $TMPDIR

# Sim setup
SIMDIR=/bettik/$USER/HL-31
CFG=M19061901-full.cfg
LOG=M19061901-full.log
if [ ! -f $CFG ]; then
	echo "Cannot find ${CFG}"
fi

OUTDIR=${SIMDIR}/full
mkdir -p ${OUTDIR}
echo "create ${OUTDIR}"

# Check if mesh is correctly partitionned: aka nparts==njobs

# Run the program
echo "Running"
echo " mpirun -np ${njobs} \
  -machinefile $OAR_NODEFILE -mca plm_rsh_agent "oarsh" --prefix $HOME/.nix-profile \
     singularity exec -H ${OAR_WORKDIR} -B $OUTDIR:/feel $HIFIMAGNET \
       feelpp_hfm_coupledcartmodel_3DP1N1 --config-file ${CFG}"

mpirun -np ${njobs} \
       -machinefile $OAR_NODEFILE \
       -mca plm_rsh_agent "oarsh" -mca btl_openib_allow_ib true \
       --prefix $HOME/.nix-profile \
     singularity exec -H ${OAR_WORKDIR} -B $OUTDIR:/feel $HIFIMAGNET \
       feelpp_hfm_coupledcartmodel_3DP1N1 --config-file ${CFG} 

# # Get the results back to my home if TPMDIR 
# cp -a ./results $OUTDIR
