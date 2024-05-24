#!/bin/bash -l
#SBATCH -N 4               # Number of nodes
#SBATCH -J roi
#SBATCH -L SCRATCH          # job requires SCRATCH files
#SBATCH -A m3890_g          # allocation
#SBATCH -C gpu
#SBATCH -q early_science    # regular queue
#SBATCH -t 02:00:00         # wall clock time limit
#SBATCH --gpus-per-node 4
#SBATCH -o job%j.out
#SBATCH -e job%j.err

export WORK=$SCRATCH/adse13_249/LY99
cd $WORK

mkdir -p $SLURM_JOB_ID; cd $SLURM_JOB_ID

export OUT_DIR=${PWD}
# NO PSF:
export DIALS_OUTPUT=${WORK}/927185
# WITH PSF:
#export DIALS_OUTPUT=${WORK}/927187
export CCTBX_NO_UUID=1
export DIFFBRAGG_USE_CUDA=1

echo "dispatch.step_list = input filter statistics_unitcell model_statistics annulus
input.path=${DIALS_OUTPUT}
input.experiments_suffix=00.img_integrated.expt
input.reflections_suffix=00.img_integrated.refl
input.keep_imagesets=True
input.read_image_headers=False
input.persistent_refl_cols=shoebox
input.persistent_refl_cols=bbox
input.persistent_refl_cols=xyzcal.px
input.persistent_refl_cols=xyzobs.px.value
input.persistent_refl_cols=delpsical.rad
input.persistent_refl_cols=panel
input.parallel_file_load.method=uniform
scaling.model=${WORK}/1m2a.pdb
scaling.unit_cell=67.2 59.8 47.2 90 110.3 90
scaling.space_group=C2
scaling.resolution_scalar=0.993420862158964
filter.algorithm=unit_cell
filter.unit_cell.algorithm=cluster
filter.unit_cell.cluster.covariance.file=${WORK}/covariance_ly99sim_30000.pickle
filter.unit_cell.cluster.covariance.component=0
filter.unit_cell.cluster.covariance.mahalanobis=4.0
filter.outlier.min_corr=-1.0
merging.d_max=None
merging.d_min=2.1
statistics.annulus.d_max=2.5
statistics.annulus.d_min=2.1
spread_roi.enable=True
spread_roi.strong=2.0
output.output_dir=${OUT_DIR}
output.log_level=0 # stdout stderr
exafel.trusted_mask=${WORK}/pixels.mask
exafel.scenario=3A
exafel.shoebox_border=0
exafel.context=kokkos_gpu
exafel.model.plot=False
exafel.model.mosaic_spread.value=0.08
exafel.model.Nabc.value=72,72,72
exafel.debug.lastfiles=False
exafel.debug.verbose=False
exafel.debug.finite_diff=-1
exafel.debug.eps=1.e-8
exafel.skin=True # whether to use diffBragg
exafel{
  refpar{
    label = *background *G *rot
    background {
      algorithm=rossmann_2d_linear
      scope=spot
      slice_init=border
      slice=all
    }
    G {
      scope=lattice
    }
  }
}
" > annulus.phil

echo "jobstart $(date)";pwd
CCTBX_GPUS_PER_NODE=4 srun -n 32 cctbx.xfel.merge annulus.phil
echo "jobend $(date)";pwd
