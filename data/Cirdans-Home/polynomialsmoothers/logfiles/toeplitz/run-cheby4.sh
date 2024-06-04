#!/bin/bash
#SBATCH --job-name=pol-@NTASK@-@NTHREAD@
#SBATCH --ntasks=@NTASK@
#SBATCH --cpus-per-task=@NTHREAD@
#SBATCH --output=log-@NTASK@-@NTHREAD@.out
#SBATCH --partition=all

module load gcc/12.2.0 openmpi/4.1.4-gcc-12.2.0 openblas/0.3.20-gcc-12.2.0 metis/5.1.0-gcc-12.2.0 suite-sparse/5.10.1-gcc-12.2.0 \
	libunwind/1.6.2-gcc-12.2.0 papi/6.0.0.1-gcc-12.2.0 binutils/2.33.1-gcc-12.2.0 libxml2/2.10.1-gcc-12.2.0 mumps/5.4.1-gcc-12.2.0
module list

# export OMP_NUM_THREADS=@NTHREAD@
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/software/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-12.2.0/metis-5.1.0-kuao2tv62i5uiifpbtfnx2nqa4zuaaci/lib:/data/software/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-12.2.0/openblas-0.3.20-4ipfq4m2x4ku6edyd6o6sq4zxprh4kmo/lib/:/data/software/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-12.2.0/suite-sparse-5.10.1-klxivyhmvz5ckb3zv4fzjtwk6itursni/lib:/data/software/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-12.2.0/netlib-scalapack-2.2.0-b4njionfcuwwa3heilv7dhvetprj2wmo/lib

idim=@SIZE@
theta=@THETA@
epsilon=@EPS@

srun ./anisopsblas >> cheby4/log_vsvbm_cheby4_l1jac_${idim}_task_@NTASK@_thr_@NTHREAD@.txt 2>&1  <<EOF
%%%%%%%%%%%  General  arguments % Lines starting with % are ignored.
CSR                     ! matrix storage format
${idim}                 ! Discretization grid size
${theta}                ! Theta Coefficient (Degree)
${epsilon}              ! Epsilon of the anisotropy
FCG                     ! Krylov Solver
2                       ! Stopping criterion
500                     ! Maximum number of iterations
1                       ! Trace of FCG
500                     ! Restart (RGMRES e BICGSTAB)
1.d-7                   ! Tolerance
%%%%%%%%%%%  Main preconditioner choices %%%%%%%%%%%%%%%%
ML-VSVBM-4CHEB4-30L1JAC ! verbose description of the prec
ML                      ! Preconditioner type
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
4                       ! degree for polynomial smoother
POLY_LOTTES             ! polynomial variant
0                       ! number of overlap layers
HALO                    ! restriction  over application of AS
NONE                    ! prolongation over application of AS
L1-JACOBI               ! local subsolver
1                       ! inner solver sweeps
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
1                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Second smoother, always ignored for non-ML  %%%%%%%%%%%%%%%%
NONE                    ! Second (post) smoother, ignored if NONE
1                       ! Number of sweeps for (post) smoother
4                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Multilevel parameters %%%%%%%%%%%%%%%%
VCYCLE                  ! AMG cycle type
1                       ! number of 1lev/outer sweeps
-3                      ! Max Number of levels in a multilevel preconditioner; if <0, lib default
-3                      ! Target coarse matrix size per process; if <0, lib default
SMOOTHED                ! Type of aggregation: SMOOTHED UNSMOOTHED
DEC                     ! Parallel aggregation: DEC, SYMDEC, COUPLED
SOC1                    ! aggregation measure SOC1, MATCHBOXPcall read_data(prec%aggr_size,inp_unit) ! Requested size of the aggregates for MATCHBOXP
8                       ! Requested size of the aggregates for MATCHBOXP
NATURAL                 ! Ordering of aggregation NATURAL DEGREE
-1.5                    ! Coarsening ratio, if < 0 use library default
FILTER                  ! Filtering of matrix:  FILTER NOFILTER
-0.0100d0               ! Smoothed aggregation threshold, ignored if < 0
-2                      ! Number of thresholds in vector, next line ignored if <= 0
0.05 0.025              ! Thresholds
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
DIST                    ! coarsest mat layout
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%%  Dump parms %%%%%%%%%%%%%%%%%%%%%%%%%%
F                       ! Dump on file?
1                       ! Minimum level to dump
10                      ! Maximum level to dump
T                       ! Dump coarse matrices
T                       ! Dump restrictors
T                       ! Dump prolongators
T                       ! Dump smoothers
T                       ! Dump coarse solver
T                       ! Dump using global numbering?
EOF
