#!/bin/bash
#SBATCH --job-name=pol-@NTASK@-@NTHREAD@
#SBATCH --nodes=@NNODES@
#SBATCH --ntasks-per-node=@NTASKS@
#SBATCH --gpus=@NGPUS@
#SBATCH --cpus-per-task=@NTHREAD@
#SBATCH --output=log-@NTASK@-@NTHREAD@.out
#SBATCH --partition=gpu

module purge
module load gpu-gcc/12.2.0 gpu-cuda/12.3.1-gcc-12.2.0 gpu-openmpi/4.1.6-cuda-12.3.1-gcc-12.2.0 gpu-metis/5.1.0-gcc-12.2.0 gpu-openblas/0.3.26-gcc-12.2.0 
module list

idim=@SIZE@
theta=@THETA@
epsilon=@EPS@
psize=@NGPUS@

srun ./anisopsblascuda_multi >> match/multi/log_match_l1jac_${idim}_task_${psize}_thr_1.txt 2>&1  <<EOF
%%%%%%%%%%%  General  arguments % Lines starting with % are ignored.
HLG                     ! matrix storage format
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
ML                      ! Preconditioner type
VCYCLE                  ! AMG cycle type
1                       ! number of 1lev/outer sweeps
-3                      ! Max Number of levels in a multilevel preconditioner; if <0, lib default
-3                      ! Target coarse matrix size per process; if <0, lib default
DIST                    ! Coarsest matrix layout
SMOOTHED                ! Type of aggregation: SMOOTHED UNSMOOTHED
COUPLED                 ! Parallel aggregation: DEC, SYMDEC, COUPLED
MATCHBOXP               ! aggregation measure SOC1, MATCHBOXPcall read_data(prec%aggr_size,inp_unit) ! Requested size of the aggregates for MATCHBOXP
8                       ! Requested size of the aggregates for MATCHBOXP
NATURAL                 ! Ordering of aggregation NATURAL DEGREE
-1.5                    ! Coarsening ratio, if < 0 use library default
FILTER                  ! Filtering of matrix:  FILTER NOFILTER
-0.0100d0               ! Smoothed aggregation threshold, ignored if < 0
-2                      ! Number of thresholds in vector, next line ignored if <= 0
0.05 0.025              ! Thresholds
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-4CHEBY4-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
4                       ! degree for polynomial smoother
POLY_LOTTES             ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-4CHEBY4OPT-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
4                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-4CHEBY1OPT-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
4                       ! degree for polynomial smoother
POLY_NEW                ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-4L1JAC-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
L1-JACOBI               ! smoother type
4                       ! (pre-)smoother / 1-lev prec sweeps
1                       ! degree for polynomial smoother
POLY_NEW                ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-6CHEBY4-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
6                       ! degree for polynomial smoother
POLY_LOTTES             ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-6CHEBY4OPT-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
6                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-6CHEBY1OPT-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
6                       ! degree for polynomial smoother
POLY_NEW                ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-6L1JAC-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
L1-JACOBI               ! smoother type
6                       ! (pre-)smoother / 1-lev prec sweeps
1                       ! degree for polynomial smoother
POLY_NEW                ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-8CHEBY4-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
8                       ! degree for polynomial smoother
POLY_LOTTES             ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-8CHEBY4OPT-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-8CHEBY1OPT-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
POLY                    ! smoother type
1                       ! (pre-)smoother / 1-lev prec sweeps
8                       ! degree for polynomial smoother
POLY_NEW                ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%%%%%%% Smoother name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ML-VMATCH-8L1JAC-L1JAC30
%%%%%%%%%%%  First smoother (for all levels but coarsest) %%%%%%%%%%%%%%%%
L1-JACOBI               ! smoother type
8                       ! (pre-)smoother / 1-lev prec sweeps
1                       ! degree for polynomial smoother
POLY_NEW                ! polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
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
8                       ! degree for polynomial smoother
POLY_LOTTES_BETA        ! Polynomial variant
POLY_RHO_EST_POWER      ! Algorithm to estimate spectral radius (ignored if next larger than 0)
1.0                     ! Spectral radius estimate
0                       ! Number of overlap layers for AS preconditioner
HALO                    ! AS restriction operator: NONE HALO
NONE                    ! AS prolongation operator: NONE SUM AVG
L1-JACOBI               ! Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
1                       ! Inner solver sweeps (GS and JACOBI)
LLK                     ! AINV variant
0                       ! Fill level P for ILU(P) and ILU(T,P)
8                       ! Inverse Fill level P for INVK
1.d-4                   ! Threshold T for ILU(T,P)
%%%%%%%%%%%  Coarse level solver  %%%%%%%%%%%%%%%%
L1-JACOBI               ! coarsest-lev solver
L1-JACOBI               ! coarsest-lev subsolver
FCG                     ! type of Krylov method
0                       ! fill-in for incompl LU
1.d-2                   ! Threshold for ILUT
30                      ! sweeps for GS/JAC subsolver
%%%%%% END OF SMOOTHERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
END-OF-TESTS
EOF

