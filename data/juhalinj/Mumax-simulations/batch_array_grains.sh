#!/bin/bash

#SBATCH --time=0-2:00:00        # maximum runtime, D-HH:MM:SS
#SBATCH --mem=1500              # memory cap (500MB seems to work most of the time)
#SBATCH --gres=gpu:1            # 1 GPU, of any type (the crucial step for mumax)
#SBATCH --array=1-11


# load CUDA if you haven't already:
module --ignore-cache load cuda/11.0

SID=$SLURM_ARRAY_TASK_ID
PROJECTNAME="Grain-simulations"
randomSeed="${SID}"
OUTNAME="SOT2_orginal_${randomSeed}"
OUTFILE="${OUTNAME}.mx3"
OUTFOLD="${OUTNAME}.out"

cat <<EOF > $OUTFILE
setGridSize(1024, 1024, 1)
setCellSize(2.5e-9, 2.5e-9, 0.8e-9)
setPBC(32,0,0)

OutputFormat = OVF2_BINARY


m = NeelSkyrmion(1, -1).transl(-1000e-9,0,0)


Msat = 900e3
Aex     = 6.5e-12
anisU   = vector(0, 0, 1)
Ku1     = 0.70e6

alpha   = 0.1
Dind = 2.0e-3

// Make the grains vary 10
// Average size 10 nm across
grainSize  := 10e-9  // m

// random number seed. Very to obtain different distributions
randomseed := ${randomSeed}

// There will be 255 different values of the anisotropy. Multiple grains belong to one region, and so have the same anisotropy. 
maxRegion  := 255
ext_makegrains(grainSize, maxRegion, randomSeed)

// Use a loop to assign a value of the anisotropy to each region. The width is given by the multiplier - here 10 % of the average value. Vary also K and the percentage value of ten
for i:=0; i<maxRegion; i++{
	K := 0.70e6
	Ku1.SetRegion(i, K + randNorm() * 0.10 * K)
}
// FixDt = 5e-14

// Constants
hbar := 1.0545718e-34
e := 1.6021766e-19
AlphaH := 0.15
d := 0.8e-9
Ms := 900e3
p := Constvector(0,-1,0)
xi_SOT := -2.0
J_SOT := abs(-2.0e11)

aj := Const(J_SOT*(hbar/2.*AlphaH/e/d/Ms))
bj := Mul(aj,Const(xi_SOT))

dampinglike := Mul(aj,Cross(m,p))
AddFieldTerm(dampinglike)

fieldlike := Mul(bj,p)
AddFieldTerm(fieldlike)
tableAdd(ext_bubblepos)
tableAdd(ext_bubblespeed)
tableAdd(ext_bubbledist)
tableadd(ext_topologicalcharge)
tableadd(ext_topologicalchargedensity)
TableAdd(B_ext)

B_ext = vector(0, 0, 200.0e-3)

relax()

autosave(m, 5e-10)
AutoSnapshot(m, 5e-10)


tableAutosave(5.0e-10)
run(25e-9)
EOF
# run the desired .mx3 file:
srun $WRKDIR/mumax3.10_linux_cuda11.0/mumax3 -cache "$WRKDIR/SOT_simulations/Cache" $WRKDIR/SOT_simulations/$PROJECTNAME/$OUTFILE

