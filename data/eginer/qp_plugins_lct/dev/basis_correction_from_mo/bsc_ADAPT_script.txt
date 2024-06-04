#!/bin/bash
#set -x
#PBS -l nodes=1:ppn=24
##PBS -l cput=455628:00:00
##PBS -l vmem=1000gb
#Request output and error output to be merged in one file 
#Request environment variables to be exported to jobs
#PBS -V
#Request name of job to be demo_gaussian
#PBS -N H2O_STO3G

hostname
SCRATCHDIR=/scratch/dtraore/${PBS_JOBID}    # scratch directory
export SCRATCHDIR                       # export SCRATCHDIR for job
echo $SCRATCHDIR
mkdir $SCRATCHDIR
echo "#################"
cat $PBS_NODEFILE                       # Name of the node
cp $PBS_O_WORKDIR/* $SCRATCHDIR
cd   $SCRATCHDIR
echo $SCRATCHDIR

ulimit -s unlimited
export SCRATCHDIR=/scratch/$USER/$PBS_JOBID
#
cd $PBS_O_WORKDIR


############################################################################
####### YOU PUT THE PATH TO YOUR

#module load python/3.9.12 
QP_ROOT=/home/dtraore/programs/qp2   # <----------- ADRESSE DE TON QP2
source ${QP_ROOT}/quantum_package.rc
####### YOU LOAD SOME LIBRARIES
#alias python3='/programmes/installation/Python/3.9.12/bin/python3'
#type -a python3

#export OMP_NUM_THREADS=12

#module load intel2016_OMPI-V2

ADAPT_ROOT=/home/dtraore/master/UCC-VQE
ADAPT1=${ADAPT_ROOT}/main_short_frozen_core.py
ADAPT2=${ADAPT_ROOT}/main_short_frozen_core2.py

#system_list='H4 H6 LiH FH Be BH H2O'
system_list='H2O'
n_qubits=12
max_iter=200
range_active_min=1
range_active_max=7

basis=sto-3g
multiplicity=1

python --version | tee python_version.out


rm -rf step*
for system in ${system_list}
do

geom=${system}.xyz
# STEP 0 : Create initial FCIDUMP
mkdir step0
cd step0
cp ../${geom} .
cp ../${basis} .
qp create_ezfio -b ${basis} -m ${multiplicity} -o ${system}_${basis}.ezfio ${geom}
qp run scf | tee scf_${system}_${basis}.out
qp_set_frozen_core ${system}_${basis}.ezfio | tee fro_core.out
qp run fcidump_pyscf
#qp run write_eff_basis_pot_in_file | tee write_eff_basis_pot_in_file.out
qp unset_file
cd ..

# STEP 1 : ADAPT iter 1
mkdir step1
cd step1
cp ../step0/${system}_${basis}.ezfio.FCIDUMP .
python ${ADAPT1} ${system}_${basis}.ezfio.FCIDUMP ${n_qubits} ${system} ${max_iter} ${range_active_min} ${range_active_max} | tee ADAPT_step1.out 
sed -i 's/\[\[//g' mo_beta_1rdm.txt
sed -i 's/\[\[//g' mo_alpha_1rdm.txt
sed -i 's/]]//g' mo_beta_1rdm.txt
sed -i 's/]]//g' mo_alpha_1rdm.txt
cd ..

# STEP 2 : Build new \bar{V}^B
mkdir step2
cd step2
cp ../step1/mo**_1rdm.txt .
cp ../${geom} .
cp ../${basis} .
cp -rf ../step0/${system}_${basis}.ezfio .
qp set_file ${system}_${basis}.ezfio
qp set sc_basis_corr basis_cor_func pbe_ueg
qp run write_dm_in_aux_quantities | tee write_dm_in_aux_quantities.out
qp set density_for_dft density_for_dft input_density
cp ../step0/pot_basis_alpha_mo.txt .
cp ../step0/pot_basis_beta_mo.txt .
qp run print_eff_basis_potential | tee print_eff_basis_potential.out
qp run basis_correction | tee basis_correction.out
qp run print_dipole_moment_from_input_density | tee print_dipole_moment_from_input_density.out
qp run fcidump_for_vbarb
qp run write_eff_basis_pot_in_file | tee write_eff_basis_pot_in_file.out
mv ${system}_${basis}.ezfio.vbarb_FCIDUMP ${system}_${basis}.ezfio.vbarb_FCIDUMP_2
cd ..

# STEP 3 : ADAPT iter 2
mkdir step3
cd step3
cp ../step2/${system}_${basis}.ezfio.vbarb_FCIDUMP_2 .
cp ../step1/ansatz.pkl .
python ${ADAPT2} ${system}_${basis}.ezfio.vbarb_FCIDUMP_2 ${n_qubits} ${system} ${max_iter} ${range_active_min} ${range_active_max}| tee ADAPT_step3.out 
sed -i 's/\[\[//g' mo_beta_1rdm.txt
sed -i 's/\[\[//g' mo_alpha_1rdm.txt
sed -i 's/]]//g' mo_beta_1rdm.txt
sed -i 's/]]//g' mo_alpha_1rdm.txt
cd ..

# STEP 4 : Build new \bar{V}^B
mkdir step4
cd step4
cp ../step3/mo**_1rdm.txt .
cp ../${geom} .
cp -rf ../step0/${system}_${basis}.ezfio .
qp set_file ${system}_${basis}.ezfio
qp set sc_basis_corr basis_cor_func pbe_ueg
qp run write_dm_in_aux_quantities | tee write_dm_in_aux_quantities.out
qp set density_for_dft density_for_dft input_density
cp ../step2/pot_basis_alpha_mo.txt .
cp ../step2/pot_basis_beta_mo.txt .
qp run print_eff_basis_potential | tee print_eff_basis_potential.out
qp run basis_correction | tee basis_correction.out
qp run print_dipole_moment_from_input_density | tee print_dipole_moment_from_input_density.out
qp run fcidump_for_vbarb
qp run write_eff_basis_pot_in_file | tee write_eff_basis_pot_in_file.out
mv ${system}_${basis}.ezfio.vbarb_FCIDUMP ${system}_${basis}.ezfio.vbarb_FCIDUMP_2
cd ..

# STEP 5 : ADAPT iter 3
mkdir step5
cd step5
cp ../step4/${system}_${basis}.ezfio.vbarb_FCIDUMP_2 .
cp ../step3/ansatz.pkl .
python ${ADAPT2} ${system}_${basis}.ezfio.vbarb_FCIDUMP_2 ${n_qubits} ${system} ${max_iter} ${range_active_min} ${range_active_max}| tee ADAPT_step5.out 
sed -i 's/\[\[//g' mo_beta_1rdm.txt
sed -i 's/\[\[//g' mo_alpha_1rdm.txt
sed -i 's/]]//g' mo_beta_1rdm.txt
sed -i 's/]]//g' mo_alpha_1rdm.txt
cd ..

# STEP 6 : Build new \bar{V}^B
mkdir step6
cd step6
cp ../step5/mo**_1rdm.txt .
#cp ../step3/mo_beta_1rdm.txt .
#cp ../step3/mo_beta_1rdm.txt mo_alpha_1rdm.txt
cp ../${geom} .
cp ../${basis} .
cp -rf ../step0/${system}_${basis}.ezfio .
qp set_file ${system}_${basis}.ezfio
mv ${system}_${basis}.ezfio.vbarb_FCIDUMP ${system}_${basis}.ezfio.vbarb_FCIDUMP_1
qp set sc_basis_corr basis_cor_func pbe_ueg
qp run write_dm_in_aux_quantities | tee write_dm_in_aux_quantities.out
qp set density_for_dft density_for_dft input_density
cp ../step4/pot_basis_alpha_mo.txt .
cp ../step4/pot_basis_beta_mo.txt .
qp run print_eff_basis_potential | tee print_eff_basis_potential.out
qp run basis_correction | tee basis_correction.out
qp run print_dipole_moment_from_input_density | tee print_dipole_moment_from_input_density.out
qp run fcidump_for_vbarb
qp run write_eff_basis_pot_in_file | tee write_eff_basis_pot_in_file.out
mv ${system}_${basis}.ezfio.vbarb_FCIDUMP ${system}_${basis}.ezfio.vbarb_FCIDUMP_2
cd ..

done
