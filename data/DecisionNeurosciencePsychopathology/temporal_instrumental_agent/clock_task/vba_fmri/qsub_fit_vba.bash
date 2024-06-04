#!/usr/bin/env sh

#PBS -l nodes=1:ppn=37
#PBS -l walltime=14:00:00
#PBS -l pmem=8gb
#PBS -A wff3_a_g_hc_default
#PBS -j oe
#PBS -M michael.hallquist@psu.edu
#PBS -m abe

# mnh5174_a_g_hc_default

# env
cd $PBS_O_WORKDIR

export matlab_cpus=37
module load matlab/R2017b

#datasets="mmclock_meg mmclock_fmri"
#datasets="mmclock_fmri"
datasets="mmclock_meg"
models="decay" # fixed decay_factorize decay_uniform decay_ps_equate decay_uniform_ps_equate"

factorize="separate factorize"
decay="uniform selective"
ps="psnarrow psequate"

#settings for group fixed pararams in fmri
#export sceptic_dataset="mmclock_fmri"
#export sceptic_model="decay_factorize_selective_psequate_fixedparams_fmri"

#matlab -nodisplay -r sceptic_fit_group_vba_ffx 2>&1 | tee ofiles/sceptic_fit_ffx_mmclock_fmri_${sceptic_model}.out

#settings for group fixed pararams in meg
export sceptic_dataset="mmclock_meg"
export sceptic_model="decay_factorize_selective_psequate_fixedparams_meg"

matlab -nodisplay -r sceptic_fit_group_vba_ffx 2>&1 | tee ofiles/sceptic_fit_ffx_${sceptic_dataset}_${sceptic_model}.out


# for d in $datasets; do
#     for m in $models; do
# 	for f in $factorize; do
# 	    for u in $decay; do
# 		for p in $ps; do
# 		    export sceptic_dataset=$d
# 		    export sceptic_model="${m}_${f}_${u}_${p}"

# 		    if [ ! -f "ofiles/sceptic_fit_mfx_${d}_${sceptic_model}.out" ]; then
# 			matlab -nodisplay -r sceptic_fit_group_vba_ffx 2>&1 | tee ofiles/sceptic_fit_ffx_${d}_${sceptic_model}.out
# 			matlab -nodisplay -r sceptic_fit_group_vba_mfx 2>&1 | tee ofiles/sceptic_fit_mfx_${d}_${sceptic_model}.out
# 		    else
# 			echo "Skipping run because sceptic_fit_ffx_${d}_${sceptic_model}.out exists."
# 		    fi
# 		done
# 	    done
# 	done
#     done
# done

R CMD BATCH --no-save --no-restore compile_trial_level_dataframes.R
