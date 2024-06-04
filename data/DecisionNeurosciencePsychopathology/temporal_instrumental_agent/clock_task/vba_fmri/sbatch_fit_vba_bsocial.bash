#!/usr/bin/env sh
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8G

# env
cd $SLURM_SUBMIT_DIR

export matlab_cpus=10
module use /proj/mnhallqlab/sw/modules

module load matlab/2022b
module load r/4.2.1

datasets="bsocial"

#models="fixed_psequate decay_factorize_selective_psequate bsocial_decay_factorize_selective_psequate_fixedparams"
models="decay_factorize_selective_psequate_nocensor"

#factorize="separate factorize"
#decay="uniform selective"
#ps="psnarrow psequate"

#settings for group fixed pararams in fmri
#export sceptic_dataset="mmclock_fmri"
#export sceptic_model="decay_factorize_selective_psequate_fixedparams_fmri"

#matlab -nodisplay -r sceptic_fit_group_vba_ffx 2>&1 | tee ofiles/sceptic_fit_ffx_mmclock_fmri_${sceptic_model}.out

for d in $datasets; do
    for m in $models; do
	export sceptic_dataset=$d
	export sceptic_model="${m}"

	if [ ! -f "ofiles/sceptic_fit_mfx_${d}_${sceptic_model}.out" ]; then
	    #matlab -nodisplay -r sceptic_fit_group_vba_ffx 2>&1 | tee ofiles/sceptic_fit_ffx_${d}_${sceptic_model}.out
	    matlab -nodisplay -r sceptic_fit_group_vba_mfx 2>&1 | tee ofiles/sceptic_fit_mfx_${d}_${sceptic_model}.out
	else
	    echo "Skipping run because sceptic_fit_mfx_${d}_${sceptic_model}.out exists."
	fi
    done
done

R CMD BATCH --no-save --no-restore compile_trial_level_dataframes.R
