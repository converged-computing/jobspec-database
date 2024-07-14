#!/usr/bin/env bash
#SBATCH --account=mwinslow
#SBATCH -t 24:00:00
#SBATCH --mem=8000
#SBATCH --requeue
#SBATCH --mail-user=eshuldin
#SBATCH --mail-type=FAIL


#######################
# To run, modify --array flag to match # of samples in project 
# Takes 2 arguments, PROJECT and ROOT
# PROJECT is whatever project id I have assigned to the project. Must match ID in all other files pertaining to project
# ROOT is /scratch/groups/dpetrov/emilys
# There must be a file in the same directory PROJECT.inp containing parameters for job array
#sh run_tumor_processing.sh UCSF_Injury 2 /labs/mwinslow/Emily/
#sbatch run_tumor_processing.sh UCSF_Injury_corr3 3 /labs/mwinslow/Emily/ 32
#python3 process_tumors.py --project=UCSF_Injury_corr3 --parameter=3 --root=/labs/mwinslow/Emily/
#python3 tumor_burden_calc.py --project=RagKO_1 --parameter=2 --root=/labs/mwinslow/Emily/
#python3 GSTR_calc_exploratory.py --project=RagKO_1_sgids_corr --parameter=3 --root=/labs/mwinslow/Emily/
#python3 TN_calc_adj_within_mice.py --project=UCSF_Injury_corr2 --parameter=2 --root=/labs/mwinslow/Emily/
#sh tubaseq.sh UCSF_Injury_corr3 3 /labs/mwinslow/Emily/ 32
# sh run_tumor_processing.sh RIT1_noKTC3 1 /labs/mwinslow/Emily/ 29
# sh run_tumor_processing.sh RagKO_1_sgids_corr 2 /labs/mwinslow/Emily/ 58
#sh run_tumor_processing.sh Laura_SpikeIns 0 /labs/mwinslow/Emily/ 8
#sbatch run_tumor_processing.sh KRAS_IP 2 /labs/mwinslow/Emily/ 22
#sbatch run_tumor_processing.sh KRAS_IP_BIGPOOL 2 /labs/mwinslow/Emily/ 24
#sbatch run_tumor_processing.sh KRAS_ESSENTIAL 3 /labs/mwinslow/Emily/ 19
#sbatch run_tumor_processing.sh KRASIP_MultiGEMM2 3 /labs/mwinslow/Emily/ 35
#sbatch run_tumor_processing.sh KRASIP_MultiGEMM2 5 /labs/mwinslow/Emily/ 35
#sbatch run_tumor_processing.sh Fasting_1_trimmed 1 /labs/mwinslow/Emily/ 21
#sbatch run_tumor_processing.sh KRAS_IP_KPTC_Mets_lungs 2 /labs/mwinslow/Emily/ 10
#sh run_tumor_processing.sh Inert-sgRNA-KPTC_nano 0 /labs/mwinslow/Emily/ 7
#sbatch run_tumor_processing.sh Jackie1 0 /labs/mwinslow/Emily/ 13
#sbatch run_tumor_processing.sh RagKO_1_sgids_corr 3 /labs/mwinslow/Emily/ 58

#sbatch run_tumor_processing.sh KRASIP_MultiGEMM2 2 /labs/mwinslow/Emily/ 35
#sbatch run_tumor_processing.sh NINJAScreen1_part1 2c /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh NINJAScreen1_part1_with_undetermined 0 /labs/mwinslow/Emily/ 39
#sh run_tumor_processing.sh AgingScreen1_part1_with_undetermined_nano 0 /labs/mwinslow/Emily/ 39

#sh run_tumor_processing.sh KRASIP_MultiGEMM2 2c /labs/mwinslow/Emily/ 35
#sh run_tumor_processing.sh AgingScreen1_part1_with_undetermined 2 /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh AgingScreen1_part1 1c /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh AgingScreen1_part1 2c /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh AgingScreen1_part1 3c /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh AgingScreen1_part1 4c /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh AgingScreen1_part1 5c /labs/mwinslow/Emily/ 38

#sbatch run_tumor_processing.sh AgingScreen1_part2 3c /labs/mwinslow/Emily/ 38
#sbatch run_tumor_processing.sh AgingScreen1 2c /labs/mwinslow/Emily/ 76

ml python/3.6.4
module load miniconda/3

#python3 process_tumors.py "--project=${1}" "--parameter=${2}" "--root=${3}"
#python3 bundle_tumors.py "--project=${1}" "--parameter=${2}" "--root=${3}"
#python3 remove_contamination.py "--project=${1}" "--parameter=${2}" "--root=${3}"

##Calculations for aging data, both parts together
#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470,MT1025,MT1071,MT1759,MT1779,MT498_PM --method=merge
#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470,MT1025,MT1071,MT1759,MT1779,MT498_PM --method=merge
#python3 tumor_burden_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470,MT1025,MT1071,MT1759,MT1779,MT498_PM
python3 GSTR_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,,MT1715,MT469_MT1786,MT470,MT1025,MT1071,MT1759,MT1779,MT498_PM




###Percentile calculation for aging data, part 1
#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver
#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver --method=median
#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470 --method=median

#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver --method=merge
#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470 --method=merge
#python3 percentile_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1025,MT1071,MT1759,MT1779,MT498_PM --method=merge

#python3 percentile_calc_per_mouse.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1025,MT1071,MT1759,MT1779,MT498_PM

#python3 percentile_calc_within_mice.py "--project=${1}" "--parameter=${2}" "--root=${3}"

#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}"

#TN and TB calculations for Aging data, part 1
#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver --method=median
#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470 --method=median

#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver --method=merge
#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470 --method=merge

#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1025,MT1071,MT1759,MT1779,MT498_PM --method=merge

#python3 tumor_burden_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1025,MT1071,MT1759,MT1779,MT498_PM

#python3 tumor_burden_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver
#python3 tumor_burden_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1715,MT469_MT1786,MT470

#python3 tumor_burden_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver
#python3 TN_calc_per_mouse.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver,MT1025,MT1071,MT1759,MT1779,MT498_PM

###GSTR for Aging data, part 1
#python3 GSTR_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --sgids_to_exclude=Pcna_2,Pten_2,Smad4_2,Rbm10_4,Rnf43_1,Rb1_1,Smad4_1,Apc_2,Kmt2c_1,Cdkn2c_3,Cdkn2a_1,Ifngr1_2,Setd2_V4,Smarca4_2,B2M_1,Ifngr1_1 --samples_to_exclude=MT475_Liver,MT475_Spleen,MT442_Liver


#making TN_calc flexible so that you can normalize to a genotype that's not cas9-negative
#python3 TN_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}" --control_genotype=KCN

#python3 TN_calc_adj_within_mice.py "--project=${1}" "--parameter=${2}" "--root=${3}"


#python3 tumor_burden_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}"
#python3 tumor_burden_calc_within_mice.py "--project=${1}" "--parameter=${2}" "--root=${3}"
# #python3 tumor_burden_relative_to_donor.py "--project=${1}" "--parameter=${2}" "--root=${3}"
#python3 GSTR_calc_exploratory.py "--project=${1}" "--parameter=${2}" "--root=${3}"
#python3 GSTR_calc.py "--project=${1}" "--parameter=${2}" "--root=${3}"