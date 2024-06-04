#!/bin/bash
#SBATCH --job-name=eval_adv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lucia.trastulla@fht.org
#SBATCH --partition=cpuq
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/eval_%j.out.log
#SBATCH --error=logs/eval_%j.err.log
#SBATCH --mem=24G

module load nlopt/2.7.0-intel-oneapi-mkl-2021.4.0 
module load R/4.1.0
GROUP_PATH='/group/iorio/lucia/'
PATH_MODEL=${GROUP_PATH}'Multiomic_VAE/experiments/experiment_1/ae_gan/'

### all samples ###
# test 1
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx 

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 2
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx 
    
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 3
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 4
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_all_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 5
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 6
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 7
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 8
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var1000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 9
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 10
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 11
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 12
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples__ngene_var5000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

### only tcga ###
# test 1
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 2
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 3
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 4
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_all_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 5
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 6
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 7
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 8
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var1000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 9
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_False_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 10
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_False_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 11
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_True_only_shared_False/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE

# test 12
Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx

Rscript experiments/plot_umap_results_run.R \
    --folder_model ${PATH_MODEL}samples_tcgaonly_ngene_var5000_norm_feat_flag_True_only_shared_True/ \
    --depmap_meta_file ${GROUP_PATH}datasets/DEPMAP_PORTAL/version_23Q2/Model.csv \
    --file_purity ${GROUP_PATH}Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx \
    --private_enc TRUE
