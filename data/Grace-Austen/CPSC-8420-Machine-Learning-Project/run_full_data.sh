#!/bin/bash -i
#
#PBS -N datamanip
#PBS -l select=1:ncpus=32:mpiprocs=32:mem=500gb:ngpus=2,
#PBS -l walltime=6:00:00
#PBS -o datamanip_output.txt
#PBS -j oe

module load matlab/2022a
cd CPSC-8420-Machine-Learning-Project
matlab -r "edited_datamanip(fin=\"data/repositories.csv\", fout=\"data/full_data\");
        train_model(model='linear', pca=0, name_data_file='data/full_data_name.mat', descript_data_file='data/full_data_descript.mat', other_data_file='data/full_data_other.mat');
        train_model(model='ridge', pca=0, name_data_file='data/full_data_name.mat', descript_data_file='data/full_data_descript.mat', other_data_file='data/full_data_other.mat');
        train_model(model='lasso', pca=0, name_data_file='data/full_data_name.mat', descript_data_file='data/full_data_descript.mat', other_data_file='data/full_data_other.mat');
        exit" -logfile full_data_logs.txt
