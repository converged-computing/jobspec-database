#!/bin/bash -l

#PBS -l walltime=2:30:00,select=1:ncpus=4:ompthreads=12:ngpus=1:mem=96gb
#PBS -N ishika_enformer
#PBS -A st-cdeboer-1-gpu
#PBS -m abe
#PBS -M ishika.luthra@ubc.ca
#PBS -o /scratch/st-cdeboer-1/iluthra/output.txt
#PBS -e /scratch/st-cdeboer-1/iluthra/error.txt

################################################################################

#load necessasry modules for enformer conda enviroment
module load gcc python miniconda3 cuda cudnn
conda activate ishika_enformer

#change path to match where enformer predictions are saved
prediction_path="/scratch/st-cdeboer-1/iluthra/randomDNA/all_iPSC_related_H1/"
#add path to where plots should be saved
mkdir /scratch/st-cdeboer-1/iluthra/randomDNA/Analysis_outputs_06152023/H1_DNase_rep1_07052023/
output_path_dir="/scratch/st-cdeboer-1/iluthra/randomDNA/Analysis_outputs_06152023/H1_DNase_rep1_07052023/"

#path where all analysis scripts are saved
scripts_path="/project/st-cdeboer-1/iluthra/enformer/test_enformer/de-former/Paper_Review_Analysis/Final_clean_scripts_figures/"

python "$scripts_path""nucleotide_shuffling_ecdfs.py" --path_to_predictions "$prediction_path"  --cell_type "H1-hESC" --cell_type_index 40 --output_path "$output_path_dir""/ecdfs_shuffle/"

python "$scripts_path""nucleotide_switch_ecdfs.py" --path_to_predictions "$prediction_path"  --cell_type "H1-hESC" --cell_type_index 40 --output_path "$output_path_dir""/ecdfs_switch/"

python "$scripts_path""plot_enformer_tracks.py" --path_to_predictions "$prediction_path" --track_to_plot "H1-hESC DNase" --track_index 40 --output_path "$output_path_dir""/H1-DNase_tracks/"

python "$scripts_path""CAGE_tracks_analysis_final.py" --path_to_predictions "/scratch/st-cdeboer-1/iluthra/randomDNA/CAGE/" --output_path "$output_path_dir""/CAGE_analysis/"

python "$scripts_path""germlayer_scatterplots.py" --path_to_predictions "$prediction_path" --track_indices 5 15 25 --output_path "$output_path_dir""/meso_endo_ecto/"

#track indicies for cell type of interest across 5 chromatin marks used
python "$scripts_path""encode_enformer_bed_files.py" --path_to_predictions "$prediction_path" --cell_type "H1-hESC" --track_indices 40 41 42 43 44 --encode_file_paths "//project/st-cdeboer-1/iluthra//enformer_random_DNA/enformer_data/ENCODE/" --output_path "$output_path_dir""/Evolution_plots/"

#need to run bedtools here 5 tracks of interest
conda deactivate
module load gcc/5.5.0
module load bedtools2/2.30.0
cd "$output_path_dir""/Evolution_plots/"

bedtools intersect -wao -a genomic_regions_coordinates_n1000_114kb_128bins_H1-hESC_Dnase.bed -b H1-hESC_Dnaseencode.bed > DNAse_enformer_encode_overlap.bed
bedtools intersect -wao -a genomic_regions_coordinates_n1000_114kb_128bins_H1-hESC_H3K27ac.bed -b H1-hESC_H3K27acencode.bed > H3K27ac_enformer_encode_overlap.bed
bedtools intersect -wao -a genomic_regions_coordinates_n1000_114kb_128bins_H1-hESC_H3K4me1.bed -b H1-hESC_H3K4me1encode.bed > H3K4me1_enformer_encode_overlap.bed
bedtools intersect -wao -a genomic_regions_coordinates_n1000_114kb_128bins_H1-hESC_H3K4me3.bed -b H1-hESC_H3K4me3encode.bed > H3K4me3_enformer_encode_overlap.bed
bedtools intersect -wao -a genomic_regions_coordinates_n1000_114kb_128bins_H1-hESC_H3K27me3.bed -b H1-hESC_H3K27me3encode.bed > H3K27me3_enformer_encode_overlap.bed

conda deactivate
module load gcc python miniconda3 cuda cudnn
conda activate ishika_enformer

python "$scripts_path""Evolution_plots_final.py" --path_to_predictions "$prediction_path" --cell_type "H1-hESC" --track_indices 40 41 42 43 44 --output_path "$output_path_dir""/Evolution_plots/"

python "$scripts_path""Cooccurance_marks_analysis_final.py" --path_to_predictions "$prediction_path" --cell_type "H1-hESC" --track_indices 40 41 42 43 44 --output_path "$output_path_dir""/Cooccurance_marks_analysis/"

#this script goes last since its takes the longest to run
python "$scripts_path""TSNE_final.py" --path_to_predictions "$prediction_path" --cell_type "H1-hESC" --track_indices 40 41 42 43 44 --output_path "$output_path_dir""/TSNE/"
