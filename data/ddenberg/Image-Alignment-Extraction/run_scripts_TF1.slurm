#!/bin/bash

#SBATCH --job-name=TF1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:59:00
#SBATCH -A molbio

module purge
module load matlab/R2023a

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog_maxproj.h5', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog', './output/231231_st7/Ch0long_nanog_centers', [0:160], 16, [891, 948, 54])"

matlab -nodisplay -nosplash -r  "align_long_short('/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog_centers', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6_centers', '/scratch/gpfs/ddenberg/231231/231231_st7/LS_align', [80:150], 16)"

## matlab -nodisplay -nosplash -r "align_histone_fusedLS('/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog', '/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog_centers', '/scratch/gpfs/ddenberg/230101_st8/Ch0short_gata6', '/scratch/gpfs/ddenberg/230101_st8/Ch0short_gata6_centers', '/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone', '/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone_centers', '/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog_Ch0short_gata6_align/tform_xy.mat', '/scratch/gpfs/ddenberg/230101_st8/Ch0LS_Ch1long_align', [0:104], 16)"

## matlab -nodisplay -nosplash -r "align_histone_TF('/scratch/gpfs/ddenberg/231214_stack9/cdx2', '/scratch/gpfs/ddenberg/231214_stack9/histone', './output/231214_stack9/cdx2_centers', './output/231214_stack9/histone_centers', './output/231214_stack9/align_cdx2_histone', [47,53,62], 16)"

## matlab -nodisplay -nosplash -r "extract_TF('/scratch/gpfs/ddenberg/230917_st10/Ch1long_sox2', '/scratch/gpfs/ddenberg/230917_st10/segmentation', './output/230917_st10/align_Ch1long_sox2_histone', './output/230917_st10/extraction', 'extract_sox2.csv', [0:120], 16)"

## matlab -nodisplay -nosplash -r  "extract_long_short('/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog', '/scratch/gpfs/ddenberg/230101_st8/Ch0short_gata6', '/scratch/gpfs/ddenberg/230101_st8/segmentation', '/scratch/gpfs/ddenberg/230101_st8/Ch0LS_Ch1long_align', '/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog_Ch0short_gata6_align/tform_xy.mat', '/scratch/gpfs/ddenberg/230101_st8/extraction', [0:104], 16)"

####################

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog_maxproj.h5', '/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog', '/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog_centers', [0:104], 16)"

## matlab -nodisplay -nosplash -r "align_histone_TF('/scratch/gpfs/ddenberg/Yonit/231214_stack0/cdx2', '/scratch/gpfs/ddenberg/Yonit/231214_stack0/histone', './output/231214_stack0/cdx2_centers', './output/231214_stack0/histone_centers', './output/231214_stack0/align_cdx2_histone', [0:180], 16)"
