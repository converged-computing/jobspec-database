#PBS -l nodes=1:ppn=28
#PBS -l walltime=10:00:00 
#!/bin/bash   
# """
# Dependencies:
# 1. Following python modules for gosr: argparse, logging, sys, itertools, numpy, pysam
# 2. Tensorflow, which can be installed here: https://www.tensorflow.org/install/install_linux
# 3. Samtools: https://sourceforge.net/projects/samtools/files/
# 4. Wig-split: https://github.com/taoliu/taolib/blob/master/Scripts/wig_split.py
# 5. Taolib directory: https://github.com/taoliu/taolib
# 6. Gap statistic code: https://github.com/minddrummer/gap
# 7. Added to your path: gosr/bin, taolib, wig_split, gap
# 8. Added to your python path: pythonmodules/gosr/lib
# 9. Make sure you are not using python3, which causes an error with the print statements in gosr and in wig_split
# 10. Make sure your version of python is compiled to use UCS2

#Variables
cd /fs/project/PAS0272/Tara/DNase_SOM/scripts/shape_learning_scripts
    CELL_LINE="GM12878"
    REGION_SIZE=4000
    BASE_FILENAME="../.."
    BAM=""
    CHROMHMM="../../chromHmm/GM_colored.bed"
    CHROMS_NUM="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
    WIG_SPLIT_PATH=""
    BIN_SIZE=50
    SHAPES_COMPREHENSIVE="../../GM12878/shapes_nopromoter"
    while getopts n:d:c:s:r:i: option; do
       case "${option}" in
           n) CELL_LINE=$OPTARG;;
           d) BASE_FILENAME=$(realpath $OPTARG);;
           c) CHROMHMM=$(realpath $OPTARG);;
           s) SHAPES_COMPREHENSIVE=$(realpath $OPTARG);;
           r) REGION_SIZE=$OPTARG;;
           i) BIN_SIZE=$OPTARG;;
       esac
    done
    BASE_PATH=$BASE_FILENAME/$CELL_LINE
    PYTHON_VERSION=2.7	
    module load python/$PYTHON_VERSION
    
#Print message to user.
    echo "Annotating with the following settings:"
    echo "Name is $CELL_LINE"
    echo "Base directory is $BASE_FILENAME"
    echo "ChromHMM file is $CHROMHMM"
    echo "Shapes will be saved in $SHAPES_COMPREHENSIVE"
    echo -e "-------------------------------------------------------------------------------------------------------------------------\n"
	
#Create all needed directories.
    CHROMHMM_INTERSECTS="$BASE_PATH/anno_intersects"
    if [[ ! -e $CHROMHMM_INTERSECTS ]]; then
        mkdir $CHROMHMM_INTERSECTS
    fi
    SOM_OUT="$BASE_PATH/som_output_nopromoter"
    if [[ ! -e $SOM_OUT ]]; then
        mkdir $SOM_OUT
    fi
    SOM_OUT_FILTERED="$BASE_PATH/som_output_filtered_nopromoter"
    if [[ ! -e $SOM_OUT_FILTERED ]]; then
        mkdir $SOM_OUT_FILTERED
    fi
    SOM_OUT_SHIFTED="$BASE_PATH/som_output_shifted_nopromoter"
    if [[ ! -e $SOM_OUT_SHIFTED ]]; then
        mkdir $SOM_OUT_SHIFTED
    fi
    SOM_OUT_FINAL="$BASE_PATH/som_output_final_nopromoter"
    if [[ ! -e $SOM_OUT_FINAL ]]; then
        mkdir $SOM_OUT_FINAL
    fi
    SHAPE_ANNOTATED="$BASE_PATH/anno_beds_nopromoter"
    if [[ ! -e $SHAPE_ANNOTATED ]]; then
        mkdir $SHAPE_ANNOTATED
    fi
    SHAPE_ANNOTATED_SORTED="$BASE_PATH/anno_beds_sorted_nopromoter"
    if [[ ! -e $SHAPE_ANNOTATED_SORTED ]]; then
        mkdir $SHAPE_ANNOTATED_SORTED
    fi
    SHAPE_ANNOTATED_FINAL="$BASE_PATH/anno_beds_final_nopromoter"
    if [[ ! -e $SHAPE_ANNOTATED_FINAL ]]; then
        mkdir $SHAPE_ANNOTATED_FINAL
    fi
    if [[ ! -e "${SHAPES_COMPREHENSIVE}_${CELL_LINE}" ]]; then
        mkdir "${SHAPES_COMPREHENSIVE}_${CELL_LINE}"
    fi
    SHAPES="$BASE_PATH/shapes_nopromoter_merged"
    SHAPES_LOG="$BASE_PATH/shapes_log_nopromoter"
    #SOM_OUT_FINAL="$BASE_PATH/som_output_final"
    WIG="$BASE_PATH/wig_chroms"
    TRAINING_FILES="$BASE_PATH/training_shifted"
    TRAINING_ANNOTATION_FILES="$BASE_PATH/training_anno"
	
# Method for running the pipeline for a chromosome.
	run_pipeline() {
		local c=$1
        
        #Remove promoters from ChromHMM.
        awk '{ if ( $4 != "AP" && $4 != "OP" && $4 != "GE" && $4 != "TS") { print; } }' $CHROMHMM > ${CHROMHMM}_nopromoter.bed
        awk '{ if ( $4 == "AP" || $4 == "OP" || $4 == "GE" || $4 == "TS") { print; } }' $CHROMHMM > ${CHROMHMM}_onlypromoter.bed
        
         #Retain only training regions contained within an annotated ChromHMM region.
         awk -F ',' '{printf ("chr%s\t%s\t%s\t%s\n", $1,$2,$3,$0)}' $TRAINING_FILES/chrom$c > $TRAINING_FILES/chrom${c}_nopromoter.bed
         bedtools intersect -a $TRAINING_FILES/chrom${c}_nopromoter.bed -b ${CHROMHMM}_onlypromoter.bed -v > $TRAINING_FILES/chrom${c}_nopromoter_intersect.bed
         awk '{ print $4 }' $TRAINING_FILES/chrom${c}_nopromoter_intersect.bed > $TRAINING_FILES/chrom${c}_nopromoter_final
        
         #Retain only regions to annotate contained within an annotated ChromHMM region.
         awk -F ',' '{printf ("chr%s\t%s\t%s\t%s\n", $1,$2,$3,$0)}' $TRAINING_ANNOTATION_FILES/chrom${c} > $TRAINING_ANNOTATION_FILES/chrom${c}_nopromoter.bed
         bedtools intersect -a $TRAINING_ANNOTATION_FILES/chrom${c}_nopromoter.bed -b ${CHROMHMM}_onlypromoter.bed -v > $TRAINING_ANNOTATION_FILES/chrom${c}_nopromoter_intersect.bed
         awk '{ print $4 }' $TRAINING_ANNOTATION_FILES/chrom${c}_nopromoter_intersect.bed > $TRAINING_ANNOTATION_FILES/chrom${c}_nopromoter_final
        
         #Run the SOM.
		 python som_vn.py $TRAINING_FILES/chrom${c}_nopromoter_final $SOM_OUT/chrom$c $WIG/$CELL_LINE.chr$c.wig $REGION_SIZE $BIN_SIZE 0 False
		 echo -e "---------------------------------------------SOM model is ready for chrom $c.-----------------------------------------\n"
		
		 #Remove all shapes to which no regions map.
		 python remove_by_cutoff.py $SOM_OUT/chrom${c}som_centroid 1 $SOM_OUT_FILTERED/chrom${c}som_centroid
		 echo -e "------------------------------------------------Removal complete for chrom $c.---------------------------------------\n"
		
		 #Merge shifted regions.
        python merge_shifted.py $SOM_OUT_FILTERED/chrom${c}som_centroid $SOM_OUT_SHIFTED/chrom${c}som_centroid 0
		echo -e "------------------------------------------------Merging complete for chrom $c.----------------------------------------\n"
		
		#Remove duplicate shapes using kmeans.
		python kmeans_shapes.py $SOM_OUT_SHIFTED/chrom${c}som_centroid $SOM_OUT_FINAL/chrom${c}som_centroid
		echo -e "-------------------------------------------------K-means complete for chrom $c.---------------------------------------\n"
		
		#Annotate regions with shape.
		python make_shape_bed.py $TRAINING_ANNOTATION_FILES/chrom${c} $SOM_OUT_FINAL/chrom${c}som_centroid $SHAPE_ANNOTATED/anno$c 0
        echo -e "\n------------------------------------Initial annotations complete for chrom $c.-----------------------------\n"
        
        bedtools sort -i  $SHAPE_ANNOTATED/anno${c} > $SHAPE_ANNOTATED_SORTED/anno${c}
        python consolidate.py $SHAPE_ANNOTATED_SORTED/anno${c} $SHAPE_ANNOTATED_FINAL/anno${c}
        cut -d$'\t' -f 1,2,3,4,5 $SHAPE_ANNOTATED_FINAL/anno${c} > $SHAPE_ANNOTATED_FINAL/anno${c}.bed
        awk '{ print $6}' $SHAPE_ANNOTATED_FINAL/anno${c} > $SHAPE_ANNOTATED_FINAL/clusters_anno${c}
        cut -d$'\t' -f 7,8,9,10 $SHAPE_ANNOTATED_FINAL/anno${c} > $SHAPE_ANNOTATED_FINAL/scores_anno${c}.bed
        echo -e "\n------------------------------------Consolidating complete for chrom $c.-----------------------------\n"
        
        #Save shapes to file.
        bedtools intersect -wao -a $SHAPE_ANNOTATED_FINAL/anno${c}.bed -b ${CHROMHMM}_nopromoter.bed > $CHROMHMM_INTERSECTS/anno${c}_nopromoter.bed	
        bedtools sort -i $CHROMHMM_INTERSECTS/anno${c}_nopromoter.bed > $CHROMHMM_INTERSECTS/anno${c}_sorted_nopromoter.bed
        python consolidate_chromHMM_peas_ground_truth.py $CHROMHMM_INTERSECTS/anno${c}_sorted_nopromoter.bed $SOM_OUT_FINAL/chrom${c}som_centroid $SHAPES_COMPREHENSIVE ${WIG}/${CELL_LINE}.chr${c}.wig ${c} $CELL_LINE $TRAINING_ANNOTATION_FILES/chrom${c} 0
        echo -e "\n------------------------------------Generating of full shape list complete for chrom $c.---------------------\n"
	}
    
    #Run the pipeline from split WIG files to final set of annotations.
    pids=""
    for f in $CHROMS_NUM;
        do 
            run_pipeline $f & 
            pids="$pids $!"
        done
        
    #Merge shapes entries across chromosomes.
    wait $pids
    python ../common_scripts/merge_significant.py ${SHAPES_COMPREHENSIVE}_05 ${SHAPES}_05 ${SHAPES_LOG}_05
        
    #Exit
	wait
	echo Done!
	exit 0