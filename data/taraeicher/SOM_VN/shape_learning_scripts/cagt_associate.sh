#PBS -l nodes=1:ppn=28
#PBS -l walltime=48:00:00 
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
    CELL_LINE=""
    REGION_SIZE=4000
    BASE_FILENAME=""
    BAM=""
    CHROMHMM=""
    CHROMS_NUM="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
    WIG_SPLIT_PATH=""
    BIN_SIZE=50
    while getopts n:d:c:p: option; do
        case "${option}" in
            n) CELL_LINE=$OPTARG;;
            d) BASE_FILENAME=$(realpath $OPTARG);;
            c) CHROMHMM=$(realpath $OPTARG);;
            p) CAGT_PATH=$(realpath $OPTARG);;
        esac
    done
    BASE_PATH=$BASE_FILENAME/$CELL_LINE
    PYTHON_VERSION=2.7	
    
#Print message to user.
    echo "Annotating with the following settings:"
    echo "Name is $CELL_LINE"
    echo "Base directory is $BASE_FILENAME"
    echo "ChromHMM file is $CHROMHMM"
    echo "Path is $CAGT_PATH"
    echo -e "-------------------------------------------------------------------------------------------------------------------------\n"
	
#Create all needed directories.
    SHAPES_COMPREHENSIVE="$BASE_PATH/shapes_cagt"
    CAGT_OUT="$CAGT_PATH/cagt/matlab/src"
    TRAINING_ANNOTATION_FILES="$BASE_PATH/training_anno"
    if [[ ! -e $TRAINING_ANNOTATION_FILES ]]; then
        mkdir $TRAINING_ANNOTATION_FILES
    fi
    TO_ANNOTATE="$BASE_PATH/annotation_files"
    if [[ ! -e $TO_ANNOTATE ]]; then
        mkdir $TO_ANNOTATE
    fi
    WIG="$BASE_PATH/wig_chroms"
    if [[ ! -e  $WIG ]]; then
        mkdir $WIG
    fi
    SHAPE_ANNOTATED="$BASE_PATH/anno_beds_cagt"
    if [[ ! -e $SHAPE_ANNOTATED ]]; then
        mkdir $SHAPE_ANNOTATED
    fi
    ANNOTATED="$BASE_PATH/annotated_cagt"
    if [[ ! -e $ANNOTATED ]]; then
        mkdir $ANNOTATED
    fi
    ANNOTATED_SORTED="$BASE_PATH/annotated_sorted_cagt"
    if [[ ! -e $ANNOTATED_SORTED ]]; then
        mkdir $ANNOTATED_SORTED
    fi
    ANNOTATED_CONSOLIDATED="$BASE_PATH/annotated_consolidated_cagt"
    if [[ ! -e $ANNOTATED_CONSOLIDATED ]]; then
        mkdir $ANNOTATED_CONSOLIDATED
    fi
    SHAPE_ANNOTATED_SORTED="$BASE_PATH/anno_beds_sorted_cagt"
    if [[ ! -e $SHAPE_ANNOTATED_SORTED ]]; then
        mkdir $SHAPE_ANNOTATED_SORTED
    fi
    SHAPE_ANNOTATED_FINAL="$BASE_PATH/anno_beds_final_cagt"
    if [[ ! -e $SHAPE_ANNOTATED_FINAL ]]; then
        mkdir $SHAPE_ANNOTATED_FINAL
    fi
    CHROMHMM_INTERSECTS="$BASE_PATH/anno_intersects_cagt"
    if [[ ! -e $CHROMHMM_INTERSECTS ]]; then
        mkdir $CHROMHMM_INTERSECTS
    fi
    if [[ ! -e ${SHAPES_COMPREHENSIVE}_${CELL_LINE} ]]; then
        mkdir ${SHAPES_COMPREHENSIVE}_${CELL_LINE}
    fi
    ANNO_MERGED="$BASE_PATH/annotated_merged_cagt"
    if [[ ! -e $ANNO_MERGED ]]; then
        mkdir $ANNO_MERGED
    fi
    SHAPES="$BASE_PATH/shapes_cagt_merged"
    SHAPES_LOG="$BASE_PATH/shapes_log_cagt"
	
# Method for running the pipeline for a chromocagte.
	run_pipeline() {
		local c=$1
		
		#Annotate regions with shape.
		python make_shape_bed.py $TRAINING_ANNOTATION_FILES/chrom${c}window3 $CAGT_OUT/ctcf_centroids_${c}.csv $SHAPE_ANNOTATED/anno$c 0
        
        bedtools sort -i  $SHAPE_ANNOTATED/anno${c} > $SHAPE_ANNOTATED_SORTED/anno${c}
        python consolidate.py $SHAPE_ANNOTATED_SORTED/anno${c} $SHAPE_ANNOTATED_FINAL/anno${c}
        cut -d$'\t' -f 1,2,3,4,5 $SHAPE_ANNOTATED_FINAL/anno${c} > $SHAPE_ANNOTATED_FINAL/anno${c}.bed
        awk '{ print $6}' $SHAPE_ANNOTATED_FINAL/anno${c} > $SHAPE_ANNOTATED_FINAL/clusters_anno${c}
        cut -d$'\t' -f 7,8,9,10 $SHAPE_ANNOTATED_FINAL/anno${c} > $SHAPE_ANNOTATED_FINAL/scores_anno${c}.bed
        echo -e "\n------------------------------------Consolidating complete for chrom $c.-----------------------------\n"
        
        #Save shapes to file.
        bedtools intersect -wao -a $SHAPE_ANNOTATED_FINAL/anno${c}.bed -b $CHROMHMM > $CHROMHMM_INTERSECTS/anno${c}.bed			
        bedtools sort -i $CHROMHMM_INTERSECTS/anno${c}.bed > $CHROMHMM_INTERSECTS/anno${c}_sorted.bed
        python consolidate_chromHMM.py $CHROMHMM_INTERSECTS/anno${c}_sorted.bed $CAGT_OUT/ctcf_centroids_${c}.csv $SHAPES_COMPREHENSIVE ${WIG}/${CELL_LINE}.chr${c}.wig ${c} $CELL_LINE $TRAINING_ANNOTATION_FILES/chrom${c}window3 0
        
	}
    
    #Run the pipeline from split WIG files to final set of annotations.
    pids=""
    for f in $CHROMS_NUM;
        do 
            run_pipeline $f & 
            pids="$pids $!"
        done
        
    #Merge shapes entries across chromocagtes.
    wait $pids
    python ../common_scripts/merge_significant.py $SHAPES_COMPREHENSIVE $SHAPES $SHAPES_LOG
    
    #Annotating the regions.
    run_anno_pipeline(){
        local c=$1
        python ../annotation_scripts/make_annotated_bed.py $TO_ANNOTATE/chrom${c}window3 $SHAPES $ANNOTATED/anno${c} $WIG/${CELL_LINE}.chr${c}.wig 0.0
        echo -e "------------------------------------------------Annotation complete for chrom $c.------------------------------------------------\n"
        
        # Sorting the annotated regions.
        bedtools sort -i  $ANNOTATED/anno${c} > $ANNOTATED_SORTED/anno${c}
        bedtools sort -i  $ANNOTATED/anno${c}clust > $ANNOTATED_SORTED/anno${c}.clust

        # Consolidating the sorted annotated regions.
		python ../common_scripts/consolidate_bed.py $ANNOTATED_SORTED/anno${c} $ANNOTATED_CONSOLIDATED/anno${c}
		echo -e "------------------------------------------------Consolidation complete for chrom $c.----------------------------------------------\n"
		
        # Creating BED, cluster, and score files.
        cut -d$'\t' -f 1,2,3,4,5 $ANNOTATED_CONSOLIDATED/anno${c} > $ANNOTATED_CONSOLIDATED/anno${c}.bed
        cut -d$'\t' -f 1,2,3,4,5 $ANNOTATED_CONSOLIDATED/anno${c}.clust > $ANNOTATED_CONSOLIDATED/anno${c}clust.bed
        awk '{ print $6}' $ANNOTATED_CONSOLIDATED/anno${c} > $ANNOTATED_CONSOLIDATED/clusters_anno${c}
        cut -d$'\t' -f 7,8,9,10 $ANNOTATED_CONSOLIDATED/anno${c} > $ANNOTATED_CONSOLIDATED/scores_anno${c}.bed
		echo -e "------------------------------------Consolidating per window size complete for chrom $c.-----------------------------------------\n"
        
        bedtools intersect -wao -a $ANNOTATED_CONSOLIDATED/anno${c}.bed -b $CHROMHMM > $ANNO_MERGED/anno${c}.bed
    }
    
    #Run the pipeline from split WIG files to final set of annotations.
    pids=""
    for f in $CHROMS_NUM;
        do 
            run_anno_pipeline $f & 
            pids="$pids $!"
        done
    wait $pids
        
    #Plot precision and recall.    
    python ../meta_analysis_scripts/plot_precision_recall_nobaselines.py $ANNO_MERGED/ $ANNOTATED_CONSOLIDATED/ $BASE_PATH/cagt  $PRECISION_RECALL/ ${WIG}/${CELL_LINE}.chr $CELL_LINE $CELL_LINE
        
    #Exit
	wait
	echo Done!
	exit 0