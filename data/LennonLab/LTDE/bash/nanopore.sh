#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=8,vmem=100gb,walltime=140:00:00
#PBS -M wrshoema@iu.edu
#PBS -m abe
#PBS -j oe

module load fastqc
module unload python
module load python/3.6.1
# base-calling with Guppy

# FLO-MIN106
# compatible flowcells
#Pore version: R9.4.1 FLO-MIN106
#Barcoding kit ID: 1D Native barcoding genomic DNA, with EXP-NBD104 barcode
#Library ligation kit ID: SQK-LSK109
# dna_r9.4.1_450bps
#<strand type>_<pore version>_<speed>[custom tags].cfg

# --calib_detect
# -k name of kit
# -f name of flowcell

cfg=/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/data/dna_r9.4.1_450bps.cfg

fast5_dir_run2=/N/dc2/projects/muri2/Task2/LTDE/GSF2099/GSF2099-Lennon-run2-20190326/DNA-NativeBarcode/GSF2099-set2/20190326_2329_MN22062_FAK62803_d8b2ba25/fast5_pass
out_dir_run2=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled/run2
out_dir_fast5_run2=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_fast5/run2

fast5_dir_run3=/N/dc2/projects/muri2/Task2/LTDE/GSF2099/GSF2099-Lennon-run3-20190408/GSF2099-set3/20190408_2058_MN22062_FAK70247_d34ce651/fast5_pass
out_dir_run3=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled/run3
out_dir_fast5_run3=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_fast5/run3

fast5_dir_run4=/N/dc2/projects/muri2/Task2/LTDE/GSF2099-set4/1/20190422_1815_MN22062_FAK67864_08d7efdf/fast5_pass
out_dir_run4=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled/run4
out_dir_fast5_run4=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_fast5/run4

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_basecaller \
#    -i $fast5_dir_run2 \
#    -t 12 \
#    -s $out_dir_run2 \
#    -k SQK-LSK109 \
#    -f FLO-MIN106 \
#    --config $cfg \
#    --fast5_out

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_basecaller \
#    -i $fast5_dir_run3 \
#    -t 12 \
#    -s $out_dir_run3 \
#    -k SQK-LSK109 \
#    -f FLO-MIN106 \
#    --config $cfg

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_basecaller \
#    -i $fast5_dir_run4 \
#    -t 12 \
#    -s $out_dir_run4 \
#    -k SQK-LSK109 \
#    -f FLO-MIN106 \
#    --config $cfg


# get fast5 files for SRA submission
#nanopore_basecalled_fast5

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_basecaller \
#    -i $fast5_dir_run2 \
#    -t 12 \
#    -s $out_dir_fast5_run2 \
#    -k SQK-LSK109 \
#    -f FLO-MIN106 \
#    --config $cfg \
#    --fast5_out

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_basecaller \
#    -i $fast5_dir_run3 \
#    -t 12 \
#    -s $out_dir_fast5_run3 \
#    -k SQK-LSK109 \
#    -f FLO-MIN106 \
#    --config $cfg \
#    --fast5_out

/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_basecaller \
    -i $fast5_dir_run4 \
    -t 12 \
    -s $out_dir_fast5_run4 \
    -k SQK-LSK109 \
    -f FLO-MIN106 \
    --config $cfg \
    --fast5_out




# --enable_trimming is on by default, so we don't have to include it

#Barcoding/demultiplexing within specific kits
out_dir_bc_run2=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_bc/run2
out_dir_bc_run3=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_bc/run3
out_dir_bc_run4=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_bc/run4

bc_cfg=/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/data/barcoding/configuration.cfg
#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_barcoder \
#    --input_path $out_dir_run2 \
#    --save_path $out_dir_bc_run2 \
#    --config $bc_cfg

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_barcoder \
#    --input_path $out_dir_run3 \
#    --save_path $out_dir_bc_run3 \
#    --config $bc_cfg

#/N/dc2/projects/muri2/Task2/LTDE/ont-guppy-cpu/bin/guppy_barcoder \
#    --input_path $out_dir_run4 \
#    --save_path $out_dir_bc_run4 \
#    --config $bc_cfg



# de-barcode with fast5



BARCODE_DIR2=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_pcbc/run2
BARCODE_DIR3=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_pcbc/run3
BARCODE_DIR4=/N/dc2/projects/muri2/Task2/LTDE/data/nanopore_basecalled_pcbc/run4



# merge files
#data_path=/N/dc2/projects/muri2/Task2/LTDE/data
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode01/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0710.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode02/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0711.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode03/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0712.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode04/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0713.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode05/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0714.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode06/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0715.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode07/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0721.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run2/barcode08/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0722.fastq"

#cat "${data_path}/nanopore_basecalled_bc/run3/barcode01/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/ATCC13985.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode02/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/ATCC43928.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode03/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0701.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode04/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0702.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode05/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0703.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode06/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0705.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode07/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0707.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run3/barcode08/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0724.fastq"

#cat "${data_path}/nanopore_basecalled_bc/run4/barcode01/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0725.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run4/barcode02/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0727.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run4/barcode03/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0801.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run4/barcode04/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0802.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run4/barcode05/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0812.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run4/barcode06/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/KBS0706.fastq"
#cat "${data_path}/nanopore_basecalled_bc/run4/barcode07/fastq_runid_"*".fastq" > "${data_path}/nanopore_basecalled_bc_merged/PHB12.fastq"

#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC01.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0710.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC02.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0711.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC03.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0712.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC04.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0713.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC05.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0714.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC06.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0715.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC07.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0721.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run2/BC08.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0722.fastq"

#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC01.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/ATCC13985.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC02.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/ATCC43928.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC03.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0701.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC04.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0702.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC05.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0703.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC06.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0705.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC07.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0707.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run3/BC08.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0724.fastq"

#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC01.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0725.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC02.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0727.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC03.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0801.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC04.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0802.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC05.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0812.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC06.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/KBS0706.fastq"
#cp "${data_path}/nanopore_basecalled_pcbc/run4/BC07.fastq" "${data_path}/nanopore_basecalled_pcbc_rn/PHB12.fastq"



#NanoFilt=/N/u/wrshoema/Carbonate/.local/lib/python3.6/site-packages/nanofilt/NanoFilt.py
#for filename in "${data_path}/nanopore_basecalled_pcbc_rn/"*".fastq"; do
#  if [[ $filename != *"_clean.fastq" ]]; then
#    strain="$( echo $filename | rev | cut -d/ -f1 | rev | cut -d. -f1 )"
#    #fastqc $filename > "${data_path}/nanopore_basecalled_bc_merged/${strain}_results.out"
#    python $NanoFilt -q 10 -l 1000 --headcrop 100 $filename > "${data_path}/nanopore_basecalled_pcbc_rn/${strain}_clean.fastq"
#    #fastqc "${data_path}/nanopore_basecalled_bc_merged/${strain}_clean.fastq" > "${data_path}/nanopore_basecalled_bc_merged/${strain}_clean_results.out"
#  fi
#done
