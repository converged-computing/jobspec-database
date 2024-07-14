"""
https://hbctraining.github.io/Intro-to-ChIPseq/lessons/07_handling-replicates-idr.html

Need to run clipper on:
All bams individually
Randomly downsampled bams for each sample
Merged bams
Randomly downsampled merged bams

Need to run SMI on:
All bams individually
Randomly downsampled bams for each sample
Merged bams
Randomly downsampled merged bams

Peak files for IDR
All bams = merged bam peak file
Randomly downsampled bam = sample bam
Randomly downsampled merged bam = merged bam peak file


Input files
Sample bams
Merged bams

Steps
1. Randomly downsample all bams
2. Clipper on any input file
3. SMI 
4. Make into bed files
5. Run appropriate IDR comparisons
"""

import re
import random
import pysam
from collections import defaultdict
import shutil
import math
import os
import copy

# Function to check paths for input files/directories
def _check_path(path):
    if os.path.exists(path):
        return os.path.abspath(path)
    else:
        sys.exit("ERROR: " + path + " does not exist.")


# Parameters from config.yaml
RBFOX_BAMS =config["RBFOX_BAMS"]
INPUT_BAMS = config["INPUT_BAMS"]

reps = list(RBFOX_BAMS.keys())

all_bam_clipper = {}

full_reps = copy.copy(reps)
full_reps.append("merged")

# All bam samples for clipper
for i in full_reps:
	for j in ["rbfox", "input"]:
		file_base = j + "_" + i
		if i == "merged":
			all_bam_clipper[file_base] = os.path.join("results/bams", file_base + ".bam")
		else:
			all_bam_clipper[file_base] = os.path.join("results/bams", file_base + "_downsample.bam")
		all_bam_clipper[file_base + "_pseudo_rep1"] = os.path.join("results/bams", 
			                                          file_base + "_pseudo_rep1_downsample.bam")
		all_bam_clipper[file_base + "_pseudo_rep2"] = os.path.join("results/bams", 
			                                         file_base + "_pseudo_rep2_downsample.bam")

for i in all_bam_clipper:
	_check_path(all_bam_clipper[i])

##################
# Manifest setup #
##################
# All groups for manifest
# Outer dict = "group"
# inner dict = rbfox file = file, input_file = input, peak_file = peak
manifest_groups = defaultdict(dict)

# All bams individually
for i in full_reps:
	group_name = "individual_peakset_" + i
	if i == "merged":
		extension = ".bam"
	else:
		extension = "_downsample.bam"
	rbfox_bam = os.path.join("results/bams", "rbfox_" + i + extension)
	input_bam = os.path.join("results/bams", "input_" + i + extension)
	rbfox_peaks = os.path.join("results/clipper", "rbfox_" + i + "_clipper.bed")
	input_peaks = os.path.join("results/clipper", "input_" + i + "_clipper.bed")
	full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
	             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
	manifest_groups[group_name] = full_dict

for i in full_reps:
	group_name = "merged_peakset_" + i
	if i == "merged":
		extension = ".bam"
	else:
		extension = "_downsample.bam"
	rbfox_bam = os.path.join("results/bams", "rbfox_" + i + extension)
	input_bam = os.path.join("results/bams", "input_" + i + extension)
	rbfox_peaks = os.path.join("results/clipper", "rbfox_merged_clipper.bed")
	input_peaks = os.path.join("results/clipper", "input_merged_clipper.bed")
	full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
	             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
	manifest_groups[group_name] = full_dict


# Randomly downsampled bams for each sample
for i in full_reps:
	for j in ["rep1", "rep2"]:
		save_name = i + "_pseudo_" + j
		group_name = "downsample_individual_peakset_" + save_name
		rbfox_bam = os.path.join("results/bams", "rbfox_" + save_name + "_downsample.bam")
		input_bam = os.path.join("results/bams", "input_" + save_name + "_downsample.bam")
		rbfox_peaks = os.path.join("results/clipper", "rbfox_" + save_name + "_clipper.bed")
		input_peaks = os.path.join("results/clipper", "input_" + save_name + "_clipper.bed")
		full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
		             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
		manifest_groups[group_name] = full_dict

for i in full_reps:
	for j in ["rep1", "rep2"]:
		save_name = i + "_pseudo_" + j
		group_name = "downsample_replicate_peakset_" + save_name
		rbfox_bam = os.path.join("results/bams", "rbfox_" + save_name + "_downsample.bam")
		input_bam = os.path.join("results/bams", "input_" + save_name + "_downsample.bam")
		rbfox_peaks = os.path.join("results/clipper", "rbfox_" + i + "_clipper.bed")
		input_peaks = os.path.join("results/clipper", "input_" + i + "_clipper.bed")
		full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
		             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
		manifest_groups[group_name] = full_dict

# Merged bams
group_name = "merged_peakset_all"
rbfox_bam = os.path.join("results/bams", "rbfox_merged.bam")
input_bam = os.path.join("results/bams", "input_merged.bam")
rbfox_peaks = os.path.join("results/clipper", "rbfox_merged_clipper.bed")
input_peaks = os.path.join("results/clipper", "input_merged_clipper.bed")
full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
manifest_groups[group_name] = full_dict

# Randomly downsampled merged bams
for j in ["rep1", "rep2"]:
	save_name = "merged_pseudo_" + j
	group_name = "downsample_pseudoreplicate_peakset_" + save_name
	rbfox_bam = os.path.join("results/bams", "rbfox_" + save_name + "_downsample.bam")
	input_bam = os.path.join("results/bams", "input_" + save_name + "_downsample.bam")
	rbfox_peaks = os.path.join("results/clipper", "rbfox_" + save_name + "_clipper.bed")
	input_peaks = os.path.join("results/clipper", "input_" + save_name + "_clipper.bed")
	full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
	             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
	manifest_groups[group_name] = full_dict

for j in ["rep1", "rep2"]:
	save_name = "merged_pseudo_" + j
	group_name = "downsample_merged_peakset_" + save_name
	rbfox_bam = os.path.join("results/bams", "rbfox_" + save_name + "_downsample.bam")
	input_bam = os.path.join("results/bams", "input_" + save_name + "_downsample.bam")
	rbfox_peaks = os.path.join("results/clipper", "rbfox_merged_clipper.bed")
	input_peaks = os.path.join("results/clipper", "input_merged_clipper.bed")
	full_dict = {"rbfox_bam":rbfox_bam, "input_bam":input_bam,
	             "rbfox_peaks":rbfox_peaks, "input_peaks":input_peaks}
	manifest_groups[group_name] = full_dict


#############
# IDR setup #
#############

# Peakset provided, smi done on individual peaks
# Pooled all 4246
# Pooled pseudo 10095
# rep 1 4187
# rep 2 5327


full_idr_dict = {}
# pooled pseudo rep1 and 2
pooled_pseudo = {"smi_peak1":"downsample_individual_peakset_merged_pseudo_rep1",
                 "smi_peak2":"downsample_individual_peakset_merged_pseudo_rep2",
                 "peak_list":"merged_peakset_all"}
# rep 1 pseudo rep 1 and 2
rep1 = {"smi_peak1":"downsample_individual_peakset_rep1_pseudo_rep1",
        "smi_peak2":"downsample_individual_peakset_rep1_pseudo_rep2",
        "peak_list":"individual_peakset_rep1"}
# rep 2 pseudo rep 1 and 2
rep2 = {"smi_peak1":"downsample_individual_peakset_rep2_pseudo_rep1",
        "smi_peak2":"downsample_individual_peakset_rep2_pseudo_rep2",
        "peak_list":"individual_peakset_rep2"}
# rep 1 and rep 2
pooled_all = {"smi_peak1":"individual_peakset_rep1",
              "smi_peak2":"individual_peakset_rep2",
              "peak_list":"merged_peakset_all"}

full_idr_dict = {"pooled_pseudo":pooled_pseudo,
				 "rep1":rep1,
				 "rep2":rep2,
				 "pooled_all":pooled_all}



rule all:
	input:
		# Make all necessary bams to do all pseudo replicates of the individual
		expand(
			"results/bams/{type}_downsample.txt",
			type = ["rbfox", "input"]
			),
		# Make bams of the psuedoreplicates of the merged
		expand(
			"results/bams/{type}_downsample_merged.txt",
			type = ["rbfox", "input"]
			),

		# Run clipper on all bams
		expand(
			"results/clipper/{sample}_clipper.bed",
			sample = all_bam_clipper.keys()
			),

		# Run SMI on all bams
		expand(
			"results/smi_manifest/{group}/manifest.txt",
			group = manifest_groups.keys()
			),
		expand(
			"results/run_smi/{group}/{group}_peaks.bed",
			group = manifest_groups.keys()
			),

		# Reformat the smi output to be bed format
		expand(
			"results/run_smi/{group}/{group}_peaks_idr.bed",
			group = manifest_groups.keys()
			),

		# Run idr
		expand(
			"results/idr/{idr_group}_idr_out.txt",
			idr_group = full_idr_dict.keys()
			)


include: "src/rules/setup_bams.snake"
include: "src/rules/clipper.snake"
include: "src/rules/idr.snake"
