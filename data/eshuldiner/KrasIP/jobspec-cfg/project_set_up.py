'''
This script creates a directory structure for a given Tuba-seq analysis (defined by a project ID and parameter ID),
and creates the .inp file with the array of arguments that get fed into the pipeline.

'''
# 1 Get arguments from tubaseq.sh
import getopt
import sys
from helper_functions import read_project_info, read_parameter_info, write_input_file, hamming_distance

try:
	opts, args = getopt.getopt(sys.argv[1:], "p:s:r",["project=", "parameter=", "root="])
except getopt.GetoptError:

	print("no args passed from tubaseq.sh")
	sys.exit(2)
for opt, arg in opts:
	
	if opt in ("--project"):
		project_id = arg

	elif opt in ("--parameter"):
		parameter_id = arg

	elif opt in ("--root"):
		root = arg

	else:
		assert False, "unhandled option passed from tubaseq.sh"

# 2 Create directory structure to store project/parameter combo 
from pathlib import Path

project_dir = root + "/" + project_id
home_dir = project_dir + "/" + parameter_id
raw_counts_dir = home_dir + "/raw_counts"
post_qc_dir = home_dir + "/post_qc"
rejected_dir = home_dir + "/rejected"
summary_dir = home_dir + "/summary"
cluster_summary_dir = summary_dir + "/clustering_summaries"
record_dir = home_dir + "/records"
figure_dir = home_dir + "/figures"	
result_figure_dir = figure_dir + "/results"
ihop_figure_dir = figure_dir + "/ihopping_check"
filtered_dir = home_dir + "/filtered"
qc_figure_dir = figure_dir + "/QC_plots"
jitterplot_figure_dir = figure_dir + "/jitter_plots"

dirs = [project_dir, home_dir, raw_counts_dir, post_qc_dir, rejected_dir, summary_dir, record_dir, figure_dir, result_figure_dir, ihop_figure_dir, filtered_dir, qc_figure_dir, jitterplot_figure_dir, cluster_summary_dir]

for d in dirs:
	Path(d).mkdir(parents=True, exist_ok=True)


# 3 Create .inp file for array (these contain all the information to process each sample)

project_info_file = root + "/tubaseq_project_files/" + project_id + "_project_file.txt"
parameter_info_file = root + "/tubaseq_parameter_files/" + parameter_id + "_parameter_file.txt"

project_info = read_project_info(project_info_file) #light validation feb 13 2020
parameter_info = read_parameter_info(parameter_info_file)

# Check that sgIDs are sufficiently dissimilar from each other

sgids = project_info.sgids
min_dist = len(sgids[0])
for i, sgid1 in enumerate(sgids):
	for j, sgid2 in enumerate(sgids):
		if j>i:
			min_dist = min(min_dist,hamming_distance(sgids[i],sgids[i+1]))
if min_dist < 2:
	print("Uh oh, minimum distance between sgids is {}\n".format(min_dist))
	sys.exit(2)
print("Minimum distance between sgids is: {}\n".format(min_dist))

write_input_file(root, project_info, parameter_info)

