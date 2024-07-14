'''
Fourth step of the Tuba-seq pipeline- takes as input the ouptut of convert_to_cells.py, and compiles this information in a single .txt file.
The resulting "tumor file" is the starting point for most statistical analyses and visualizations of the data. 
'''

#python3 process_tumors.py --project_name=UCSF_Injury_corr2 --parameter_name=2 --root=/labs/mwinslow/Emily/

from helper_functions import read_project_info, read_parameter_info, get_gc
import getopt
import glob
import sys
from subprocess import call

####################################################
# Step 1, Take arguments, get project info
####################################################
try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["root=", 
        "project_name=", 
        "parameter_name="])
except getopt.GetoptError:
    print("no arguments recognized\n")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--root"):
        root = arg
        print("found root, {}\n".format(root))
    elif opt in ("--project_name"):
        project_id = arg
        print("found project_name, {}\n".format(project_id))
    elif opt in ("--parameter_name"):
        parameter_name = arg
        print("found parameter_name, {}\n".format(parameter_name))
    else:
        assert False, "unhandled option"

project_info_file = root + "/tubaseq_project_files/" + project_id + "_project_file.txt"
parameter_info_file = root + "/tubaseq_parameter_files/" + parameter_name + "_parameter_file.txt"
outfname = root + project_id + "/" + parameter_name + "/summary/" + project_id + "_" + parameter_name + "_tumors.txt"
outfname_no_cutoff = root + project_id + "/" + parameter_name + "/summary/" + project_id + "_" + parameter_name + "_tumors_no_size_cutoff.txt"

spi_out_name = root + project_id + "/" + parameter_name + "/summary/" + project_id + "_" + parameter_name + "_spike_ins.txt"

o = open(outfname, 'wt')
o_all = open(outfname_no_cutoff, 'wt')
s_o = open(spi_out_name, 'wt')



project_info = read_project_info(project_info_file)
parameter_info = read_parameter_info(parameter_info_file)

tumor_output_dir = root + project_id + "/" + parameter_name + "/filtered/"


if project_info.project_type == "Transplant":
	o.write("sgID,BC,Count,CellNum,Gene,Sample,Genotype,Sex,Lung_weight,Titer,GC,Mouse,Source,Organ,Treatment\n")
	o_all.write("sgID,BC,Count,CellNum,Gene,Sample,Genotype,Sex,Lung_weight,Titer,GC,Mouse,Source,Organ,Treatment\n")
	s_o.write("sgID,BC,Count,CellNum,Gene,Sample,Genotype,Sex,Lung_weight,Titer,GC,Mouse,Source,Organ,Treatment\n")
	for s in project_info.sample_ids:
		fname = tumor_output_dir + project_id + "_" + parameter_name + "_" + s + "_final.txt"
		f = open(fname,'rt')
		f.readline()
		for l in f:
			fields = l.strip().split(",")
			if len(fields) >1:
				sgid = fields[0]
				bc = fields[1]
				gc = get_gc(bc)
				o_all.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(l.strip(), project_info.sgids_to_gene[sgid], s, project_info.sample_to_gt[s], project_info.sample_to_gender[s], project_info.sample_to_lungweight[s], project_info.sample_to_titer[s], gc, project_info.sample_to_mouse[s], project_info.sample_to_source[s], project_info.sample_to_organ[s], project_info.sample_to_treatment[s]))
			

				if sgid == "Spi":
					s_o.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(l.strip(), project_info.sgids_to_gene[sgid], s, project_info.sample_to_gt[s], project_info.sample_to_gender[s], project_info.sample_to_lungweight[s], project_info.sample_to_titer[s], gc,project_info.sample_to_mouse[s], project_info.sample_to_source[s], project_info.sample_to_organ[s],project_info.sample_to_treatment[s]))
				cellnum = float(fields[3])
           
            
				if cellnum > parameter_info.min_t_size:
					o.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(l.strip(), project_info.sgids_to_gene[sgid], s, project_info.sample_to_gt[s], project_info.sample_to_gender[s], project_info.sample_to_lungweight[s], project_info.sample_to_titer[s], gc, project_info.sample_to_mouse[s], project_info.sample_to_source[s], project_info.sample_to_organ[s],project_info.sample_to_treatment[s]))
else:
	o.write("sgID,BC,Count,CellNum,Gene,Sample,Genotype,Sex,Lung_weight,Titer,GC\n")
	o_all.write("sgID,BC,Count,CellNum,Gene,Sample,Genotype,Sex,Lung_weight,Titer,GC\n")
	s_o.write("sgID,BC,Count,CellNum,Gene,Sample,Genotype,Sex,Lung_weight,Titer,GC\n")
	for s in project_info.sample_ids:
		fname = tumor_output_dir + project_id + "_" + parameter_name + "_" + s + "_final.txt"
		f = open(fname,'rt')
		f.readline()
		for l in f:
			fields = l.strip().split(",")
			if len(fields) >1:
				sgid = fields[0]
				bc = fields[1]
				gc = get_gc(bc)
				o_all.write("{},{},{},{},{},{},{},{},NA,NA,NA\n".format(l.strip(), project_info.sgids_to_gene[sgid], s, project_info.sample_to_gt[s], project_info.sample_to_gender[s], project_info.sample_to_lungweight[s], project_info.sample_to_titer[s], gc))
				if sgid == "Spi":
					s_o.write("{},{},{},{},{},{},{},{},NA,NA,NA\n".format(l.strip(), project_info.sgids_to_gene[sgid], s, project_info.sample_to_gt[s], project_info.sample_to_gender[s], project_info.sample_to_lungweight[s], project_info.sample_to_titer[s], gc))
				cellnum = float(fields[3])
				if cellnum > parameter_info.min_t_size:
					o.write("{},{},{},{},{},{},{},{},NA,NA,NA\n".format(l.strip(), project_info.sgids_to_gene[sgid], s, project_info.sample_to_gt[s], project_info.sample_to_gender[s], project_info.sample_to_lungweight[s], project_info.sample_to_titer[s], gc))

o.close()
o_all.close()