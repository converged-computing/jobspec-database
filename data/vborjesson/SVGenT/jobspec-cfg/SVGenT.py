#!/usr/bin/python 

import sys
import os 
import argparse
import subprocess
import math
import time
import numpy as np
from numpy import loadtxt, dtype,float32
import warnings
cwd = os.getcwd()
module_path = '{}{}'.format(cwd, '/modules')
sys.path.insert(0, module_path)
from coverage_db import read_cov_db
from coverage_db import median_cov
from genotype_caller import call_genotype
from cigar import cigar_count
from check_bam_flag import bam_flag
from vcf_info import create_info
from scipy.stats import norm
#import pandas as pd

################# ARGPARSER ######################

usage = '''SVGenT takes vcf- and bam-files as input and improve the prediction of genotype''' 

parser = argparse.ArgumentParser(description=usage)

parser.add_argument('--vcf', dest='vcf_in', help = 'Path to vcf-file', required= True)
parser.add_argument('--bam', dest='bam_in', help = 'Path to bam-file', required= True) 
parser.add_argument('--tab', dest='tab_in', help = 'Path to tab_file', required= True)
parser.add_argument('--ID', dest='ID', help= 'sample ID', required = False)
parser.add_argument('--sam', dest='sam', help= 'path to sam-file for dry run', required=False)
parser.add_argument('--bwa_ref', dest='bwa_ref', help = 'Path to reference genome for bwa mem', required=True)
#parser.add_argument('--fa', dest= 'fa', help= 'Path to fasta-file with contigs generated from abyss', required = False)

args = parser.parse_args()
#sys.path.insert('./modules/')

vcf = args.vcf_in
bam = args.bam_in
tab = args.tab_in
ID = args.ID
sam = args.sam
bwa_ref = args.bwa_ref
#fasta = args.fa

################# FUNCTION - REGION SPECIFIC ASSEMBLY (called variant-region) ##################################

# This function will generate a new bam file. For every called region; start-position +/- 1000bp and end-position +/- 1000bp are saved. 
# All reads in these regions are then extracted from the BAM-file into a new region-specific BAM-file. This function will call a bah script 
# that execute assembly, including converting bam to fasta, run assembly and mapping back to ref-genome. 
# This function will generate a sam-file. 


def region_specific_assembly (vcf, bam, ID, db, bwa_ref):	

	counter = 0
	subprocess.call ('mkdir ' + ID + '_bam', shell = True)
	subprocess.call ('mkdir ' + ID + '_fasta', shell = True)
	subprocess.call ('mkdir ' + ID + '_assembly', shell = True)
	subprocess.call ('mkdir ' + ID + '_SVGenT_out', shell = True)
	subprocess.call('chmod +x assembly.sh', shell=True)

	file_name = '{}{}{}{}'.format(ID, '_SVGenT_out/', ID, '_SVGenT.vcf')

	valid_chrom = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '16', '17', '18', '19', '20', '21', '22', 'x', 'y']

	with open (vcf, "r") as vcf_in, open (file_name, 'w') as f_out:
		print 'Reading VCF for region-specific assembly'
		ID_counter = 0 # all SVs will have I uniqe number
		info_field = False
		for line in vcf_in:
			if line.startswith("#"):
				if line.startswith("##INFO"):
					info_field = True 
				else: 
					if info_field == True:
						info_GlenX = "##INFO=<ID=SVGenT,Number=10,Type=Float,Description='contig_length|contig_sequence|de_novo_tool|normalized_RD_breakpoint1|normalized_RD_breakpoint2|normalized_RD_breakpoint3|avg(RD)for_all_regions_above_mappability_threshold_0.5|gc-content|mappability-score|genotype2_(using_read-coverage_information)'"
						f_out.write(info_GlenX + '\n')
						info_field = False
				f_out.write(line)		
				continue 
			else:
				split_line=line.lower().replace("chr","").split("\t")
				
				if len(split_line) <= 6: # This SV do not have all the correct information and will therefor not be analyzed 
					f_out.write(line.upper())
					continue

				chromA = split_line[0]

				if chromA not in valid_chrom: # if the chromosome is not one of the usual -> skip 
					#print line
					f_out.write(line.upper()) 
					continue

				posA = split_line[1]
				posB = 0
				tags = split_line[7].split(";")
				for tag in tags:
					if tag.startswith("end="):
						tag = tag.split('=')
						chromB = chromA
						posB = tag[1]	
					elif tag.startswith("svtype=bnd"):
						alt_column = split_line[4].replace("n","").replace("[", "").replace("]","")
						alt_column = alt_column.split(":")
						chromB = alt_column[0]
						posB = alt_column[1]
				if posB == 0: # No secondary mapping have been found; continue with next SV
					f_out.write(line.upper()) 
					continue	

				# unique ID for every region-specific bam-file	
				counter += 1	# Unique ID 
				region_ID = '{}{}{}'.format(ID, "_region_", str(counter))
				assembly_map = '{}{}'.format(ID, '_assembly')

				posA = int(posA)
				posB = int(posB)
				posA_start = posA - 1000
				posA_end = posA + 1000
				posB_start = posB - 1000
				posB_end = posB + 1000	
				region2 = ""			  

				# Check for overlapping regions (we do not want double reads)
				if chromA == chromB:
					if posA < posB:
						if posA_end >= posB_start: # overlapping region
							region = '{}:{}-{}'.format(str(chromA), str(posA_start), str(posB_end))
							region2 = '{}:{}-{}'.format(str(chromA), 0,0)
						else: 
							region = '{}:{}-{}'.format(str(chromA), str(posA_start), str(posA_end)) 
							region2 = '{}:{}-{}'.format(str(chromB), str(posB_start), str(posB_end))

					if posA > posB:
						if posB_end >= posA_start:
							region = '{}:{}-{}'.format(str(chromA), str(posB_start), str(posA_end))
							region2 = '{}:{}-{}'.format(str(chromA), 0,0)
						else:
							region = '{}:{}-{}'.format(str(chromA), str(posA_start), str(posA_end)) 
							region2 = '{}:{}-{}'.format(str(chromB), str(posB_start), str(posB_end))				
				else:
					region = '{}:{}-{}'.format(str(chromA), str(posA_start), str(posA_end))
					region2 = '{}:{}-{}'.format(str(chromB), str(posB_start), str(posB_end))
				
				# median chromosome coverage divided in four, will give us a threshold of minimum read coverage that will be returned
				#cov_med = median_cov (db)
				# minimum kmer coverage for de novo assembly
				#min_cov = int(cov_med) / 4
				min_cov = 7
				print 'Minimum coverage accepted: ', min_cov
				print 'Initiate de novo assembly and mapping contigs back to reference'
				#print 'region1-2', region, region2
				start = time.time()
				process = ['./assembly.sh', bam, region, ID, region_ID, str(min_cov), bwa_ref, region2, '>/dev/null']
				os.system(" ".join(process))
				ass_time = time.time() - start 
				subprocess.call('echo '+ str(ass_time) + ' >> time_2.txt', shell = True)
				sam = '{}/{}_mapped.sam' .format(assembly_map, region_ID)

			print sam, chromA, chromB, posA_start, posA_end, posB_start, posB_end, db	
			sv_info, genotype1, sv_type, statistics = call_genotype (sam, chromA, chromB, posA_start, posA_end, posB_start, posB_end, db)	
			
			# If no breakpoints with good quality could be found inside our region, the old SV wiil be written to our new vcf
			if sv_type == 'N/A':
				f_out.write(line.upper()) 
				print 'Variant not found, old info is copied to vcf-file'
				continue

			# If SV classification have been done using only read_coverage over region
			if len(sv_info) == 3:
				ID_counter += 1
				split_line[0] = chromA
				split_line[1] = str(sv_info[1]) # start_pos
				split_line[2] = 'SVGenT_{}'.format(str(ID_counter))	
				split_line[4] = '<{}>'.format(sv_type)
				split_line[6] = 'PASS'
				split_line[8] = 'GT'
				split_line[9] = '{}'.format (genotype1)
				GlenX_stats = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format('N/A', 'N/A', 'N/A', statistics['RD_norm_1'], statistics['RD_norm_2'], statistics['RD_norm_3'], statistics['RD_all'], statistics['RD_gc_1'], statistics['map_1'], statistics['genotype2'])
				sv_len = int(sv_info[1]) - int(sv_info[2])
				old_info = split_line[7]
				glen_info = 'END={};SVTYPE={};SVLEN={};SVGenT={}'.format(sv_info[2], sv_type, sv_len, GlenX_stats)	

			else:	
				# Manipulate line with new improved SV information
				ID_counter += 1
				split_line[0] = str(chromA)
				split_line[1] = str(sv_info[2]) # Breakpoint for SV
				split_line[2] = 'SVGenT_{}'.format(str(ID_counter))
				split_line[4] = '<{}>'.format(sv_type)
				split_line[6] = 'PASS'
				split_line[8] = 'GT'
				split_line[9] = '{}'.format (genotype1)
				
				contig_l = sv_info[9]
				contig_seq = sv_info[10]
				de_novo_tool = sv_info[11]
				sv_len = int(sv_info[2]) - int(sv_info[6])

				if sv_type == 'tBND':
					split_line[4] = '<BND>'
					# INFO-field: contig length, seq, normalized read ceverage, raw read-coverage, gc-content and mappabilty score
					GlenX_stats = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(contig_l, contig_seq, de_novo_tool, 'NA', statistics['RD_norm_2'], statistics['RD_norm_3'], statistics['RD_all'], 'NA', 'NA', statistics['genotype2']) 
					old_info = split_line[7]					
					glen_info = 'SVTYPE=BND;SVGenT={}'.format(GlenX_stats) #;CHRA=' + chromA  + ';CHRB=split_line[7] = 'SVTYPE=BND' #;CHRA=' + chromA  + ';CHRB=' + chromB + ';END=' + sv_info[7] ' + chromB + ';END=' + sv_info[7] # mating breakpoint 
					split_line[4] = 'N[{}:{}[' .format(chromB, sv_info[6])								



				else:	
					# INFO-field: contig length, seq, normalized read ceverage, raw read-coverage, gc-content and mappabilty score
					GlenX_stats = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(contig_l, contig_seq, de_novo_tool, statistics['RD_norm_1'], statistics['RD_norm_2'], statistics['RD_norm_3'], statistics['RD_all'], statistics['RD_gc_1'], statistics['map_1'], statistics['genotype2']) 
					old_info = split_line[7]

					if sv_type == 'BND':
					 	glen_info = 'SVTYPE=BND;SVGenT={}'.format(GlenX_stats) #;CHRA=' + chromA  + ';CHRB=split_line[7] = 'SVTYPE=BND' #;CHRA=' + chromA  + ';CHRB=' + chromB + ';END=' + sv_info[7] ' + chromB + ';END=' + sv_info[7] # mating breakpoint 
						split_line[4] = 'N[{}:{}[' .format(chromB, sv_info[6])
					elif sv_type != 'BND':
						if sv_type == "DEL":
							glen_info = 'END={};SVTYPE={};SVLEN={};SVGenT={}'.format(sv_info[6], sv_type, sv_len, GlenX_stats) #'SVTYPE=' + sv_type + ';CHRA=' + chromA  + ';CHRB=' + chromB + ';END=' + sv_info[7] 
						if sv_type == "DUP" or sv_type == "INV":
							glen_info = 'SVTYPE={};END={};SVGenT={}'.format(sv_type, sv_info[6], GlenX_stats)

			#Create a info filed in INFO containing old info and new info. 			
			info = create_info(old_info, glen_info)		
			split_line[7] = info

			print 'Variant updated in vcf-file'
			vcf_sv = '\t'.join(split_line) # make new tab seperated line of list, (preparations for writing to vcf-file) 
			string = str(vcf_sv)
			f_out.write(string.upper() + "\n")

	return 'All variants in the vcf-file have been assembled, mapped and analyzed.'		


#======================================================================================================
# START SVGenT
#======================================================================================================

# Start by creating database containing gc-content, read coverage, mappability and position per 100 base.  
print 'Starts SVGenT..'
db = read_cov_db(ID, tab)
print 'The database: ', db, 'is completed'	

# Read in the vcf file and perform assemble, statistics and update VCF.
assembly = region_specific_assembly (vcf, bam, ID, db, bwa_ref)


