# make_merge_list.py
# Makes parameters file for smartpca 
# Authors: Angela Taravella and Anagha Deshpande

import re
import sys 
from optparse import  OptionParser

###############################################################################
USAGE = """
python mk_merge_list_auto.py	--path <path to the plink files >
											--stem <stem file name of plink file without chrosome number ie. without "chri_" > 
											--out <stem name of the merge list text file >

path == path to the plink files
stem == stem file name of plink files without chr number
out == output stem file name of the merge list 
"""

parser = OptionParser(USAGE)
parser.add_option('--path',dest='path', help = ' path to the ped files ')
parser.add_option('--stem',dest='stem', help = ' stem file name needing to go into analysis')
parser.add_option('--out',dest='out', help = ' output stem name for the merge list file')

(options, args) = parser.parse_args()

parser = OptionParser(USAGE)
if options.path is None:
	parser.error('path to files not given')
if options.stem is None:
	parser.error('stem file name not given')
if options.out is None:
	parser.error('output stem name not given')
############################################################################

Outfile = options.out + '.txt'
OutFile = open(Outfile, "w")

for i in range(1,23):  # this will specify all the chromosome numbers
	file_line = options.path + "chr" + str(i) + options.stem + ".ped " + "autosomes/merge/" + "chr" + str(i) + options.stem + ".map"
	OutFile.write(file_line)
	OutFile.write("\n")

OutFile.close()
