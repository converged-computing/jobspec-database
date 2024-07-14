# make_par.py
# Makes parameters file for smartpca 
# Authors: Angela Taravella and Anagha Deshpande

import re
import sys 
from optparse import  OptionParser

###############################################################################
USAGE = """
python make_par.py	--map <name of map, with full directory > 
								--ped <name of ped, with full directory >
								--ev < eginvector stem file name, with full directory >
								--out < parameter stem file name, with full directory >

map == stem file name of map
ped == stem file name of ped
out == parameter file stem name
"""

parser = OptionParser(USAGE)
parser.add_option('--map',dest='map', help = ' map file needing to go into analysis')
parser.add_option('--ped',dest='ped', help = 'ped file needing to go into analysis')
parser.add_option('--ev',dest='ev', help = 'output stem name for egienvector file')
parser.add_option('--par',dest='par', help = 'output stem name for parameter file')

(options, args) = parser.parse_args()

parser = OptionParser(USAGE)
if options.map is None:
	parser.error('map file name not given')
if options.ped is None:
	parser.error('ped file name not given')
if options.ev is None:
	parser.error('egienvector file name not given')
if options.par is None:
	parser.error('par file name not given')
############################################################################
# What we want the par file to look like 
# genotypename: < stem file name of ped, with full directory >.ped
# snpname: < stem file name of map, with full directory >.map
# indivname: < stem file name of ped, with full directory >.ped
# evecoutname: <stem file name with directory>.evec
# evaloutname: <stem file name with directory>.eval
# altnormstyle: NO
# familynames: NO 
# numchrom: 23
# noxdata: NO
# numoutlieriter: 0

Outfile = options.par + '_PCA.par'
OutFile = open(Outfile, 'w')

gename = 'genotypename: ' + options.ped + ' '
snpname = 'snpname: ' + options.map + ' '
indname = 'indivname: ' + options.ped + ' '
evec = 'evecoutname: ' + options.ev + '.evec' + ' '
eval = 'evaloutname: ' + options.ev + '.eval' + ' '
options = 'altnormstyle: NO ' + '\n' + 'familynames: NO ' + '\n' + 'numchrom: 23 ' + '\n' + 'noxdata: NO ' + '\n' + 'numoutlieriter: 0 '

OutFile.write('%s\n%s\n%s\n%s\n%s\n%s\n' % (gename, snpname, indname, evec, eval, options))

OutFile.close()



