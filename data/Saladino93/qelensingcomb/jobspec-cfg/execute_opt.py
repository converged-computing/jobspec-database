import argparse

import yaml

from mpi4py import MPI

import pathlib

import re

import numpy as np

import os

import itertools

my_parser = argparse.ArgumentParser(description = 'Configuration file.')

my_parser.add_argument('Configuration',
                       metavar='configuration file',
                       type = str,
                       help = 'the path to configuration file')
my_parser.add_argument('Crosses',
                       type = int)

args = my_parser.parse_args()

values_file = args.Configuration
crosses = args.Crosses

if not pathlib.Path(values_file).exists():
    print('The file specified does not exist')
    sys.exit()

with open(values_file, 'r') as stream:
            data = yaml.safe_load(stream)

output = data['analysisdirectory']

#path = pathlib.Path(output)
#all_lmaxes_directories =  [x.name for x in path.iterdir() if x.is_dir()]

estimatorssubset = data['estimatorssubset']

estimators_dictionary = data['estimators']

if estimatorssubset != '':
    estimators = estimatorssubset


lista_lmaxes = []

for e in estimators:
    elemento = estimators_dictionary[e]
    lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
    num = elemento['number']
    lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]   



all_lmaxes_directories = list(itertools.product(*lista_lmaxes))

print(len(all_lmaxes_directories))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = 0

mock_numb = len(all_lmaxes_directories)
delta = mock_numb/size

iMax = (rank+1)*delta+start
iMin = rank*delta+start

iMax = int(iMax)
iMin = int(iMin)

if iMax > mock_numb:
	iMax = mock_numb
elif (iMax >= (mock_numb - delta)) and iMax < mock_numb:
	iMax = mock_numb

optdict = data['optimisation']

gtol = optdict['gtol']
fbs = optdict['fbs']
inv_variances = optdict['inv_variances']
noiseequalsbias = optdict['noiseequalsbias']



for inv_ in inv_variances:
    for neb in noiseequalsbias:
        for fb in fbs:
            for i in range(iMin, iMax):
                lista = all_lmaxes_directories[i] #re.findall(r'\d+', all_lmaxes_directories[i])
                s = ''
                for l in lista:
                    s += f'{l} '
                if len(lista) <= 4:
                    os.system(f'python lmax_optimize.py {values_file} {fb} {gtol} {neb} {inv_} {crosses} {s} diff-ev')            
