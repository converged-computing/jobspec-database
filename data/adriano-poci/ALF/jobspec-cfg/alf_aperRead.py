# -*- coding: utf-8 -*-
"""
    alf_aperRead.py
    Adriano Poci
    Durham University
    2022

    <adriano.poci@durham.ac.uk>

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module loads an aperture-spectrum fit, to be executed on the command-
        line.

    Author
    ------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:   27 September 2022
v1.1:   Receive `dcName` command-line argument. 14 July 2023
v1.2:   Generate `*.bestspec2` for aperture fit. 7 November 2023
"""
from __future__ import print_function, division

# General modules
import pathlib as plp
import shutil as su
import numpy as np

# Custom modules
from alf.Alf import Alf
import alf.alf_utils as au
from alf_MUSE import makeSpecFromSum

curdir = plp.Path(__file__).parent

dDir = au._ddir()

import argparse
parser = argparse.ArgumentParser(
    description='Read an alf fit to an aperture spectrum.',
    usage='python alf_aperRead.py -g <galaxy> -sn <SN>'
)
parser.add_argument('-g', '--galaxy', dest='galaxy', type=str)
parser.add_argument('-sn', '--SN', dest='SN', type=int)
parser.add_argument('-dc', '--dcName', dest='dcName', type=str)
args = parser.parse_args()

out = plp.Path(curdir/'results'/f"{args.galaxy}_SN{args.SN:02d}_aperture.mcmc")
print(out)
alf = Alf(out.parent/out.stem, mPath=out.parent)
alf.get_total_met()
alf.normalize_spectra()
alf.abundance_correct()
mIdx = alf.results['Type'].tolist().index('mean')
eIdx = alf.results['Type'].tolist().index('error')

apV = alf.results['velz'][mIdx]
apS = alf.results['sigma'][mIdx]
aph3 = alf.results['h3'][mIdx]
aph4 = alf.results['h4'][mIdx]
apVe = alf.results['velz'][eIdx]
apSe = alf.results['sigma'][eIdx]
aph3e = alf.results['h3'][eIdx]
aph4e = alf.results['h4'][eIdx]

clabels = ['velz', 'sigma', 'h3', 'h4', 'logage', 'zH', 'IMF1', 'IMF2', 'Na']
alf.plot_model(curdir/f"{args.galaxy}{args.dcName}"/'specFit_aperture.pdf')
alf.plot_corner(curdir/f"{args.galaxy}{args.dcName}"/'corner_aperture', clabels)

makeSpecFromSum(args.galaxy, args.SN, full=True, apers=['aperture'],
    dcName=args.dcName)

print()
print(f"\n{apV:.7f},{apVe:.7f},{apS:.7f},{apSe:.7f},{aph3:.7f},{aph3e:.7f},"\
    f"{aph4:.7f},{aph4e:.7f}")