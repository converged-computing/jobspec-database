#!/bin/bash
# This script cleans files generated from VASP runs.
# -Uthpala Herath

rm -rf CHG DOSCAR EIGENVAL ENERGY IBZKPT OSZICAR* OUTCAR* PCDAT REPORT  TIMEINFO  WAVECAR XDATCAR wannier90.wout wannier90.amn wannier90.mmn wannier90.eig wannier90.chk wannier90.node* CHGCAR* PROCAR *.o[0-9]* vasprun.xml
