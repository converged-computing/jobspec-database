#!/usr/bin/env python3
# script: HSPR_AZ_tidy.py
# author: Rituparna Goswami, Enrico Sandro Colizzi, Alexis Jiayi Zeng
script_usage="""
usage
    For de novo parameter setting: HSPR_AZ_hpc.py -nit <numberInteration> -wa2 <withA2> -mdn <modelVersion> [options]
    For imported parameter setting: HSPR_AZ_hpc.py -nit <numberInteration> -ids <ImportDataset>
    To run on HPC, use -psd 0

version
    HSPR_AZ_hpc.py 0.0.2 (alpha)

dependencies
    Python v3.9.7, Scipy v1.11.2, NumPy v1.22.2, viennarna v2.5.1, Matplotlib v3.5.1, pandas v2.1.0

description
    For Gillespie Simulation

################################################################

--numberInteration,-nit
    The number of interations for Gillespi simulation

--importParamDataset,-ids
    Dataset name from which parameters are extracted. From SA: '2024-02-24_step2_time900_hss600_hsd50/11.792037432365467'. // From simuData: '2023-11-26_numIter1_Time20000.002736231276_HSstart10000_HSduration5000'. // From variance analysis: "159.461275414781"...... All without .csv (default: )

--modelName,-mdn
    which model version or Gillespie Function to use (default: )

--timeStep,-tsp
    The time duration for each Gillespi simulation run/iteration (default:1000)

--heatShockStart,-hss
    The time point at which a heat shock is introduced (default:600)

--heatShockStart2,-hs2
    The time point at which a second heat shock is introduced (default: 0)

--heatShockDuration,-hsd
    The duration of heat shock introduced (default: 30)

--withA2,-wa2
    whether HSFA2 is present in the model (default: 0)

--A2positiveAutoReg,-a2p
    Whether HSFA2 positively regulates itself in the model (default: 0)

####################################################################

--misfoldRateNormal,-mfn
    The formation rate of misfolded protein from folded protein under normal temperature (default: 0.01)

--misfoldRateHS,-mfh
    The formation rate of misfolded protein from folded protein under heat shock condition (default: 0.05)

--assoA1_HSPR,-aah
    c1, association rate between A1 and HSPR (default:10.0)

--repMMP_A1H,-rma
    c2, rate at which MMP replaces A1 from A1_HSPR complex, to form MMP_HSPR instead (default: 5.0)

--assoMMP_HSPR,-amh
    c3, The association rate between MMP and HSPR (default: 0.5)

--repA1_MMPH,-ram
    c4, rate at which A1 replaces MMP from MMP_HSPR complex, to form A1_HSPR instead (default: 0.5)

--disoA1_HSPR,-dah
    d1, dissociation rate of A1-HSPR (default: 0.1)

--HSdisoA1_HSPR,-hda
    d1_HS, dissociation rate of A1-HSPR (default: 0.1)

--disoMMP_HSPR,-dmh
    d3, dissociation rate of A1-HSPR (default: 0.01)

--hillCoeff,-hco
    The Hill Coefficient (default: 2)

##################################################################################

--leakage_A1,-lga
    Trancription leakage for HSFA1 (default: 0.001)

--leakage_B,-lgb
    Trancription leakage for HSFB (default: 0.001)

--leakage_HSPR,-lgh
    Trancription leakage for HSPR (default: 0.001)

--leakage_A2,-la2
    Trancription leakage for HSFA2 (default: 0.001)

################################################################################

--hilHalfSaturation,-hhs
    The conc of inducer/repressor at half-max transcriptional rate (default: 1.0)

--KA1actA1,-hAA
    h1, K constant for A1, activate by A1, repress by B (default: 1.0)

--KA1actHSPR,-hAH
    h2, K constant for HSPR synthesis, activate by A1, repress by B (default: 1.0)

--KA2,-hA2
    h4, K constant for HSFA2, activate by A1, repress by B (default: 1.0)

--KA2act,-h3A
    'h3', A2 activating A1 and HSPR, on top of HSFB repression (default: 5.0)

--KA1actB,-hAB
    h5, K constant for HSFB (default: 1.0)

##################################################################################

--initFMP,-ifp
    Initial FMP abundance (default: 5000)

--initMMP,-imp
    Initial MMP abundance (default: 0)

--initA1_HSPR,-iah
    initial C_HSFA1_HSPR abundance (default: 50)

--initA1free,-iaf
    initial free A1 (default: 1)

--initB,-ibf
    initial HSFB (default: 1)

--initHSPRfree,-ihf
    initial free HSPR (default: 2)

--initMMP_HSPR,-imh
    initial C_MMP_HSPR abundance (default: 50)

##################################################################################

--maxRateB,-a5B
    a5, Max transcription rate of HSFB (default: 10)

--maxRateA1,-a1A
    a1, Max transcription rate of HSFA1 (default: 10)

--maxRateHSPR,-a2H
    a2, Max transcription rate of HSFB (default: 100)

--maxRateA2act,-a3A
    'a3', by how much A2 activate A1 and HSPR (default: 50)

--maxRateA2,-a4A
    a4, max transcription rate of HSFA2 (default: 10)

--foldedProduction,-fpp
    a7, folded protein production rate (default: 300)

-refoldRate,-a6R
    a6, refolding rate from MMP-HSPR (default: 0.2)

--globalDecayRate,-gdr
    the decay rate of species except for MMP (default:0.04)

###################################################################################

--A1mut,-a1m
    if 1, no HSFA1 (default: 0)

--Bmut,-bmu
    if 1, no HSFB (default: 0)

--HSPRmut,-hmu
    if 1, no HSPR (default: 0)

#####################################################################################

--outputFormat,-ofm
    Whether to save Gillespi simulation output as csv or picklefile (default: csv)

--samplingFreq,-spf
    How often is simulation datasaved (default: 0.1)

--thread,-thr
    The number of threads used for multiprocessing (default: 4)

--plotsimuData,-psd
    whether to plot simulation data (default: 1)

--savesimuData,-ssd
    whether to save simulation data (default: 1)

--saveFig,-sfg
    Whether to save the figures and plots generated. Default = True (default: 1)

--showFig,-shf
    Whether to show  the figures generated (default: 0)

################################################################################

reference
    A.J.Zeng
    xxxxxxxxx
"""

from pathlib import Path
import time 
import re
import argparse as ap
import pickle
from pydoc import describe
from datetime import datetime
import traceback

import random
import math
import matplotlib.pyplot as plt #for my sanity
import numpy as np
import os
import csv
from os.path import join
import pandas as pd
import sys
import multiprocessing as mp
import multiprocessing as mp


def main(opt):
    #print(type(sys.argv)) #a list
    #print(sys.argv.index('sfaisuf'))
    #exit()
    print("Step1: Specify output directory")
    data_dir, param_rootdir, varPara_dir, varData_dir = dir_gen()

    print("Step2: Specify parameters")
    if bool(opt.ids) == False: # de novo setting
        param_dict = param_spec(opt)
        
    else: 
        param_dict, opt = load_Param_fromFile(param_rootdir, data_dir, varPara_dir, opt)
    
    print("Step3: Simulation begins")
    ## no multiprocessing

    listM4, listtime2, numberofiteration, end_time, rr_list2, model_name  = gillespie(param_dict, opt)

    ## with multiprocessing
    #listM2, listtime, end_time = gillespie_woA2_mp(param_dict, opt)
    #listM6, numberofiteration, end_time = parallel_gillespie_woA2(param_dict, opt)

    print("Step4: Combine and save data")
    listM6 = combine_data(listtime2, listM4, rr_list2, opt)
    if bool(opt.ssd) == True:
        data_file = saveGilData(listM6, param_dict, data_dir, varData_dir, param_rootdir, numberofiteration, end_time, model_name, opt)

    
    if bool(opt.ids) == False: # de novo setting
        saveParam(param_dict, data_dir, numberofiteration, end_time, model_name, opt)

    if bool(opt.psd) == True:
        print("Step 5: Plot Gillespie Outcome")
        plot_hpcSimuOutcome(listM6, data_file, param_dict, opt)





##################################################################
## To Extract From Existing File
##################################################################


def load_Param_fromFile(param_rootdir, data_dir, varPara_dir, opt):
    try: 
        S, cost_func, param_dict = loadData(f"{param_rootdir}/{opt.ids}.pcl")
        para_csv_name = f"{param_rootdir}/{opt.ids}.pcl"
        opt.S = S
        opt.cost_func = cost_func
        if os.path.getctime(para_csv_name) < datetime(2024,2,28):
                model_name = "replaceA1"
        else: model_name = extract_model_name(opt.ids)
    except FileNotFoundError:
        if os.path.exists(f"{param_rootdir}/{opt.ids}.csv"):
            para_csv_name = f"{param_rootdir}/{opt.ids}.csv"
            ctime = datetime.fromtimestamp(os.path.getmtime(para_csv_name))
            #print(f"{ctime} {datetime(2024,2,28)}")
            #print(f"{bool(ctime < datetime(2024,2,28))}")
            #exit()
            if ctime < datetime(2024,2,28):
                model_name = "replaceA1"
            else: model_name = extract_model_name(opt.ids)
        elif os.path.exists(f"{data_dir}/Exp3_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/Exp3_Para_{opt.ids}.csv"
            model_name = model_from_date(para_csv_name)
        elif os.path.exists(f"{data_dir}/replaceA1_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/replaceA1_Para_{opt.ids}.csv"
            model_name = "replaceA1"
        elif os.path.exists(f"{data_dir}/woA2_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/woA2_Para_{opt.ids}.csv"
            model_name = "woA2"
        elif os.path.exists(f"{data_dir}/d1upCons_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/d1upCons_Para_{opt.ids}.csv"
            model_name = "d1upCons"
        #### variance analysis
        elif os.path.exists(f"{varPara_dir}/{opt.ids}.csv"):
            para_csv_name = f"{varPara_dir}/{opt.ids}.csv"
            model_name = "replaceA1"

        param_dict = {}
        with open(para_csv_name, 'r') as param_file:
            csv_reader = csv.reader(param_file)
            headers = next(csv_reader)
            data = next(csv_reader)
        for key, val in zip(headers, data):
            if key == 'model_name': 
                param_dict[key] = str(val)
            else: param_dict[key] = float(val)

    if not 'hstart2' in param_dict: param_dict['hstart2'] = 0
    if not 'model_name' in locals(): model_name = param_dict['model_name']
    param_dict['model_name'], opt.mdn = model_name, model_name
    opt.para_csv_name = para_csv_name

    param_dict['numberofiteration'] = int(opt.nit)

    if any(s in sys.argv for s in ['-tsp', '-hsd', 'hss']):
        #print(sys.argv)
        opt = customise_simu(opt)
    #else: print('pass')

    init_HSFA1, a1, leakage_A1, init_HSFB, a5, leakage_B, init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR, init_C_HSFA1_HSPR_val = gen_mutant(opt)
    param_dict = reAssign_paramVal(param_dict, init_HSFA1, a1, leakage_A1, init_HSFB, a5, leakage_B, init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR, init_C_HSFA1_HSPR_val)
    return param_dict, opt


def customise_simu(opt):
    if '-tsp' or '--timeStep' in sys.argv:
        try: pos = sys.argv.index('-tsp')
        except ValueError: pos = sys.argv.index('--timeStep')
        opt.tsp = int(sys.argv[pos + 1])
    if '-hss' or '--heatShockStart' in sys.argv:
        try: pos = sys.argv.index('-hss')
        except ValueError: pos = sys.argv.index('--heatShockStart')
        opt.hss = int(sys.argv[pos + 1])
    if '-hsd' or '--heatShockDuration' in sys.argv:
        try: pos = sys.argv.index('-hsd')
        except ValueError: pos = sys.argv.index('--heatShockDuration')
        opt.hsd = int(sys.argv[pos + 1])
    return opt

def model_from_date(file):
    ctime = datetime.fromtimestamp(os.path.getmtime(file))
    woA2_start = datetime(2023,12,6,23,30)
    replaceA1_start = datetime(2024,2,16,15,8)
    others_start = datetime(2024,2,26)
    only_d1upCons = datetime(2024,2,19,10,25)
    if ctime >= woA2_start and ctime < replaceA1_start: model_name = "woA2"
    elif ctime > replaceA1_start and ctime <= others_start: 
        if ctime == only_d1upCons: model_name = 'd1upCons'
        else: model_name = "replaceA1"
    else: print('model_from_date() receives unexpected param_file creation date')
    return model_name


def extract_model_name(filename):
    # Define the pattern for extracting the model name
    pattern = re.compile(r'\(\w+\)_(\w+)_')
    # Use the pattern to find a match in the filename
    match = pattern.search(filename)
    # If a match is found, return the captured group (model name)
    if match:
        return match.group(1)
    else:
        print('Error, cannot extract model name with extract_model_name()')
        exit()  # Return None if no match is found

def reAssign_paramVal(param_dict, init_HSFA1, a1, leakage_A1, init_HSFB, a5, leakage_B, init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR, init_C_HSFA1_HSPR_val):
    param_dict['init_HSFA1'] = init_HSFA1
    param_dict['a1'] = a1
    param_dict['leakage_A1'] = leakage_A1
    param_dict['init_HSFB'] = init_HSFB
    param_dict['a5'] = a5
    param_dict['leakage_B'] = leakage_B
    param_dict['init_C_HSPR_MMP'] = init_C_HSPR_MMP
    param_dict['init_HSPR'] = init_HSPR
    param_dict['a2'] = a2
    param_dict['leakage_HSPR'] = leakage_HSPR
    param_dict['init_C_HSFA1_HSPR'] = init_C_HSFA1_HSPR_val
    return param_dict

def gen_mutant(opt):
    if bool(opt.a1m) == False:
        init_HSFA1, a1, leakage_A1 = int(opt.iaf), int(opt.a1A), float(opt.lga)
    elif bool(opt.a1m) == True: 
        print("A1 mutant")
        init_HSFA1, a1, leakage_A1 = 0, 0,0

    if bool(opt.bmu) == False: 
        init_HSFB, a5, leakage_B = int(opt.ibf), int(opt.a5B), float(opt.lgb)
    elif bool(opt.bmu) == True: 
        print("HSFB mutant")
        init_HSFB, a5, leakage_B = 0,0, 0

    if bool(opt.hmu) == False:
        init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR = int(opt.imh), int(opt.ihf), int(opt.a2H), float(opt.lgh)
    elif bool(opt.hmu) == True: 
        print("HSPR mutant")
        init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR = 0,0,0, 0

    if bool(opt.hmu) == True or bool(opt.a1m) == True:
        init_C_HSFA1_HSPR_val = 0
    else: init_C_HSFA1_HSPR_val = int(opt.iah)
    return init_HSFA1, a1, leakage_A1, init_HSFB, a5, leakage_B, init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR, init_C_HSFA1_HSPR_val

#######################################################################
## 1. Parameter specification
#######################################################################
def param_spec(opt):
    init_HSFA1, a1, leakage_A1, init_HSFB, a5, leakage_B, init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR, init_C_HSFA1_HSPR_val = gen_mutant(opt)
    param_dict = {
        ## initial concentrations
        'init_HSFA1': init_HSFA1,
        'init_C_HSFA1_HSPR': init_C_HSFA1_HSPR_val,
        'init_HSPR': init_HSPR,
        'init_MMP': int(opt.imp),
        'init_FMP': int(opt.ifp),
        'init_C_HSPR_MMP': init_C_HSPR_MMP,
        'init_HSFB': init_HSFB,
        'Time': 0.0,
        ## Maximum expression level in Hill equation
        'a1': a1,
        'a2': a2,
        'a5': a5,
        'a6': float(opt.a6R), # refolding rate from MMP-HSPR
        'a7': int(opt.fpp), #folded protein production rate
        #'a8': 50.0,
        ## Ka in Hill equation
        'h1': float(opt.hAA),
        'h2': float(opt.hAH),
        #'h3': int(opt.hBA),
        #'h4': int(opt.hBH),
        'h5': float(opt.hAB),
        #'h6': int(opt.hBB),
        ## association rates
        'c1': float(opt.aah), # binding rate between A1 and HSPR
        'c3': float(opt.amh), # binding rate between MMP and HSPR
        ## decay rates
        'd1': float(opt.dah), # decay path 1 of A1-HSPR
        'd3': float(opt.dmh), # dissociation rate of MMP-HSPR
        'd4_heat': float(opt.mfh),
        'd4_norm': float(opt.mfn),
        'Decay1': float(opt.gdr),
        'Decay2': float(opt.gdr), # decay of free HSPR
        'Decay4': float(opt.gdr),
        'Decay6': float(opt.gdr),
        'Decay7': float(opt.gdr), # decay path 2 of A1-HSPR
        'Decay8': float(opt.gdr), # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage_A1': leakage_A1,
        'leakage_B': leakage_B,
        'leakage_HSPR': leakage_HSPR,
        'numberofiteration': int(opt.nit),
        'hillcoeff': int(opt.hco),
        'hstart':int(opt.hss),
        'hstart2':int(opt.hs2),
        'hduration':int(opt.hsd),
        'model_name': str(opt.mdn)
    }
    if bool(opt.wa2) == True:
        param_dict['leakage_A2'] = float(opt.la2)
        param_dict['a4'] = int(opt.a4A) # A1 activate A2
        param_dict['h4'] = float(opt.hA2) # A1 activate A2
        param_dict['a3'] = int(opt.a3A) # A2 activate A1 and HSPR
        param_dict['h3'] = int(opt.h3A) # A2 activate A1 and HSPR
        param_dict['Decay3'] = float(opt.gdr)
        param_dict["init_HSFA2"] = 1
    if opt.mdn == 'replaceA1':
        param_dict['c2'] = float(opt.rma) # MMP replace A1 in complex with HSPR
        param_dict['c4'] = float(opt.ram) # A1 replace MMP in complex with HSPR
    elif opt.mdn == 'd1upCons':
        param_dict['d1_HS'] = float(opt.hda)
    
    print(param_dict)
    return param_dict

##########################################################################
## 2. Generate Output Directories
##########################################################################
def dir_gen():
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)

    data_dir = os.path.join(partiii_dir,"Ritu_simulation_data")
    if not os.path.isdir(data_dir): os.makedirs(data_dir, 0o777)
    #plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    #if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)
    param_rootdir = os.path.join(partiii_dir,"Param_Optimisation")
    if not os.path.isdir(param_rootdir): os.makedirs(param_rootdir, 0o777)

    varPara_dir = os.path.join(partiii_dir,"param_for_varAna")
    if not os.path.isdir(varPara_dir): os.makedirs(varPara_dir, 0o777)
    varData_dir = os.path.join(partiii_dir,"simuData_for_varAna")
    if not os.path.isdir(varData_dir): os.makedirs(varData_dir, 0o777)
    return data_dir, param_rootdir, varPara_dir, varData_dir

############################################################################
## 3. Gillespi Simulation
############################################################################

def gillespi_archive(param_dict, opt):
    listM4=[]
    listtime2=[]
    numberofiteration = param_dict["numberofiteration"]
    for i in range(numberofiteration):    
        print(f" iteration: {i}\n")
        listM = np.array([param_dict["init_HSFA1"],
                          param_dict["init_HSPR"],
                          param_dict["init_C_HSFA1_HSPR"],
                          param_dict["init_MMP"],
                          param_dict["init_FMP"],
                          param_dict["init_C_HSPR_MMP"],
                          param_dict["init_HSFA2"],
                          param_dict["init_HSFB"]])
        listM2 =[listM]
        Time=0
        listtime =[Time]

        counter = 0
        while Time < int(opt.tsp): 
            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')

            a1 = param_dict['a1']
            a2 = param_dict['a2']
            a3 = param_dict['a3']
            a4 = param_dict['a4']
            a5 = param_dict['a5']
            a6 = param_dict['a6']
            a7 = param_dict['a7']
            a8 = param_dict['a8']
            h1 = param_dict['h1']
            h2 = param_dict['h2']
            h3 = param_dict['h3']
            h4 = param_dict['h4']
            h5 = param_dict['h5']
            h6 = param_dict['h6']
            c1 = param_dict['c1']
            c3 = param_dict['c3']
            d1 = param_dict['d1']
            d3 = param_dict['d3']
            Decay1 = param_dict['Decay1']
            Decay2 = param_dict['Decay2']
            Decay3 = param_dict['Decay3']
            Decay4 = param_dict['Decay4']
            Decay6 = param_dict['Decay6']
            Decay7 = param_dict['Decay7']
            Decay8 = param_dict['Decay8']
            Decay5 = param_dict['Decay5']
            leakage = param_dict['leakage']
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
            else: d4 = param_dict['d4_norm']
                
            #HSFa1 andHSFA2 may makes complex
            #increase in HSFA1 by transcription and dessociation from the complex C_HSFA1_HSPR
            R_HSFA1_inc=leakage+a1*HSFA1/(h1+HSFA1+HSFB)+ a3*HSFA2/(h3+HSFA2+HSFB) # + d1*C_HSFA1_HSPR
            #decrease in HSFA1 by association to the 1st complex C_HSFA1_HSPR and decay in the protein
            R_HSFA1_dec= Decay1*HSFA1
            #increase in HSPR by transcription and dess
            R_HSPR_inc= leakage+a2*HSFA1/(h2+HSFA1+HSFB)+ a3*HSFA2/(h3+HSFA2+HSFB)
            #decrease in HSPR by transcription and dess **-> should be decay
            R_HSPR_dec= Decay2*HSPR
            #increase in C_HSFA1_HSPR association to the 1st complex
            R_C_HSFA1_HSPR_inc=c1*HSFA1*HSPR
            #decrease in C_HSFA1_HSPR dissociation from the 1st complex and degradation of the complex as a whole (?)
            R_C_HSFA1_HSPR_dec1=d1*C_HSFA1_HSPR
            R_C_HSFA1_HSPR_dec2=Decay7*C_HSFA1_HSPR
            #increase in MMP when 2nd complex decreases
            R_MMP_inc= d4*FMP
            #decrease in MMP by 2nd complex increases (may be we want to change the slope of dexcay later)
            R_MMP_dec= Decay5*MMP
            #increase in FMP by FMP production and MMP to FMP
            R_FMP_inc=a7  #how to write the production?
            #decrease in FMP by decay or
            R_FMP_dec= Decay6*FMP
            #increase in HSPR_MMP by association to the 2nd complex
            R_C_HSPR_MMP_inc=c3*HSPR*MMP #how to write the production?
            #decrease in HSPR_MMP by dissociation from the 2nd complex
            R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP
            R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP
            R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP
            #increase in HSFA2 by transcription with TF HSFA1 
            R_HSFA2_inc=leakage+a4*HSFA1/(h4+HSFA1+HSFB) # + a8*HSFA2/(h6+HSFA2+HSFB)
            #decrease in HSFA2 by transcription and dess
            R_HSFA2_dec=Decay3*HSFA2
            #increase in HSFB by transcription with TF HSFA1 and HSFB
            R_HSFB_inc=leakage+a5*HSFA1/(h5+HSFA1+HSFB)
            #decrease in HSFB by transcription and dess
            R_HSFB_dec=Decay4*HSFB


            listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3, R_HSFA2_inc,R_HSFA2_dec,R_HSFB_inc,R_HSFB_dec])

            TotR = sum(listR) #production of the MRNA 
            Rn = random.random() #getting random numbers
            Tau=-math.log(Rn)/TotR #when the next thing happen
            #Rn2= random.uniform(0,TotR) # for the next random number
            # HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB
            Stoich = [[1,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,-1,0,0,0,0,0,0], [-1,-1,1,0,0,0,0,0], [1,1,-1,0,0,0,0,0],[0,0,-1,0,0,0,0,0],
                    [0,0,0,1,-1,0,0,0], [0,0,0,-1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,-1,0,0,0], [0,-1,0,-1,0,1,0,0], 
                    [0,1,0,1,0,-1,0,0], #R_C_HSPR_MMP_dec2 = dissociation of the complex to form free HSPR and MMP -> dissociation
                    [0,1,0,0,1,-1,0,0], #R_C_HSPR_MMP_dec2 = dissociation of the complex to form free HSPR and FMP -> refolding step
                    [0,0,0,0,0,-1,0,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
                    [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,-1]]

            Outcome = random.choices(Stoich, weights = listR, k=1)
            #print(f"listM before: {listM}")
            #print(f"outcome: {Outcome} \n {Outcome[0]}")
            listM = listM+Outcome[0] ### why does it add term-by-term??? -> because listM is a np.array
            #print(f"listM after: {listM}")
            #exit()
            Time+=Tau # the new time the time before the step +the time to happen the next step ()
            counter += 1
            # print (Time,listM)
            listtime.append(Time) #this is to add stuff to the list
            listM2.append(listM)
        listM4.append(listM2)
        listtime2.append(listtime)
        end_time = Time
    return listM4, listtime2, numberofiteration, end_time

def gillespie_woA2_mp(param_dict, opt):
    listM = np.array([param_dict["init_HSFA1"], param_dict["init_HSPR"], param_dict["init_C_HSFA1_HSPR"], param_dict["init_MMP"], param_dict["init_FMP"], param_dict["init_C_HSPR_MMP"], param_dict["init_HSFB"]])
    a1 = param_dict['a1']
    a2 = param_dict['a2']
    #a3 = param_dict['a3']
    #a4 = param_dict['a4']
    a5 = param_dict['a5']
    a6 = param_dict['a6']
    a7 = param_dict['a7']
    #a8 = param_dict['a8']
    h1 = param_dict['h1']
    h2 = param_dict['h2']
    #h3 = param_dict['h3']
    #h4 = param_dict['h4']
    h5 = param_dict['h5']
    #h6 = param_dict['h6']
    c1 = param_dict['c1']
    c3 = param_dict['c3']
    d1 = param_dict['d1']
    d3 = param_dict['d3']
    Decay1 = param_dict['Decay1']
    Decay2 = param_dict['Decay2']
    #Decay3 = param_dict['Decay3']
    Decay4 = param_dict['Decay4']
    Decay6 = param_dict['Decay6']
    Decay7 = param_dict['Decay7']
    Decay8 = param_dict['Decay8']
    Decay5 = param_dict['Decay5']
    leakage = param_dict['leakage']
    n = param_dict['hillcoeff']
    Stoich = [[1,0,0,0,0,0,0], #R_HSFA1_inc
          [-1,0,0,0,0,0,0], #R_HSFA1_dec
          [0,1,0,0,0,0,0], #R_HSPR_inc
          [0,-1,0,0,0,0,0], #R_HSPR_dec
          [-1,-1,1,0,0,0,0], #R_C_HSFA1_HSPR_inc
          [1,1,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec1
          [0,0,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec2
          [0,0,0,1,-1,0,0], #R_MMP_inc
          [0,0,0,-1,0,0,0], #R_MMP_dec
          [0,0,0,0,1,0,0], #R_FMP_inc
          [0,0,0,0,-1,0,0], #R_FMP_dec
          [0,-1,0,-1,0,1,0], #R_C_HSPR_MMP_inc
          [0,1,0,1,0,-1,0], #R_C_HSPR_MMP_dec1 = dissociation of the complex to form free HSPR and MMP
          [0,1,0,0,1,-1,0], #R_C_HSPR_MMP_dec2 = refolding step, dissociation of the complex to form free HSPR and FMP 
          [0,0,0,0,0,-1,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
          [0,0,0,0,0,0,1], #R_HSFB_inc
          [0,0,0,0,0,0,-1] #R_HSFB_dec
          ]

    listM2 =[listM]
    Time=0
    listtime =[Time]
    counter = 0

    while Time < int(opt.tsp): 
        HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM
        if counter % 5000 ==0 and counter != 0:
            print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')
        if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
        else: d4 = param_dict['d4_norm']
            
        #HSFa1 andHSFA2 may makes complex
        #increase in HSFA1 by transcription and dessociation from the complex C_HSFA1_HSPR
        R_HSFA1_inc=leakage+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n) # + d1*C_HSFA1_HSPR
        #decrease in HSFA1 by association to the 1st complex C_HSFA1_HSPR and decay in the protein
        R_HSFA1_dec= Decay1*HSFA1
        #increase in HSPR by transcription and dess
        R_HSPR_inc= leakage+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n)
        #decrease in HSPR by transcription and dess **-> should be decay
        R_HSPR_dec= Decay2*HSPR
        #increase in C_HSFA1_HSPR association to the 1st complex
        R_C_HSFA1_HSPR_inc=c1*HSFA1*HSPR
        #decrease in C_HSFA1_HSPR dissociation from the 1st complex and degradation of the complex as a whole (?)
        R_C_HSFA1_HSPR_dec1=d1*C_HSFA1_HSPR
        R_C_HSFA1_HSPR_dec2=Decay7*C_HSFA1_HSPR
        #increase in MMP when 2nd complex decreases
        R_MMP_inc= d4*FMP
        #decrease in MMP by 2nd complex increases (may be we want to change the slope of dexcay later)
        R_MMP_dec= Decay5*MMP
        #increase in FMP by FMP production and MMP to FMP
        R_FMP_inc=a7  #how to write the production?
        #decrease in FMP by decay or
        R_FMP_dec= Decay6*FMP
        #increase in HSPR_MMP by association to the 2nd complex
        R_C_HSPR_MMP_inc=c3*HSPR*MMP #how to write the production?
        #decrease in HSPR_MMP by dissociation from the 2nd complex
        R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP
        R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP
        R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP
        #increase in HSFB by transcription with TF HSFA1 and HSFB
        R_HSFB_inc=leakage+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
        #decrease in HSFB by transcription and dess
        R_HSFB_dec=Decay4*HSFB
        listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec])
        TotR = sum(listR) #production of the MRNA 
        Rn = random.random() #getting random numbers
        Tau=-math.log(Rn)/TotR #when the next thing happen
        #Rn2= random.uniform(0,TotR) # for the next random number
        # HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB

        Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec'])
        print(Stoich_df)
        exit()
        Outcome = random.choices(Stoich, weights = listR, k=1)
        #print(f"listM before: {listM}")
        #print(f"outcome: {Outcome} \n {Outcome[0]}")
        listM = listM+Outcome[0] ### why does it add term-by-term??? -> because listM is a np.array
        print(f"listM after: {listM}")
        #exit()
        last_time = Time
        Time+=Tau # the new time the time before the step +the time to happen the next step ()
        counter += 1
        # print (Time,listM)
        if int(Time) == int(last_time) + opt.spf:
            listtime.append(Time) #this is to add stuff to the list
            listM2.append(listM)
    end_time = Time
    return listM2, listtime, end_time

def parallel_gillespie_woA2(param_dict, opt):
    numberofiteration = param_dict["numberofiteration"]
    arg = [(param_dict,opt)]*opt.thr
    #print(arg)
    with mp.Pool(processes= opt.thr) as pool:
        results = pool.starmap(gillespie_woA2_mp, arg)
    listM4 = []
    listtime2 = []
    listM6 = []
    listM7 = []
    all_end_time = []
    for i, iter_result in enumerate(results):
        listM2, listtime, end_time = iter_result
        all_end_time.append(end_time)
        listM4.append(listM2)
        listtime2.append(listtime)
    for time_step, conc_list in zip(listtime2, listM4):
        listM7 = [f"Iteration {i}"]+ [time_step] + conc_list
        listM6.append(listM7)
    param_dict['end_time'] = max(all_end_time)
    max_end_time = max(all_end_time)
    return listM6, numberofiteration, max_end_time

######################################

def gillespie(param_dict, opt):
    model_name = opt.mdn
    listM4, listtime2, rr_list2, numberofiteration =[], [], [], int(opt.nit)
    Stoich, Stoich_df, header = get_stoich(opt)
    a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d3, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n = unpack_param_dict(param_dict)
    if (opt.wa2) == True:
        Decay3, a4, h4, leakage_A2, a3, h3 = param_dict['Decay3'], param_dict['a4'], param_dict['h4'], param_dict['leakage_A2'], param_dict['a3'], param_dict['h3']
    if opt.mdn == 'replaceA1':
        c2, c4 = float(param_dict['c2']), float(param_dict['c4'])

    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
        listM = get_listM(opt, param_dict)
        listM2, Time, listtime, rr_list, counter = init_iter()

        while Time < int(opt.tsp): 
            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')

            d1, d4 = get_d1_d4(param_dict, Time, opt)
            if bool(opt.wa2) == False: 
                HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM
                listR = cal_basic_rate(a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d1, d3, d4, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n, HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB)
            else: ## with A2 = true
                HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB = listM
                listR = cal_basic_rate_withA2(a1, a2, a3, a4, a5, a6, a7, h1, h2, h3, h4, h5, c1, c3, d1, d3, d4, Decay1, Decay2, Decay3, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_A2, leakage_B, leakage_HSPR, n, HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB, opt)


            if opt.mdn == 'replaceA1':
                MMP_replace_A1HSPR = c2*C_HSFA1_HSPR*MMP
                A1_replace_MMPHSPR = c4*C_HSPR_MMP*HSFA1
                listR = np.append(listR, [MMP_replace_A1HSPR, A1_replace_MMPHSPR])

            listM, Time, counter, listtime, listM2, rr_list = update_save_tsp(listR, Stoich, listM, Time, counter, listtime, listM2, rr_list, opt)

        listM4, listtime2, rr_list2, end_time, param_dict = save_iter(listM2, listtime, rr_list, listM4, listtime2, rr_list2, Time, param_dict)
    return listM4, listtime2, numberofiteration, end_time, rr_list2, model_name


def unpack_param_dict(param_dict):
    a1 = float(param_dict['a1'])
    a2 = float(param_dict['a2'])
    a5 = float(param_dict['a5'])
    a6 = float(param_dict['a6'])
    a7 = float(param_dict['a7'])
    h1 = float(param_dict['h1'])
    h2 = float(param_dict['h2'])
    h5 = float(param_dict['h5'])
    c1 = float(param_dict['c1'])
    c3 = float(param_dict['c3'])
    #d1 = float(param_dict['d1'])
    d3 = float(param_dict['d3'])
    Decay1 = float(param_dict['Decay1'])
    Decay2 = float(param_dict['Decay2'])
    Decay4 = float(param_dict['Decay4'])
    Decay6 = float(param_dict['Decay6'])
    Decay7 = float(param_dict['Decay7'])
    Decay8 = float(param_dict['Decay8'])
    Decay5 = float(param_dict['Decay5'])
    try: 
        leakage_A1 = float(param_dict['leakage_A1'])
        leakage_B = float(param_dict['leakage_B'])
        leakage_HSPR = float(param_dict['leakage_HSPR'])
    except KeyError:
        leakage_A1, leakage_B, leakage_HSPR = float(param_dict['leakage']), float(param_dict['leakage']), float(param_dict['leakage'])
    n = int(param_dict['hillcoeff'])
    return a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d3, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n

def get_stoich(opt):
    if bool(opt.wa2) == False:
        if opt.mdn == 'woA2' or opt.mdn == 'd1upCons':
            Stoich = [[1,0,0,0,0,0,0], #R_HSFA1_inc
              [-1,0,0,0,0,0,0], #R_HSFA1_dec
              [0,1,0,0,0,0,0], #R_HSPR_inc
              [0,-1,0,0,0,0,0], #R_HSPR_dec
              [-1,-1,1,0,0,0,0], #R_C_HSFA1_HSPR_inc
              [1,1,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec1
              [0,0,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec2
              [0,0,0,1,-1,0,0], #R_MMP_inc
              [0,0,0,-1,0,0,0], #R_MMP_dec
              [0,0,0,0,1,0,0], #R_FMP_inc
              [0,0,0,0,-1,0,0], #R_FMP_dec
              [0,-1,0,-1,0,1,0], #R_C_HSPR_MMP_inc
              [0,1,0,1,0,-1,0], #R_C_HSPR_MMP_dec1 = dissociation of the complex to form free HSPR and MMP
              [0,1,0,0,1,-1,0], #R_C_HSPR_MMP_dec2 = refolding step, dissociation of the complex to form free HSPR and FMP 
              [0,0,0,0,0,-1,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
              [0,0,0,0,0,0,1], #R_HSFB_inc
              [0,0,0,0,0,0,-1] #R_HSFB_dec
              ]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec'])
        elif opt.mdn == 'replaceA1':
            Stoich = [[1,0,0,0,0,0,0], #R_HSFA1_inc
              [-1,0,0,0,0,0,0], #R_HSFA1_dec
              [0,1,0,0,0,0,0], #R_HSPR_inc
              [0,-1,0,0,0,0,0], #R_HSPR_dec
              [-1,-1,1,0,0,0,0], #R_C_HSFA1_HSPR_inc
              [1,1,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec1
              [0,0,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec2
              [0,0,0,1,-1,0,0], #R_MMP_inc
              [0,0,0,-1,0,0,0], #R_MMP_dec
              [0,0,0,0,1,0,0], #R_FMP_inc
              [0,0,0,0,-1,0,0], #R_FMP_dec
              [0,-1,0,-1,0,1,0], #R_C_HSPR_MMP_inc
              [0,1,0,1,0,-1,0], #R_C_HSPR_MMP_dec1 = dissociation of the complex to form free HSPR and MMP
              [0,1,0,0,1,-1,0], #R_C_HSPR_MMP_dec2 = refolding step, dissociation of the complex to form free HSPR and FMP 
              [0,0,0,0,0,-1,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
              [0,0,0,0,0,0,1], #R_HSFB_inc
              [0,0,0,0,0,0,-1], #R_HSFB_dec
              [1,0,-1,-1,0,1,0], #MMP_replace_A1HSPR
              [-1,0,1,1,0,-1,0] #A1_replace_MMPHSPR
              ]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
        else: print('opt.mdn empty or unexpected')
    elif bool(opt.wa2) == True:
        if opt.mdn == 'woA2' or opt.mdn == 'd1upCons':
            Stoich = [[1,0,0,0,0,0,0,0], 
                      [-1,0,0,0,0,0,0,0], 
                      [0,1,0,0,0,0,0,0], 
                      [0,-1,0,0,0,0,0,0], 
                      [-1,-1,1,0,0,0,0,0], 
                      [1,1,-1,0,0,0,0,0], 
                      [0,0,-1,0,0,0,0,0], 
                      [0,0,0,1,-1,0,0,0], 
                      [0,0,0,-1,0,0,0,0], 
                      [0,0,0,0,1,0,0,0], 
                      [0,0,0,0,-1,0,0,0], 
                      [0,-1,0,-1,0,1,0,0], 
                      [0,1,0,1,0,-1,0,0], 
                      [0,1,0,0,1,-1,0,0], 
                      [0,0,0,0,0,-1,0,0], 
                      [0,0,0,0,0,0,0,1], 
                      [0,0,0,0,0,0,0,-1],
                      [0,0,0,0,0,0,1,0], 
                      [0,0,0,0,0,0,-1,0],]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFA2', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3',  'R_HSFB_inc', 'R_HSFB_dec', 'R_HSFA2_inc', 'R_HSFA2_dec'])
        elif opt.mdn == 'replaceA1':
            Stoich = [[1,0,0,0,0,0,0,0], 
                      [-1,0,0,0,0,0,0,0], 
                      [0,1,0,0,0,0,0,0], 
                      [0,-1,0,0,0,0,0,0], 
                      [-1,-1,1,0,0,0,0,0], 
                      [1,1,-1,0,0,0,0,0], 
                      [0,0,-1,0,0,0,0,0], 
                      [0,0,0,1,-1,0,0,0], 
                      [0,0,0,-1,0,0,0,0], 
                      [0,0,0,0,1,0,0,0], 
                      [0,0,0,0,-1,0,0,0], 
                      [0,-1,0,-1,0,1,0,0], 
                      [0,1,0,1,0,-1,0,0], 
                      [0,1,0,0,1,-1,0,0], 
                      [0,0,0,0,0,-1,0,0], 
                      [0,0,0,0,0,0,0,1], 
                      [0,0,0,0,0,0,0,-1],
                      [1,0,-1,-1,0,1,0,0], #MMP_replace_A1HSPR
                      [-1,0,1,1,0,-1,0,0], #A1_replace_MMPHSPR
                      [0,0,0,0,0,0,1,0], 
                      [0,0,0,0,0,0,-1,0], 
                      ]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFA2', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3',  'R_HSFB_inc', 'R_HSFB_dec','R_HSFA2_inc', 'R_HSFA2_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
        else: print('opt.mdn empty or unexpected')
    header = ['Iteration_Identifier', 'time'] + Stoich_df.columns.to_list() + Stoich_df.index.to_list()
    return Stoich, Stoich_df, header

def get_listM(opt, param_dict):
    if bool(opt.wa2) == False:
        listM = np.array([int(param_dict["init_HSFA1"]),
                      int(param_dict["init_HSPR"]),
                      int(param_dict["init_C_HSFA1_HSPR"]),
                      int(param_dict["init_MMP"]),
                      int(param_dict["init_FMP"]),
                      int(param_dict["init_C_HSPR_MMP"]),
                      int(param_dict["init_HSFB"])])
    else:
        listM = np.array([param_dict["init_HSFA1"],
                          param_dict["init_HSPR"],
                          param_dict["init_C_HSFA1_HSPR"],
                          param_dict["init_MMP"],
                          param_dict["init_FMP"],
                          param_dict["init_C_HSPR_MMP"],
                          param_dict["init_HSFA2"],
                          param_dict["init_HSFB"]])
    return listM

def init_iter():
    listM2 =[]
    Time=0
    listtime =[]
    rr_list = []
    counter = 0
    return listM2, Time, listtime, rr_list, counter

def get_d1_d4(param_dict, Time, opt):
    hss, hsd = opt.hss, opt.hsd
    if Time >= int(hss) and Time <= int(hss) + int(hsd): 
        d4 = param_dict['d4_heat']
        if opt.mdn == 'd1upCons': 
            d1 = param_dict['d1_HS']
        else:
            d1 = param_dict['d1']
    elif bool(opt.hs2) == True:
        hs2 = opt.hs2
        if Time >= int(hs2) and Time <= int(hs2) + int(hsd): 
            d4 = param_dict['d4_heat']
            if opt.mdn == 'd1upCons': 
                d1 = param_dict['d1_HS']
            else:
                d1 = param_dict['d1']
        else: 
            d4 = param_dict['d4_norm']
            d1 = param_dict['d1']
    else: 
        d4 = param_dict['d4_norm']
        d1 = param_dict['d1']
    return d1, d4


def cal_basic_rate(a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d1, d3, d4, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n, HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB):
    R_HSFA1_inc=leakage_A1+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n) 
    R_HSFA1_dec= Decay1*HSFA1
    R_HSPR_inc= leakage_HSPR+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n)
    R_HSPR_dec= Decay2*HSPR
    R_C_HSFA1_HSPR_inc=c1*HSFA1*HSPR
    R_C_HSFA1_HSPR_dec1=d1*C_HSFA1_HSPR
    R_C_HSFA1_HSPR_dec2=Decay7*C_HSFA1_HSPR
    R_MMP_inc= d4*FMP
    R_MMP_dec= Decay5*MMP
    R_FMP_inc=a7
    R_FMP_dec= Decay6*FMP
    R_C_HSPR_MMP_inc=c3*HSPR*MMP
    R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP
    R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP
    R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP
    R_HSFB_inc=leakage_B+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
    R_HSFB_dec=Decay4*HSFB

    listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec])
    return listR

def cal_basic_rate_withA2(a1, a2, a3, a4, a5, a6, a7, h1, h2, h3, h4, h5, c1, c3, d1, d3, d4, Decay1, Decay2, Decay3, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_A2, leakage_B, leakage_HSPR, n, HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB, opt):

    R_HSFA1_inc=leakage_A1+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n) + a3*HSFA2**n/(h3**n+HSFA2**n+HSFB**n)
    R_HSFA1_dec= Decay1*HSFA1
    R_HSPR_inc= leakage_HSPR+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n) + a3*HSFA2**n/(h3**n+HSFA2**n+HSFB**n)
    R_HSPR_dec= Decay2*HSPR
    R_C_HSFA1_HSPR_inc=c1*HSFA1*HSPR
    R_C_HSFA1_HSPR_dec1=d1*C_HSFA1_HSPR
    R_C_HSFA1_HSPR_dec2=Decay7*C_HSFA1_HSPR
    R_MMP_inc= d4*FMP
    R_MMP_dec= Decay5*MMP
    R_FMP_inc=a7
    R_FMP_dec= Decay6*FMP
    R_C_HSPR_MMP_inc=c3*HSPR*MMP
    R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP
    R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP
    R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP
    R_HSFB_inc=leakage_B+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
    R_HSFB_dec=Decay4*HSFB
    R_HSFA2_inc=leakage_A2+a4*HSFA1**n/(h4**n+HSFA1**n+HSFB**n)
    R_HSFA2_dec=Decay3*HSFA2

    listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec,R_HSFA2_inc,R_HSFA2_dec])
    return listR

def update_save_tsp(listR, Stoich, listM, Time, counter, listtime, listM2, rr_list, opt):
    Tau = -math.log(random.random())/sum(listR) 
    #print(f'Stoich length = {len(Stoich)}, {Stoich}')
    #print(f'listR length = {len(listR)}, {listR}')

    Outcome = random.choices(Stoich, weights = listR, k=1)
    listM = listM+Outcome[0] 
    last_time = Time
    Time += Tau 
    counter += 1

    if "{:.1f}".format(Time) == "{:.1f}".format(last_time + opt.spf) or Time == 0:
        listtime.append("{:.1f}".format(Time)) #this is to add stuff to the list
        listM2.append(listM)
        rr_list.append(listR)
    return listM, Time, counter, listtime, listM2, rr_list

def save_iter(listM2, listtime, rr_list, listM4, listtime2, rr_list2, Time, param_dict):
    listM4.append(listM2)
    listtime2.append(listtime)
    rr_list2.append(rr_list)
    end_time = Time
    param_dict['end_time'] = end_time
    return listM4, listtime2, rr_list2, end_time, param_dict

###########################################################
## 4. Combine and save data
###########################################################


def combine_data(listtime2, listM4, rr_list2, opt):
    #to combine the data 
    listM6 = []
    listM7 = []
    # listM = list of protein conc at a single time point
    # listM2 = list of listM, storing conc at each time point in a single iteration
    # listM4 =list of listM2, storing different iterations
    for Iteration_Identifier, (time_list, iter_conc_list, rate_list) in enumerate(zip(listtime2, listM4, rr_list2)):
        #print(f"ratelist length {len(rate_list)}")
        #print(f"listM2 length {len(iter_conc_list)}")
        for time_step, conc_list, listR in zip(time_list[:-1], iter_conc_list[:-1],rate_list):
            listM7 = [f"Iteration {Iteration_Identifier}"]+ [time_step] + conc_list.tolist() + listR.tolist()
            #print(f"listM7: {listM7}")
            #print(f"conc_list: {conc_list.tolist()}")
            #print(f"listR: {len(listR)}")
            listM6.append(listM7)
    return listM6


def saveGilData(list, param_dict, data_dir, varData_dir, param_rootdir, numberofiteration, end_time, model_name, opt): ## function not in use
    date = datetime.now().date()
    if bool(opt.ids) == False: #de novo
        data_file = f"{data_dir}/{model_name}_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_withA2-{bool(opt.wa2)}"
        saveParam(param_dict, data_dir, numberofiteration, end_time, model_name, opt)
    elif bool(re.search(re.compile(r'/'), opt.ids)) == True: #input from SA param:
        data_file = f"{param_rootdir}/{opt.ids}_SimuData_{date}_{opt.mdn}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
    elif bool(re.search(re.compile(r'Para'), opt.ids)) == True: ## imported param from simuData
        data_file = f"{opt.para_csv_name[:-4].replace('Para', 'SimuData')}-rerunon{date}_numIter{numberofiteration}"
    else: ## variance analysis
        data_file = f"{varData_dir}/{opt.ids}_{model_name}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}"
    if bool(opt.hs2) == True:
        data_file = data_file + f"_hs2-{opt.hs2}"
    data_file = data_file + '.csv'
    data_file = get_unique_filename(data_file)
    print(data_file)

    # Open the CSV file in write mode with the specified directory and file name
    Stoich, Stoich_df, header = get_stoich(opt)
    with open(data_file, 'w') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(list) #how different it is to use .writerow and .writerows
    return data_file



def saveParam(param_dict, data_dir, numberofiteration, end_time, model_name, opt):
    date = datetime.now().date()
    if bool(opt.hs2) == True:
        param_name = f"{data_dir}/{model_name}_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSstart2{opt.hs2}_HSduration{opt.hsd}.csv"
    else:
        param_name = f"{data_dir}/{model_name}_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
    param_outfile = get_unique_filename(param_name)
    header = param_dict.keys()
    with open(param_outfile, 'w', newline='') as csvfile_2:
        # Create a CSV writer object
        writer = csv.DictWriter(csvfile_2, fieldnames=header)
        # Write the header
        writer.writeheader()
        # Write the parameter values
        writer.writerow(param_dict)
    #if opt.ofm =="pcl":
    #    param_name = f"{data_dir}/Exp3_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.pcl"
    #    param_outfile = get_unique_filename(param_name)
    #    saveData(param_dict, param_outfile)
    print(f" Parameters Saved as {opt.ofm}")


def loadData(fname):
    ## load data with pickle
    pcl_file = os.path.join(f"{fname}")
    with open(pcl_file, "r+b") as pcl_in:
        pcl_data = pickle.load(pcl_in)
    return pcl_data

def saveData(pcl_data, fname):
    ## save data with pickle
    pcl_file = os.path.join(f"{fname}")
    with open(pcl_file, "w+b") as pcl_out:
        pickle.dump(pcl_data, pcl_out , protocol=4)


def get_unique_filename(base_filename):
    counter = 1
    new_filename = base_filename
    # Keep incrementing the counter until a unique filename is found
    while os.path.exists(new_filename):
        counter += 1
        filename, extension = os.path.splitext(base_filename)
        new_filename = f"{filename}-run{counter}{extension}"
    return new_filename


######################################################################
## 5. Plot results
######################################################################


def plot_hpcSimuOutcome(listM6, data_file, param_dict, opt):
    Stoich, Stoich_df, header = get_stoich(opt)
    data_df, grouped_data = Gillespie_list_to_df(header, listM6, opt)
    path, filename = os.path.split(data_file)
    print(f"plot name suffix = {filename[:-4]}")
    plot_results(Stoich_df, path, filename[:-4], data_df, grouped_data, param_dict, opt)


def Gillespie_list_to_df(header, listM6, opt):
    data_df = pd.DataFrame(listM6, columns = header)
    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    data_df['totalA1'] = data_df['HSFA1'] + data_df['C_HSFA1_HSPR']
    conc_list = ['HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','totalHSPR', 'totalA1']
    for column in header:
        if column == 'Iteration_Identifier': data_df[column] = data_df[column].astype(str)
        elif column in conc_list: data_df[column] = data_df[column].astype(int)
        else: data_df[column] = data_df[column].astype(float)
    grouped_data = data_df.groupby('Iteration_Identifier')
    return data_df, grouped_data



def plot_results(Stoich_df, param_dir, data_name, data_df, grouped_data, param_dict, opt):

    numberofiteration= param_dict['numberofiteration']
    hss = opt.hss
    hsd = opt.hsd

    conc = Stoich_df.columns.to_list()
    conc =['HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','totalHSPR', 'totalA1']
    rates = Stoich_df.index.to_list()

    #plotReactionRate(rates, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, opt)
    plot_FMPMMPvsTime_2(conc, data_df, grouped_data, param_dir, numberofiteration,data_name, hss, hsd, opt)
    plot_FMPMMP_zoom(conc, data_df, grouped_data, param_dir, numberofiteration,data_name, hss, hsd, opt)




def plotReactionRate(rates, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, opt):
    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(ax, data_df, 'time', rates, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):
            plot_trajectory(ax, group_data, 'time', rates, hss, hsd, Iteration_Identifier = Iteration_Identifier)

    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, data_name, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()
    saveFig(param_dir, data_name, opt, prefix =f'ReactionRate')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2(conc, data_df, grouped_data, param_dir, numberofiteration,data_name, hss, hsd, opt):

    print(" Plot trajectories of Proteins and Regulators")
    protein = ['FMP','MMP']
    reg = list(set(conc) - set(protein))

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0], data_df, 'time', protein, hss, hsd, "iteration 0")
        plot_trajectory(ax[1], data_df, 'time', reg, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], group_data, 'time', protein, hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], group_data, 'time', reg, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, data_name, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(param_dir, data_name, opt, prefix =f'ProReg2')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMP_zoom(conc, data_df, grouped_data, param_dir, numberofiteration,data_name, hss, hsd, opt):
    print(" Zoomed In Protein & Regulator Trajectory Around HeatShock")
    cut_data_df = data_df[(data_df['time'] >= hss -50) & (data_df['time'] <= hss + hsd + 100)]
    reg_conc_col = list(set(conc) - set(['FMP','MMP']))
    
    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0], cut_data_df, 'time', ['FMP','MMP'], hss, hsd, "iteration 0")
        plot_trajectory(ax[1], cut_data_df, 'time', reg_conc_col, hss, hsd, "iteration 0")
    else:
        grouped_data = cut_data_df.groupby('Iteration_Identifier')
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], group_data, 'time', ['FMP','MMP'], hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], group_data, 'time', reg_conc_col, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    fig.text(0.5, 0.99, data_name, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    #fig.suptitle('Zoomed In Trajectories, Around HeatShock')
    plt.tight_layout()

    saveFig(param_dir, data_name, opt, prefix ='ProRegZoom')
    if bool(opt.shf) == True: plt.show()
    plt.close()

def plot_trajectory(ax, data_df, x_col, y_col_list, hss, hsd, Iteration_Identifier):
    for y_col in y_col_list:
        ax.plot(data_df[x_col], data_df[y_col], label=y_col, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Protein copy number')
    ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.2)
    ax.set_title(f"{Iteration_Identifier}")
    ax.legend(loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def saveFig(plot_dir, name_suffix, opt, prefix):
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/{prefix}_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        #print(f" save figure {opt.sfg == True}")



##################
## parser
################################################################################

class options(object):
    def __init__(self, **data):
        self.__dict__.update((k,v) for k,v in data.items())
    def plot(self, sep):
        ldat = sep.join([f"{var}" for key,var in vars(self).items()])
        return ldat

if __name__ == "__main__":

    ############################################################################
    ## get time and save call
    sscript = sys.argv[0]
    start_time = time.time()
    current_time = time.strftime('%x %X')
    scall = " ".join(sys.argv[1:])
    with open(f"{sscript}.log", "a") as calllog:
        calllog.write(f"Start : {current_time}\n")
        calllog.write(f"Script: {sscript}\n")
        calllog.write(f"Call  : {scall}\n")
    print(f"Call: {scall}")
    print(f"Status: Started at {current_time}")
    ############################################################################
    ## transform string into int, float, bool if possible
    def trans(s):
        if isinstance(s, str):
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            try: return int(s)
            except ValueError:
                try: return float(s)
                except ValueError:
                    if s in ["True", "False"]: return s == "True"
                    else: return s
        else: return s
    ############################################################################
    ## save documentation
    rx_text = re.compile(r"\n^(.+?)\n((?:.+\n)+)",re.MULTILINE)
    rx_oneline = re.compile(r"\n+")
    rx_options = re.compile(r"\((.+?)\:(.+?)\)")
    help_dict, type_dict, text_dict, mand_list = {}, {}, {}, []
    for match in rx_text.finditer(script_usage):
        argument = match.groups()[0].strip()
        text = " ".join(rx_oneline.sub("",match.groups()[1].strip()).split())
        argopts = {"action":"store", "help":None, "default":None, "choices":None}
        for option in rx_options.finditer(text):
            key = option.group(1).strip()
            var = option.group(2).strip()
            if var == "False": argopts["action"] = "store_true"
            if var == "True": argopts["action"] = "store_false"
            if key == "choices": var = [vs.strip() for vs in var.split(",")]
            if key == "default": var = trans(var)
            argopts[key] = var
        if argopts["default"]: add_default = f" (default: {str(argopts['default'])})"
        else: add_default = ""
        argopts["help"] = rx_options.sub("",text).strip()+add_default
        argnames = argument.split(",")
        if len(argnames) > 1:
            if argopts["default"] == None:
                mand_list.append(f"{argnames[1][1:]}")
            type_dict[f"{argnames[1][1:]}"] = argopts["default"]
            argopts["argshort"] = argnames[1]
            help_dict[argnames[0]] = argopts
        else:
            text_dict[argnames[0]] = argopts["help"]
    ############################################################################
    ## get arguments
    if text_dict["dependencies"]:
        desc = f"{text_dict['description']} (dependencies: {text_dict['dependencies']})"
    else:
        desc = text_dict['description']
    p = ap.ArgumentParser(prog=sscript, prefix_chars="-", usage=text_dict["usage"],
                          description=desc, epilog=text_dict["reference"])
    p.add_argument("-v", "--version", action="version", version=text_dict["version"])
    for argname,argopts in help_dict.items():
        argshort = argopts["argshort"]
        if argopts["choices"]:
            p.add_argument(argshort, argname,            dest=f"{argshort[1:]}",\
                           action=argopts["action"],     help=argopts["help"],\
                           default=argopts["default"],   choices=argopts["choices"])
        else:
            p.add_argument(argopts["argshort"], argname, dest=f"{argshort[1:]}",\
                           action=argopts["action"],     help=argopts["help"],\
                           default=argopts["default"])
    p._optionals.title = "arguments"
    opt = vars(p.parse_args())
    ############################################################################
    ## validate arguments
    if None in [opt[mand] for mand in mand_list]:
        print("Error: Mandatory arguments missing!")
        print(f"Usage: {text_dict['usage']} use -h or --help for more information.")
        sys.exit()
    for key,var in opt.items():
        if key not in mand_list:
            arg_req, arg_in = type_dict[key], trans(var)
            if type(arg_req) == type(arg_in):
                opt[key] = arg_in
            #else:
            #    print(f"Error: Argument {key} is not of type {type(arg_req)}!")
            #    sys.exit()
    #############################################################################
    ## add log create options class
    opt["log"] = True
    copt = options(**opt)
    ############################################################################
    ## call main function
    try:
        #saved = main(opt)
        saved = main(copt)
    except KeyboardInterrupt:
        print("Error: Interrupted by user!")
        sys.exit()
    except SystemExit:
        print("Error: System exit!")
        sys.exit()
    except Exception:
        print("Error: Script exception!")
        traceback.print_exc(file=sys.stderr)
        sys.exit()
    ############################################################################
    ## finish
    started_time = current_time
    elapsed_time = time.time()-start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    current_time = time.strftime('%x %X')
    if saved:
        with open(f"{sscript}.log", "a") as calllog,\
             open(os.path.join(saved,f"call.log"), "a") as dirlog:
            calllog.write(f"Save  : {os.path.abspath(saved)}\n")
            calllog.write(f"Finish: {current_time} in {elapsed_time}\n")
            ## dirlog
            dirlog.write(f"Start : {started_time}\n")
            dirlog.write(f"Script: {sscript}\n")
            dirlog.write(f"Call  : {scall}\n")
            dirlog.write(f"Save  : {os.path.abspath(saved)}\n")
            dirlog.write(f"Finish: {current_time} in {elapsed_time}\n")
    print(f"Status: Saved at {saved}")
    print(f"Status: Finished at {current_time} in {elapsed_time}")
    sys.exit(0)

