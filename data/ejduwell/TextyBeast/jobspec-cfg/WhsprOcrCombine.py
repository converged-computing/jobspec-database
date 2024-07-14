#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:37:31 2023

@author: eduwell
"""

#%% Import Packages
import os
import sys
import pandas as pd
import numpy as np
import copy

def combine(whsprFile,mmocrFile,outdir):
    
    # Read in input files
    #whsprDta = pd.read_csv(whsprFile)
    whsprDta = pd.read_table(whsprFile)
    mmocrDta = pd.read_csv(mmocrFile)
    
    tmpdf = pd.DataFrame(index=range(len(mmocrDta["txtDta"])),columns=range(1))
    tmpdf.rename(columns={0: "WhsprTxt"})
 
    for rowz in range(0,len(mmocrDta["txtDta"])):
        txtTmp=""
        frmStart = mmocrDta["startTime"][rowz]
        frmEnd = mmocrDta["endTime"][rowz]
        pause = ""
        for idx in range(0,len(whsprDta["start"])):
            tknStart = whsprDta["start"][idx]/1000
            if (tknStart>=frmStart) and (tknStart<=frmEnd):
                txtTmp = txtTmp+whsprDta["text"][idx]+" "
                
        
        tmpdf[0][rowz]=copy.deepcopy(txtTmp)
        pause ="";
        
    
    pause ="";
    dataOut = pd.concat([copy.deepcopy(mmocrDta),copy.deepcopy(tmpdf)],axis=1)
    dataOut = dataOut.rename(columns={0: "AudioText", "txtDta": "ImageFrameText", "Unnamed: 0": "UniqueFrame"})
    
    # Unpack the clusters of text to make them human readable again
    frame_itr = 0;
    #for frames in txt_out:
    for rowz in range(0,len(dataOut["ImageFrameText"])):
        frames_txt = dataOut["ImageFrameText"][frame_itr]
        #frmsCmd = "frames = frames_txt"
        frames = eval(frames_txt)
        txtTmp=""
        
        # Loop through txt_out and print contained text into output .txt file
        cntr = 0
        for txt_clusters in frames:
            for sub_clustrs in txt_clusters:
                if cntr > 0:
                    #f.write('\n')
                    #print("")
                    txtTmp=txtTmp+'\n'
                    
                for txt_strgs in sub_clustrs:
                    txt2read = txt_strgs
                    txtTmp = txtTmp+txt2read
                    #f.write(txt2read)
                    #print(txt2read)
                    txtTmp = txtTmp+'\n'
                    #f.write('\n')
            txtTmp = txtTmp+'\n'
            txtTmp = txtTmp+'\n'
            #f.write('\n')
            #f.write('\n')
            #print("")
            cntr = cntr+1
        
        dataOut["ImageFrameText"][frame_itr] = txtTmp
        frame_itr = frame_itr+1

    pause = ""
    
    # generate output file name
    mmocrFile2 = os.path.basename(mmocrFile)
    outFileName = os.path.splitext(mmocrFile2)[0]+"-whsprTxt"+".csv"
    
    # Save dataOut as a .csv 
    dataOut.to_csv(outdir+"/"+outFileName);
    
    

#%% Get Input Variables and Run

#whsprFile="/Users/eduwell/python_projects/data_scraping/output-03-09-23-13-52-17-356728779/out_dir/audio_speech_dta/test4_trimmed.wav.tsv"
whsprFile=sys.argv[1];

#mmocrFile="/Users/eduwell/python_projects/data_scraping/output-03-09-23-13-52-17-356728779/out_dir/video_img_dta/test4_trimmed_ufTxt-Time.csv"
mmocrFile=sys.argv[2];

#outdir = "/Users/eduwell/python_projects/data_scraping/output-03-09-23-13-52-17-356728779/out_dir/"
outdir=sys.argv[3];

# Run It
combine(whsprFile,mmocrFile,outdir)