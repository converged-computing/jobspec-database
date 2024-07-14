#!/usr/bin/env python
# coding: utf-8


import os
import glob
import obspy
import numpy as np
import h5py
from obspy import UTCDateTime as UTC
import matplotlib.pyplot as plt
from scipy import signal
from obspy.signal.util import smooth
import time
import sys
sys.path.append('/home/yaolinm/Projects/Barcelona/ncf/')
from viens22 import *

index=int(sys.argv[1])

files_10hz=np.sort(glob.glob('/nfs/turbo/lsa-zspica/data/Florence/10HzData/*.h5'))
data=np.array(h5py.File(files_10hz[index],'r')['DAS'])
for i in range(len(data)):
    data[i,:]=preprocessing_trace(data[i,:],0.01,4,10,normalize=True)
    
out_dir='/scratch/zspica_root/zspica1/yaolinm/Florence/testing/10hz_10min_1000ch_30s/'
title=os.path.basename(files_10hz[index])[:-3]

chs=1000 ### 1000 virtual receivers
delta=10 ### sampling rate of 10hz
savetime=45 ### 30 seconds output on each side
sources=np.arange(950,1050,1) ### virtual sources

for source in sources:
    
    folder_name=MakeDir(out_dir+'ch'+str(source).zfill(4))
    
    df=np.zeros((chs,delta*savetime*2))
    
    for i in range(chs):
        df[i,:]=corr_fft(data[source],data[source+i],delta,savetime)
        
    np.save(folder_name+'/'+title+'.npy',df)