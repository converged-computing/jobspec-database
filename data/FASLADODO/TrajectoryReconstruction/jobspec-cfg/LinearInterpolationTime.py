#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import numpy as np
import pandas as pd
import os
import geopy.distance as distance
import random
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


'''
TripStatsData: will help to find trips with desired time interval.
TripMainData : will help to provide positional coordinated for trips.
'''
# import trip stats
with (open('/scratch/skp454/Trajectory/TaxiDataSet/data/20190709_TaxiTripStatsData.pkl', "rb")) as openfile:
    df_tripStat = pickle.load(openfile, encoding = 'latin1')

# import trip main
with (open('/scratch/skp454/Trajectory/TaxiDataSet/data/20190709_TaxiTripMainData.pkl', "rb")) as openfile:
    df_tripMain = pickle.load(openfile, encoding = 'latin1')


# In[3]:


def findError(row, df_tripMain, toTime):
    try:
        df_subMain = df_tripMain[(df_tripMain['trip_id'] == row['trip_id'])*(df_tripMain['Stationary'] == False)]
        df_subMain['Time'] = df_subMain['TimeDiff'].cumsum()
        t = row['TimeDiff'] - toTime
        index1 = list((df_subMain['Time']<t).index)
        random.shuffle(index1)
        df_subMain['Time'] = (df_subMain.index>index1[0]) * df_subMain['TimeDiff']
        df_subMain['Time'] = df_subMain['Time'].cumsum()
        df_subMain = df_subMain.loc[index1[0]:(df_subMain['Time']<toTime).index[-1]+1]
        df_subMain['predLAT'] = ((df_subMain.iloc[-1]['LAT'] - df_subMain.iloc[0]['LAT']) *                                   (df_subMain['Time']/df_subMain.iloc[-1]['Time'])) + df_subMain.iloc[0]['LAT']
        df_subMain['predLON'] = ((df_subMain.iloc[-1]['LON'] - df_subMain.iloc[0]['LON']) *                                   (df_subMain['Time']/df_subMain.iloc[-1]['Time'])) + df_subMain.iloc[0]['LON']
        df_subMain['Error'] = df_subMain.apply(lambda row: distance.geodesic((row['predLAT'],row['predLON'])
                                                                          ,(row['LAT'],row['LON'])).km, axis=1)
        return np.sum(df_subMain['Error'])/(len(df_subMain)-2)
    except:
        np.nan


# In[4]:


def errorSummary(fromTime, timeFrame, df_tripStat, df_tripMain):
    try:
        df_subStat = df_tripStat[(df_tripStat['TimeDiff']>fromTime)]
        index = list(df_subStat.index)
        random.shuffle(index)
        df_subStat = df_subStat.loc[index[0:100],:]
        toTime = fromTime + timeFrame
        df_subStat['Error'] = df_subStat.apply(lambda row: findError(row, df_tripMain, toTime), axis =1)
        return df_subStat['Error'].describe()
    except:
        return "error"


df_maskStat = {}
for dist in np.arange(0,3600,180):
    df_maskStat[dist] = errorSummary(dist, 180, df_tripStat, df_tripMain)

# save the statistics for linear interpolation
outfile = open('/scratch/skp454/Trajectory/TaxiDataSet/data/20190709_TaxiLinearInterpolationTimeShort.pkl','wb')
pickle.dump(df_maskStat,outfile)
outfile.close()

df_maskStat = {}
for dist in np.arange(0,18000,600):
    df_maskStat[dist] = errorSummary(dist, 600, df_tripStat, df_tripMain)

# save the statistics for linear interpolation
outfile = open('/scratch/skp454/Trajectory/TaxiDataSet/data/20190709_TaxiLinearInterpolationTimeLong.pkl','wb')
pickle.dump(df_maskStat,outfile)
outfile.close()


