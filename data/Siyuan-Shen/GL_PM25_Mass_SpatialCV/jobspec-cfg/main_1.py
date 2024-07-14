import csv
import numpy as np
import time
import gc
import os
from Spatial_CV.CV_Func import MultiyearMultiAreas_AVD_SpatialCrossValidation_CombineWithGeophysicalPM25,MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25_AllfoldsTogether_GBDAreas,MultiyearAreaModelCrossValid,plot_from_data, MultiyearMultiAreasSpatialCrossValidation, EachAreaForcedSlope_MultiyearMultiAreasSpatialCrossValidation, MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25, MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25_GBDAreas
from Spatial_CV.ConvNet_Data import Learning_Object_Datasets
from Spatial_CV.utils import *
from LRP_Func.Assemble import MultiyearAreaModelLRP
import toml



#######################################################################################
##                               Cross Validation SOP                                ##
## 1. Change Net.                                                                    ##
##    -> change the net in Saptial_CV.Net_Construction.py to desired Net.            ##
##    -> change the import parts in Spatial_CV.CV_Func, and the net line in Func     ##
##    {MultiyearAreaModelCrossValid}                                                 ##
##    -> check the special_name.                                                     ##
##                                                                                     
#######################################################################################
cfg = toml.load('./config.toml')
cfg_outdir = Config_outdir + '{}/Results/results-SpatialCV/'.format(version)
if not os.path.isdir(cfg_outdir):
    os.makedirs(cfg_outdir)

typeName = get_typeName()

total_time_start = time.time()
#######################################################################################
##                                   Initial Settings                                ##
#######################################################################################
YYYY = ['1998', '1999', '2000', '2001', '2002', '2003', '2004',
        '2005', '2006', '2007', '2008', '2009', '2010', '2011',
        '2012', '2013', '2014', '2015', '2016', '2017', '2018',
        '2019']
MM = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
channel_name = channel_names

'''
channel_name = ['EtaAOD_Bias','EtaCoastal','EtaMixing','EtaSGAOD_Bias','EtaSGTOPO_Bias',
                'AOD','ETA',
                'GC_PM25','GC_NH4','GC_SO4','GC_NIT','GC_SOA','GC_OC','GC_BC','GC_DST','GC_SSLT',
                'GeoPM25','Lat','Lon',
                'Landtype','Elevation',
                'BC_Emi','OC_Emi','DST_Emi','SSLT_Emi',
                'PBLH','T2M','V10M','U10M','RH', 'GRN', 'PRECTOT','Q850','PS'
                'Population',
                'Total_RoadDensity','Type1_RoadDensity','Type2_RoadDensity','Type3_RoadDensity','Type4_RoadDensity','Type5_RoadDensity'
                ]
'''
'''
channel_name = ['EtaAOD_Bias','EtaCoastal','EtaMixing','EtaSGAOD_Bias','EtaSGTOPO_Bias',
                'AOD','ETA',
                'GC_PM25','GC_NH4','GC_SO4','GC_NIT','GC_SOA','GC_OC','GC_BC','GC_DST','GC_SSLT',
                'GeoPM25','Lat','Lon',
                'Landtype','Elevation',
                'BC_Emi','OC_Emi','DST_Emi','SSLT_Emi',
                'PBLH','T2M','V10M','U10M','RH',
                'Population',
                'Total_RoadDensity','Type1_RoadDensity','Type2_RoadDensity','Type3_RoadDensity','Type4_RoadDensity','Type5_RoadDensity'
                ]
'''
#channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#channel_index = [29,9,28,8,23,24,15,18,27,4,5,17,12,11,1,2,16,26,13]
#channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,31]
# channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,33] #Met Extra
#normlized_channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21,22,23,24,25,26,27,28,29]
#channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29]
channel_index = channel_index
nchannel = len(channel_index)

train_infile = training_infile
#######################################################################################
##                             Input and output Directories                          ##
#######################################################################################

input_dir = ground_observation_data_dir



#######################################################################################
##                                Initial Arrays and Constants                       ##
#######################################################################################

kfold = kfold
repeats = repeats

num_epochs = epoch
batchsize = batchsize
learning_rate = lr0

#beginyear = [2001,2005,2010,2015]
#endyear = [2004,2009,2014,2019]
beginyear = beginyears
endyear = endyears
databeginyear = 1998
version = version
Area = training_area
special_name = special_name #_SigmoidMSELossWithGeoPenalties_alpha0d005_beta8d0_gamma3d0_lambda1-0d2' #'_exclude_longitude_landtype_GeoPenaltySum_constrain_alpha0d75_beta0d75_lambda1_0d5_lambda2_0d5'
extent_dic = extent_table()
extent = extent_dic[Area]

#########################################################
#                   Main Process Settings               #
#########################################################

MultiAreas = True
CV = Spatial_CrossValidation_Switch
OnlyCV_plot = True

ForcedSlopeUnity = ForcedSlopeUnity # True: force the slope to unity and offset to zero with Training datasets
EachAreaForcedSlopeUnity = False # True: force the slope to unity and offset to zero by each area; False: by global
EachMonthForcedSlopeUnity = EachMonthForcedSlopeUnity # True: force the slope to unity and offset to zero by each year, each month; False: by each month, all year average 

Combine_with_GeoPM25 = Combine_with_geophysical   #### For optimal model

LRP = False
LRP_Calculation = False
LRP_Plot =True


cfg_outfile = cfg_outdir + 'config_SpatialCV_{}_{}_{}Channel_{}x{}{}.csv'.format(typeName,version,nchannel,11,11,special_name)
f = open(cfg_outfile,'w')
toml.dump(cfg, f)
f.close()

availability, devices_number, devices_names, current_device = get_gpu_information()

if __name__ == '__main__':

    print('Train infile:',train_infile,'\nEpoch: ', num_epochs,'\n batchsize: ',batchsize,'\ninitial learning rate: ',learning_rate,
    '\nbeginyear: ', beginyear,'\nendyear: ',endyear,'\nversion:', version,'\nArea:', Area,'\nSpecial Name:', special_name,
    '\nbias:',bias,'\nNormalized PM2.5: ',normalize_species,'\nAbsolute PM2.5:', absolute_species,'\nLog PM2.5: ',log_species,
    '\nCV:',CV,'\nLRP:',LRP,'\nLRP Calculation:', LRP_Calculation,'\nLRP Plot:', LRP_Plot,'\n Channel INDEX: ', channel_index,'\nChannel Name: ', channel_name,
    '\nForcedSlopeUnity: ',ForcedSlopeUnity, '\nEachAreaForcedSlopeUnity:',EachAreaForcedSlopeUnity,'\nEachMonthForcedSlopeUnity:',EachMonthForcedSlopeUnity)

    
    if CV == True:
        CV_time_start = time.time()
        train_input = np.load(train_infile)
        true_input = Learning_Object_Datasets(bias=bias,unit_normalize_bias=unit_normalize_bias,Normalized_PM25Bias=normalize_bias,Normlized_PM25=normalize_species,Absolute_PM25=absolute_species,Log_PM25=log_species)
        txt_outfile = MultiyearMultiAreas_AVD_SpatialCrossValidation_CombineWithGeophysicalPM25(train_input=train_input, true_input=true_input, channel_index=channel_index,kfold=kfold,repeats=repeats,
                                                                                  extent=extent,num_epochs=num_epochs,batch_size=batchsize,learning_rate=learning_rate,Area=Area,version=version,special_name=special_name,
                                                                                  databeginyear=databeginyear,beginyear=beginyear,endyear=endyear
                                                                                  ,EachMonthSlopeUnity=EachMonthForcedSlopeUnity)
        
        del train_input,true_input
        gc.collect()
        CV_time_end = time.time()
        CV_time = CV_time_end - CV_time_start
        with open(txt_outfile,'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time for CV: {} seconds'.format(str(np.round(CV_time,4)))])
    if LRP == True:
        
        train_input = np.load(train_infile)
        true_input = Learning_Object_Datasets(bias=bias,unit_normalize_bias=unit_normalize_bias,Normalized_PM25Bias=normalize_bias,Normlized_PM25=normalize_species,Absolute_PM25=absolute_species,Log_PM25=log_species)
        MultiyearAreaModelLRP(train_input=train_input, true_input=true_input,
                        channel_index=channel_index, kfold=kfold, repeats=repeats,
                         extent=extent,
                         Area=Area,version=version,special_name=special_name,model_outdir=model_outdir,
                         databeginyear=databeginyear,beginyear=beginyear, endyear=endyear, bias=bias, Normlized_PM25=normalize_species, Absolute_Pm25=absolute_species,
                         Log_PM25=log_species,calculate=LRP_Calculation,plot=LRP_Plot)
        del train_input,true_input
        gc.collect()
    

    if OnlyCV_plot == True:
        typeName = get_typeName()
        Area = ['GL']
        data_indic =  Scatter_plots_outdir + '{}/data_recording/'.format(version)
        for iarea in Area:
            obs_pm25_outfile = data_indic + typeName+'_ObservationPM25_'+str(nchannel)+'Channel_'+iarea+'_Alltime'+special_name+'.npy'
            pre_pm25_outfile = data_indic + typeName+'_PredictionPM25_'+str(nchannel)+'Channel_'+iarea+'_Alltime'+special_name+'.npy'
            plot_from_data(infile=pre_pm25_outfile,true_infile=obs_pm25_outfile,Area=iarea,version=version,special_name=special_name,nchannel=nchannel,bias=bias,
        Normlized_PM25=normalize_species,Absolute_Pm25=absolute_species,Log_PM25=log_species)

total_time_end = time.time()

Total_time = total_time_end - total_time_start

