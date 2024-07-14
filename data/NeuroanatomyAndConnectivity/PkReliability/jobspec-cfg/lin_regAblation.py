from hcp_class import hcp_subj
import numpy as np
from functools import reduce
import nibabel as nib
import pickle
import sys
import seaborn as sn
import subprocess as sp
import os
from utils import save_gifti
from nilearn.plotting import view_surf
from statistics import mode


from sklearn.linear_model import LinearRegression
import pandas as pd
import os

from surfdist import analysis,utils


subj=sys.argv[1]
out=f'/well/margulies/projects/pkReliability/DistLinRegression/{subj}/'
outpath=out
os.makedirs(out,exist_ok=True)

subj=hcp_subj(subj,4)
print(subj.subj)    


def check_gradOrientation(ID):
    subj=hcp_subj(ID,4)
    
    if subj.Lgrad[1] ==True and subj.Rgrad[1] == True:
        return 1
    else:
        return 0 

zones={'lPFC':1,'lPar':2,'mPFC':3,'vmPFC':4,'lTmp':5,'Insula':6,'mPar':7,'mTmp':8,'OP':9}


def ablateGradient(subj,thr,hemi):
    grad=subj.extract_topX(subj.Lgrad,subj.Rgrad,thr)
    L10,R10=subj.extract_topX(subj.Lgrad,subj.Rgrad,thr)
    
    Lsrf=(subj.Lcoords,subj.Lfaces)
    Rsrf=(subj.Rcoords,subj.Rfaces)
    
    ### choose watershed hemi###
    if hemi =='L':
        WS=nib.load('watershed_templates/LWS.28.max.label.gii').darrays[0].data
        grad=grad[0]
        
        ##### remove anything left in occipital pole peak from gradient
        OP=np.where(WS==9)[0]
        if len(np.intersect1d(OP,L10))>0:
            print('removing gradient values over threshold in occiptal pole')
            NO_OP_roi=L10[~np.isin(L10,OP)]
            L10=NO_OP_roi
        
        ### ground truth -- apex network 90% to rest of cortex
        gt=analysis.dist_calc(Lsrf,subj.Lfill,L10)
        ### ground truth -- add in medial temporal to apex network for full DMN representation
  
        mtmp=np.where(WS==8)[0]
        
        if len(np.intersect1d(mtmp,L10))==0:
 
            mtmp_thr=np.percentile(subj.Lgrad[0][mtmp],90)
            mtmp_mask=np.where(subj.Lgrad[0]>=mtmp_thr)[0]
            mtmp_mask=np.intersect1d(mtmp_mask,mtmp)

            fullDMN_REP=np.concatenate([L10,mtmp_mask])
            gt_dmn=analysis.dist_calc(Lsrf,subj.Lfill,fullDMN_REP)
        else:
            fullDMN_REP=L10
            gt_dmn=gt
            
        #### do the ablation 
        ablation={}
        ablation['groundTruth']=gt
        ZoneDist={}
        for key in zones:
            val=zones[key]
            ablate=np.where(WS==val)[0]
            if val !=9:
                roi=L10[~np.isin(L10,ablate)]
                ablation[key]=analysis.dist_calc(Lsrf,subj.Lfill,roi)

                peak=fullDMN_REP[np.isin(fullDMN_REP,ablate)]
                ZoneDist[key]=analysis.dist_calc(Lsrf,subj.Lfill,peak)
        ablation['thrInclMedtmp']=all(gt==gt_dmn)
        
        ablation_DMN={}
        ablation_DMN['groundTruth']=gt_dmn
        ZoneDist={}
        for key in zones:
            val=zones[key]
            ablate=np.where(WS==val)[0]
            if val !=9:
                roi=fullDMN_REP[~np.isin(fullDMN_REP,ablate)]
                ablation_DMN[key]=analysis.dist_calc(Lsrf,subj.Lfill,roi)

                peak=fullDMN_REP[np.isin(fullDMN_REP,ablate)]
                ZoneDist[key]=analysis.dist_calc(Lsrf,subj.Lfill,peak)
        ablation_DMN['thrInclMedtmp']=all(gt==gt_dmn)
        
        
                
        
    elif hemi =='R':
#         WS=nib.load('watershed_templates/RWS.28.max.label.gii').darrays[0].data
        WS=nib.load('watershed_templates/LWS.28.max.label.gii').darrays[0].data
        grad=grad[1]
##### remove anything left in occipital pole peak from gradient
        OP=np.where(WS==9)[0]
        if len(np.intersect1d(OP,R10))>0:
            print('removing gradient values over threshold in occiptal pole')
            NO_OP_roi=R10[~np.isin(R10,OP)]
            R10=NO_OP_roi
        
        ### ground truth -- apex network 90% to rest of cortex
        gt=analysis.dist_calc(Rsrf,subj.Rfill,R10)
        ### ground truth -- add in medial temporal to apex network for full DMN representation
  
        mtmp=np.where(WS==8)[0]
        
        if len(np.intersect1d(mtmp,R10))==0:
 
            mtmp_thr=np.percentile(subj.Rgrad[0][mtmp],90)
            mtmp_mask=np.where(subj.Rgrad[0]>=mtmp_thr)[0]
            mtmp_mask=np.intersect1d(mtmp_mask,mtmp)

            fullDMN_REP=np.concatenate([R10,mtmp_mask])
            gt_dmn=analysis.dist_calc(Rsrf,subj.Rfill,fullDMN_REP)
        else:
            fullDMN_REP=R10
            gt_dmn=gt
            
        #### do the ablation 
        ablation={}
        ablation['groundTruth']=gt
        ZoneDist={}


        for key in zones:
            val=zones[key]
            ablate=np.where(WS==val)[0]
            if val !=9:
                roi=R10[~np.isin(R10,ablate)]
                ablation[key]=analysis.dist_calc(Rsrf,subj.Rfill,roi)

                peak=R10[np.isin(R10,ablate)]
                ZoneDist[key]=analysis.dist_calc(Rsrf,subj.Rfill,peak)

        ablation['thrInclMedtmp']=all(gt==gt_dmn)
        
        ablation_DMN={}
        ablation_DMN['groundTruth']=gt_dmn
        ZoneDist={}
        
        for key in zones:
            val=zones[key]
            ablate=np.where(WS==val)[0]
            if val !=9:
                roi=fullDMN_REP[~np.isin(fullDMN_REP,ablate)]
                ablation_DMN[key]=analysis.dist_calc(Rsrf,subj.Rfill,roi)
                peak=fullDMN_REP[np.isin(fullDMN_REP,ablate)]
                ZoneDist[key]=analysis.dist_calc(Rsrf,subj.Rfill,peak)
        
        ablation_DMN['thrInclMedtmp']=all(gt==gt_dmn)
                
    return ablation,ablation_DMN,ZoneDist

def getSensory(label):
    data=nib.load(label).darrays[0].data 
    A1=np.hstack([np.where(data==33)[0],np.where(data==75)[0]])
    S1=np.hstack([np.where(data==28)[0],np.where(data==46)[0]])
    V1=np.hstack([np.where(data==45)[0]])#,np.where(data==43)[0] ####exclude the occipital pole
    return A1,S1,V1

def SensVals(subj,distArr,hemi):
    data=distArr
    if hemi == 'L':
        A1,S1,V1=getSensory(subj.Laparc)      
    elif hemi =='R':
        A1,S1,V1=getSensory(subj.Raparc)
   
    A1Vals=data[A1]
    S1Vals=data[S1]
    V1Vals=data[V1]
    

    equi=reduce(np.intersect1d,[np.round(A1Vals),np.round(S1Vals),np.round(V1Vals)])
#     distVals=np.hstack([A1Vals,S1Vals,V1Vals])
    ##### check that there is at least one equidistant value for this subject 
    if equi.shape[0] <1:
        return False,A1Vals,S1Vals,V1Vals
    else:
        return True,A1Vals,S1Vals,V1Vals


def influence2Sens(subj,data_dict,hemi):
    ids=list(data_dict.keys())
    globalRef=data_dict[ids[0]]
    norm_max=np.max(globalRef)
    print(norm_max)
    print('####')
    ref=SensVals(subj,data_dict[ids[0]],hemi)
    A1ref=ref[1]
    S1ref=ref[2]
    V1ref=ref[3]
    rsquared={}
    for key in ids[1:-1]:
        
        test_max=np.max(data_dict[key])
        print(key,test_max)
        mdl=LinearRegression().fit(data_dict[key].reshape(-1,1)/norm_max,globalRef.reshape(-1,1)/norm_max)
        g_score=mdl.score(data_dict[key].reshape(-1,1)/test_max,globalRef.reshape(-1,1)/norm_max)
        
        #### modality specific regressions now 
        equi,A1abl,S1abl,V1abl=SensVals(subj,data_dict[key],hemi)
        print(equi)
        
        A1mdl=LinearRegression().fit(A1abl.reshape(-1,1)/test_max,A1ref.reshape(-1,1)/norm_max)
        A1score=A1mdl.score(A1abl.reshape(-1,1)/test_max,A1ref.reshape(-1,1)/norm_max)
        
        S1mdl=LinearRegression().fit(S1abl.reshape(-1,1)/test_max,S1ref.reshape(-1,1)/norm_max)
        S1score=S1mdl.score(S1abl.reshape(-1,1)/test_max,S1ref.reshape(-1,1)/norm_max)
        
        V1mdl=LinearRegression().fit(V1abl.reshape(-1,1)/test_max,V1ref.reshape(-1,1)/norm_max)
        V1score=V1mdl.score(V1abl.reshape(-1,1)/test_max,V1ref.reshape(-1,1)/norm_max)
        rsquared[key]=[g_score,A1score,S1score,V1score,equi]
    rsquared=pd.DataFrame.from_dict(rsquared)
    rsquared=rsquared.T
    rsquared=rsquared.rename(columns={0:'cortex',1:'A1',2:'S1',3:'V1',4:'equi'})
        
    return rsquared
    
        
        
### do the left
### measure distance
abl_L,abl_dmn_L,ZdistsL=ablateGradient(subj,90,'L')

l1=influence2Sens(subj,abl_L,'L')
l2=influence2Sens(subj,abl_dmn_L,'L')

### save output 
l1.to_csv(f'{outpath}/{subj.subj}.L.ablation.csv',sep=',')
l2.to_csv(f'{outpath}/{subj.subj}.L.ablation.DMN.csv',sep=',')


abl_L=pd.DataFrame.from_dict(abl_L)
abl_L.to_csv(f'{outpath}/{subj.subj}.L.ablation.Distances.csv',sep=',')
abl_dmn_L=pd.DataFrame.from_dict(abl_dmn_L)
abl_dmn_L.to_csv(f'{outpath}/{subj.subj}.L.ablation.DMN.Distances.csv',sep=',')
ZdistsL=pd.DataFrame.from_dict(ZdistsL)
ZdistsL.to_csv(f'{outpath}/{subj.subj}.L.ZoneDists.csv',sep=',')


### linear regress


## do the right 
### measure dist
abl_R,abl_dmn_R,ZdistsR=ablateGradient(subj,90,'R')

### linear regress
r1=influence2Sens(subj,abl_R,'R')
r2=influence2Sens(subj,abl_dmn_R,'R')

r1.to_csv(f'{outpath}/{subj.subj}.R.ablation.csv',sep=',')
r2.to_csv(f'{outpath}/{subj.subj}.R.ablation.DMN.csv',sep=',')


abl_R=pd.DataFrame.from_dict(abl_R)
abl_R.to_csv(f'{outpath}/{subj.subj}.R.ablation.Distances.csv',sep=',')
abl_dmn_R=pd.DataFrame.from_dict(abl_dmn_R)
abl_dmn_R.to_csv(f'{outpath}/{subj.subj}.R.ablation.DMN.Distances.csv',sep=',')

ZdistsR=pd.DataFrame.from_dict(ZdistsR)
ZdistsR.to_csv(f'{outpath}/{subj.subj}.R.ZoneDists.csv',sep=',')

