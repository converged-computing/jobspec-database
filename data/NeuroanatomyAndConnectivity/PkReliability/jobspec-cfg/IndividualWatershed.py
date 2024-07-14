#! /usr/bin/env python
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

SubjectIn=sys.argv[1]

##### this is the watershed function we call 
def cortiGradWS(W,D,m,method):
    ##### adapted from eyal soreq's matlab code
    #### W is masked metric file to use in watershed
    #### D is masked distance matrix for a previously specified radius
    m=m
    #### get the order of the elements of local minima or the maximum depending on method provided
    idx=np.argsort(W)
    if method =='min':
        idx=idx
    elif method =='max':
        idx=idx[::-1]
    
#     #### need to go in order from most to least not on the distL alone
#     #### so reorder the distL to go by idx
    
    
    N= len(idx)
    C=np.zeros(W.shape)
    for ii in idx:
        ##### find where the nodes in your ROI are
        subNodes=np.where(D[ii]>0)[0]
        c=C[subNodes]
        c=c[c>0]
        
        ##### next part conditional
        
        if c.size == 0:
            C[ii]=m
            m=m+1
            # comment back in. this is to see if you can plot just the watershed itself
        elif ~np.any(np.diff(c)):
            C[ii]=c[0]
        else:
            C[ii]=mode(c)
    return C

    
def getSensory(label):
    data=nib.load(label).darrays[0].data
    
    A1=np.hstack([np.where(data==33)[0],np.where(data==75)[0]])
    S1=np.hstack([np.where(data==28)[0],np.where(data==46)[0]])
    V1=np.hstack([np.where(data==45)[0],np.where(data==43)[0]])
    return A1,V1,S1


def SensVals(subj,distArr,hemi):
    data=np.round(distArr)
    if hemi == 'L':
        A1,V1,S1=getSensory(subj.Laparc)
    elif hemi =='R':
        A1,V1,S1=getSensory(subj.Raparc)
    A1Vals=data[A1]
    S1Vals=data[S1]
    V1Vals=data[V1]
#     print(A1Vals.shape,S1Vals.shape,V1Vals.shape)
    
    equi=reduce(np.intersect1d,[A1Vals,S1Vals,V1Vals])
    distVals=np.hstack([A1Vals,S1Vals,V1Vals])
#     equi=np.intersect1d(A1Vals,S1Vals)
#     equi=np.intersect1d(equi,V1Vals)
#     print(equi)
    if equi.shape[0] <1:
        print(f'subject {subj.subj} has no equidistant value from gradient mask to the {hemi} sensory cortex')
        val = 0
        highest=0
        valid=False
        return np.nan,np.nan,np.nan,np.nan
    else:
        equi=list(equi)
        val=[np.count_nonzero(distVals==val) for val in equi ]
        valid=True

        common=equi[np.argmax(val)]
        highest=np.max(equi)
        lowest=np.min(equi)
        med=np.median(equi)
#             print(f'the common is {common},max is {highest},min is {lowest}, median ins {np.round(med)}')
        return common,highest,lowest,np.round(med)

def runWS(subject):
    
    subj_str=subject
    inst=hcp_subj(subj_str,4)
    
    
    Ldist=np.round(np.load(f'/well/margulies/projects/pkReliability/Dist2SensoryBorder/{subj_str}/L.top10toCort.npy'))
    Rdist=np.round(np.load(f'/well/margulies/projects/pkReliability/Dist2SensoryBorder/{subj_str}/R.top10t0Cort.npy'))
    
    keys=['Common','MaxEq','MinEq','MedEq']
    
    L=SensVals(inst,Ldist,'L')
    R=SensVals(inst,Rdist,'R')
    if np.isnan(L[0])==True or np.isnan(R[0])==True:
        pass
    else:
        
        L=dict(zip(keys,L))
        print(L)
        R=dict(zip(keys,R)) 
        print(R)
    
        outpath=f'tmp.{inst.subj}/'
        sp.run(f'mkdir -p {outpath}',shell=True)
        
        WS_outPath=f'/well/margulies/projects/pkReliability/Dist2SensoryBorder/{inst.subj}/WS_seg'
        sp.run(f'mkdir -p {WS_outPath}',shell=True)
        
        for key in L:
            cifti_out=f'{outpath}/{key}.L.0{int(L[key])}.dconn.nii'
            cmd=f'wb_command -surface-geodesic-distance-all-to-all {inst.Lsrf} {cifti_out} -limit {L[key]}'
            sp.run(cmd,shell=True)
            
            dist=nib.load(cifti_out).get_fdata()
            dist=dist[inst.Lfill,:][:,inst.Lfill]
            grad=inst.Lgrad[0][inst.Lfill]
            
            WS=cortiGradWS(grad,dist,1,'max')
            del dist
            
            cmd=f'rm  {cifti_out}'
            sp.run(cmd,shell=True)
            
            out=np.zeros(32492)
            out[inst.Lfill]=WS
            save_gifti(out,f'{WS_outPath}/L.{key}.0{int(L[key])}')
            
        for key in R:
            cifti_out=f'{outpath}/{key}.R.0{int(R[key])}.dconn.nii'
            cmd=f'wb_command -surface-geodesic-distance-all-to-all {inst.Rsrf} {cifti_out} -limit {R[key]}'
            sp.run(cmd,shell=True)
            
            dist=nib.load(cifti_out).get_fdata()
            dist=dist[inst.Rfill,:][:,inst.Rfill]
            grad=inst.Rgrad[0][inst.Rfill]
            
            WS=cortiGradWS(grad,dist,1,'max')
            del dist
            cmd=f'rm  {cifti_out}'
            sp.run(cmd,shell=True)
            
            sp.run(f'rm -rf {outpath}',shell=True)
            
            out=np.zeros(32492)
            out[inst.Rfill]=WS
            save_gifti(out,f'{WS_outPath}/R.{key}.0{int(R[key])}')


runWS(SubjectIn)
  
