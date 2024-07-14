#!/usr/bin/env python
import os
import sys
import pathlib
import argparse
import nibabel as nib
import numpy as np
import brainspace as bs
import subprocess as sp 
from nilearn import signal
from scipy import spatial
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from brainspace.gradient import GradientMaps
from brainspace.gradient.embedding import diffusion_mapping
from scipy.spatial.distance import pdist, squareform
import sys
from mapalign import embed
from utils import *
print('imports are good')



parser = argparse.ArgumentParser(description='Embeds functional Connectivity Matrix using PCA and diffusion map embedding, and then determines how far peaks are across sessions of the same subject',\
	usage='pkReliability.py --subj <HCP subject> --odir <output directory> ',\
	epilog=("Example usage: "+"pkReliability.py --subj 100206 --odir /well/margulies/projects/pks"),add_help=True)


if len(sys.argv) < 2:
	parser.print_help()
	sys.exit(1)

req_grp = parser.add_argument_group(title='Required arguments')
req_grp.add_argument('--subj',type=str,metavar='',required=True,help='HCP subject to run on')
req_grp.add_argument('--odir',type=str,metavar='',required=True,help='Output directory base. Output will be saved as odir/subj/...')
req_grp.add_argument('--kernel',type=str,metavar='',required=True,help='specify smoothing kernel of time series')

op_grp= parser.add_argument_group(title='Optional arguments')
args=parser.parse_args()

#get arguments into variables
subj=args.subj
odir=args.odir
kernel=args.kernel

### set up subjects output directory 
odir=f'{odir}/{subj}'
print(odir)
print(kernel)
os.makedirs(odir,exist_ok=True)

# cluster_path='/well/margulies/users/mnk884/data20/'
cluster_path='/well/win-hcp/HCP-YA/subjectsAll'
###### set up files 
subjdir=f'{cluster_path}/{subj}' ### the folder containing the Structural and Rest folders. Change to match cluster when access given
fdir=f'{subjdir}/MNINonLinear/Results/'
anatdirNat=f'{subjdir}/T1w/Native/'
anatdir32=f'{subjdir}/T1w/fsaverage_LR32k/'
MNINonlinearDir=f'{subjdir}/MNINonLinear'


##### resting state time series paths 
ses1LR=f'{fdir}/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ses2LR=f'{fdir}/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ses1RL=f'{fdir}/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ses2RL=f'{fdir}/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'
func_ses=[ses1LR,ses2RL,ses1RL,ses2LR]


##### cifti indices 
lcort=slice(0,29696)
rcort=slice(29696, 59412)
cortAll=slice(0,59412)


#### left anatomical files
Lsrf32=f'{anatdir32}/{subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
LsrfNative=f'{anatdirNat}/{subj}.L.midthickness.native.surf.gii'
# Lsphere32=f'{anatdir32}/{subj}.L.sphere_MSMAll.32k_fs_LR.surf.gii'
# LsphereNat=f'{anatdirNat}/{subj}.L.sphere.reg.reg_LR.native.surf.gii'
# Laparc=f'{MNINonlinearDir}/fsaverage_LR32k/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'

#### right anatomical files 
Rsrf32=f'{anatdir32}/{subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
RsrfNative=f'{anatdirNat}/{subj}.R.midthickness.native.surf.gii'
# Rsphere32=f'{anatdir32}/{subj}.R.sphere.32k_fs_LR.surf.gii'
# RsphereNat=f'{anatdirNat}/{subj}.R.sphere.reg.reg_LR.native.surf.gii'
# Raparc=f'{MNINonlinearDir}/fsaverage_LR32k/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'

##### start doing the things ####

print('getting cortex info')
func_ses411=[]
for data in range(len(func_ses)):
	#### start by getting the indices of cortical vertices
	func_ses411.append(get_corticalVertices(func_ses[data]))
	##### smooth and clean the funcitonal time series

print(f'using kernel {kernel}')
### save the cifti encoded brainmodel information 
np.save(f'{odir}/{subj}.cifti.info.npy', func_ses411[0])
#### to load dict  back do as follows tst=np.load('.cifti.info.npy',allow_pickle=True) 

print('smooth and clean the functional time series')
for data in range(len(func_ses)):
	##### smooth and clean the funcitonal time series
	print(func_ses[data])
	func_ses[data]=wb_smoothCleanTs(subj,func_ses[data],kernel,Lsrf32,Rsrf32)


print('concatenating time series')

data=np.vstack(func_ses).T

data1=np.vstack([func_ses[0],func_ses[2]]).T

data2=np.vstack([func_ses[1],func_ses[3]]).T

del func_ses


print(data.shape)
print('the data type of the input is')
print(data.dtype)
print('######################')


print("generating cortical connectome")
print('full cort shape')
print(data[cortAll].shape)
rmat=np.corrcoef(data[cortAll])
print(rmat.shape)
print('correlation matrix done')


#np.save(f'{odir}/{subj}rmat.npy',rmat)

thresh=95
#### the first version of gradients is 90. this one is 95 and will be reflected in file output
thr=threshMat(rmat,thresh)
print('thresholding conn matrix to top 10% connectivity')
del rmat 


# Check for minimum value
print("Minimum value is %f" % thr.min())

# The negative values are very small, but we need to know how many nodes have negative values
# Count negative values per row
N = thr.shape[0]
neg_values = np.array([sum(thr[i,:] < 0) for i in range(N)])
print("Negative values occur in %d rows" % sum(neg_values > 0))

thr[thr < 0] = 0


aff=cosine_similarity(thr)
#aff = 1 -squareform(pdist(thr, metric='cosine'))
		    
del thr
print('is the affinity matrix symmetric?')
print(np.allclose(aff,aff.T))
print('#######################################')
print('')
print('affinity matrix built')
print(aff.shape)
#np.save(f'{odir}/{subj}CosAff.npy',aff)


print('running a quick little PCA')
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(aff)
np.save(f'{odir}/{subj}.pca.{thresh}.s0{kernel}mm.npy',pca.components_)
print('pca output has dimensions')
print(pca.components_.shape)


print('doing embedding with mapalign')
emb= embed.compute_diffusion_map(aff, alpha = 0.5,n_components=3)

np.save(f'{odir}/{subj}.mapalign.{thresh}.diffmaps.0{kernel}mm.npy',emb.T)
print('embedding run wihtout errors')

del aff 

#################################################################################
#################################################################################
#################################################################################
#REWRITE THIS WHOLE SECTOIN AS A FUNCTIONAL LATER 



print(data1.shape)
print('the data1 type of the input is')
print(data1.dtype)
print('######################')


print("generating cortical connectome SESSION 1")
print('full cort shape')
print(data1[cortAll].shape)
rmat=np.corrcoef(data1[cortAll])
print(rmat.shape)
print('correlation matrix done')


#np.save(f'{odir}/{subj}rmat.npy',rmat)


thr=threshMat(rmat,95)
print('thresholding conn matrix to top 10% connectivity')
del rmat 


# Check for minimum value
print("Minimum value is %f" % thr.min())

# The negative values are very small, but we need to know how many nodes have negative values
# Count negative values per row
N = thr.shape[0]
neg_values = np.array([sum(thr[i,:] < 0) for i in range(N)])
print("Negative values occur in %d rows" % sum(neg_values > 0))

thr[thr < 0] = 0


aff=cosine_similarity(thr)
#aff = 1 -squareform(pdist(thr, metric='cosine'))
		    
del thr
print('is the affinity matrix symmetric?')
print(np.allclose(aff,aff.T))
print('#######################################')
print('')
print('affinity matrix built')
print(aff.shape)
#np.save(f'{odir}/{subj}CosAff.npy',aff)


print('running a quick little PCA')
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(aff)
np.save(f'{odir}/{subj}.pca.{thresh}.ses1.s0{kernel}mm.npy',pca.components_)
print('pca output has dimensions')
print(pca.components_.shape)


print('doing embedding with mapalign')
emb= embed.compute_diffusion_map(aff, alpha = 0.5,n_components=3)

np.save(f'{odir}/{subj}.mapalign.{thresh}.ses1.diffmap.s0{kernel}mm.npy',emb.T)
print('embedding run wihtout errors')

del aff



print(data2.shape)
print('the data2 type of the input is')
print(data2.dtype)
print('######################')


print("generating cortical connectome SESSION 2")
print('full cort shape')
print(data2[cortAll].shape)
rmat=np.corrcoef(data2[cortAll])
print(rmat.shape)
print('correlation matrix done')


#np.save(f'{odir}/{subj}rmat.npy',rmat)


thr=threshMat(rmat,95)
print('thresholding conn matrix to top 10% connectivity')
del rmat 


# Check for minimum value
print("Minimum value is %f" % thr.min())

# The negative values are very small, but we need to know how many nodes have negative values
# Count negative values per row
N = thr.shape[0]
neg_values = np.array([sum(thr[i,:] < 0) for i in range(N)])
print("Negative values occur in %d rows" % sum(neg_values > 0))

thr[thr < 0] = 0


aff=cosine_similarity(thr)
#aff = 1 -squareform(pdist(thr, metric='cosine'))
		    
del thr
print('is the affinity matrix symmetric?')
print(np.allclose(aff,aff.T))
print('#######################################')
print('')
print('affinity matrix built')
print(aff.shape)
#np.save(f'{odir}/{subj}CosAff.npy',aff)


print('running a quick little PCA')
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(aff)
np.save(f'{odir}/{subj}.pca.{thresh}.ses2.s0{kernel}mm.npy',pca.components_)
print('pca output has dimensions')
print(pca.components_.shape)


print('doing embedding with mapalign')
emb= embed.compute_diffusion_map(aff, alpha = 0.5,n_components=3)

np.save(f'{odir}/{subj}.mapalign.{thresh}.ses2.s0{kernel}mm.diffmap.npy',emb.T)
print('embedding run wihtout errors')
