from hcp_class import *
import pickle
import surfdist
import subprocess as sp
import surfdist.analysis
import surfdist.utils
from surfdist.utils import find_node_match
from surfdist.utils import find_node_match
import surfdist as sd
import sys

subj=sys.argv[1]
print(f'calculating distance from gradient mask to  parcels for subject {subj}')

def DistFromGradMask(subj,threshold):
	subj_inst=hcp_subj(subj,4)
	grads=subj_inst.extract_topX(subj_inst.Lgrad,subj_inst.Rgrad,threshold)

	#### insert intersection to ensure peaks include parietal zone
	####mparietal zone == 7 
	LWS=nib.load('watershed_templates/LWS.28.max.label.gii').darrays[0].data
	LparZone=np.where(LWS==7)[0]
	LoccZone=np.where(LWS==9)[0]
	RWS=nib.load('watershed_templates/RWS.28.max.label.gii').darrays[0].data
	RparZone=np.where(RWS==7)[0]
	RoccZone=np.where(RWS==9)[0]

	Linter=np.intersect1d(LparZone,grads[0])
	Rinter=np.intersect1d(RparZone,grads[1])

	LinterOcc=np.intersect1d(LoccZone,grads[0])
	RinterOcc=np.intersect1d(RoccZone,grads[0])


	if Linter.size>0 and Rinter.size>0 and LinterOcc.size==0 and RinterOcc.size==0:
		print('subject gradient includes medial parietal zone')
		Lsurf=[subj_inst.Lcoords,subj_inst.Lfaces]
		Ldist=surfdist.analysis.dist_calc(Lsurf,subj_inst.Lfill,grads[0])
		Rsurf=[subj_inst.Rcoords,subj_inst.Rfaces]
		Rdist=surfdist.analysis.dist_calc(Rsurf,subj_inst.Rfill,grads[1])
		##### get the surface areas -- sqrt(sum of cortical vertex areas) 
		sp.run(f'wb_command -surface-vertex-areas {subj_inst.Lsrf} tmp/{subj}.L.area.func.gii',shell=True)
		Larea=np.sqrt(np.sum(nib.load(f'tmp/{subj}.L.area.func.gii').darrays[0].data[subj_inst.Lfill]))
		# sp.run('rm L.area.func.gii',shell=True)
		sp.run(f'wb_command -surface-vertex-areas {subj_inst.Rsrf} tmp/{subj}.R.area.func.gii',shell=True)
		Rarea=np.sqrt(np.sum(nib.load(f'tmp/{subj}.R.area.func.gii').darrays[0].data[subj_inst.Rfill]))
		# sp.run('rm R.area.func.gii',shell=True)
		print('if you see this message than it has been run succesffuly despite all the numba warnings from surfdist')
	else: 
		print('gradient for {subj} does not contain medial parietal zone')
	return [Ldist,Larea,Rdist,Rarea]

results=DistFromGradMask(subj,90)

with open(f"results/distancesFromGradMask/{subj}.DistFromGrad.pickle", "wb") as dist:
	pickle.dump(results, dist)