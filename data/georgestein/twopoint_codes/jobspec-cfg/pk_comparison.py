#/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as py
import numpy as np
import struct 
import sys
import os
import subprocess
#from mpi4py import MPI

#mpirank=MPI.COMM_WORLD.Get_rank()
#mpisize=MPI.COMM_WORLD.Get_size()

Lbox    = float(sys.argv[1])
nmesh   = int(sys.argv[2])
fmt     = int(sys.argv[3])
nxi     = int(sys.argv[4])
ncut    = int(sys.argv[5])
Lag     = int(sys.argv[6])
shuff   = int(sys.argv[7])
folder  = str(sys.argv[8]) 
dirout  = str(sys.argv[9]) 
nproc   = int(sys.argv[10])

dirout=dirout+"/"
def cut_cat(pp_cat,ncut,Lag):

        pkfile_in = open(pp_cat,"rb")
        Non       = np.fromfile(pkfile_in, dtype=np.int32, count=1)
        RTH       = np.fromfile(pkfile_in, dtype=np.float32, count=1)
	zin       = np.fromfile(pkfile_in, dtype=np.float32, count=1)

	outnum    = 23
        npkdata   = outnum*Non
        peakdata  = np.fromfile(pkfile_in, dtype=np.float32, count=npkdata)
        peakdata  = np.reshape(peakdata,(Non,outnum))

	pkfile_in.close()

        #Sort by Rth
        peakdata = peakdata[peakdata[:,6].argsort()[::-1]]
        peakdata = peakdata[:ncut,:]

	if Lag==1: #Use approx z=0 conversion of V to S
		peakdata[:,0] -= peakdata[:,3]/32.37668
		peakdata[:,1] -= peakdata[:,4]/32.37668
		peakdata[:,2] -= peakdata[:,5]/32.37668

	Non = ncut
        peakdata = np.reshape(peakdata,(Non*outnum,1))

        pkfile_out = open(pp_cat+"_"+str(ncut)+"_"+str(Lag),"wb")
        pkfile_out.write(struct.pack('1i',Non))
        pkfile_out.write(struct.pack('1f',RTH))
        pkfile_out.write(struct.pack('1f',zin))
        pkfile_out.write(struct.pack('%sf' % len(peakdata) , *peakdata))
        pkfile_out.close()


#Get all merged files
if fmt==0:
	folder = folder+'output/'
	f = os.listdir(folder)
	f = [s for s in f if "merge" in s]
	f = [s for s in f if "shuffle" not in s]
	f = [s for s in f if "0000" not in s]
	f = sorted(f)
	seeds = [x[-5:] for x in f]

if fmt==1:
	folder = folder+'fields/'
	f = os.listdir(folder)
	f = sorted(f)
	seeds = [x[:5] for x in f]

print f
print seeds
for i in range(min(len(f),nxi)):
	seed = seeds[i] 
	pp_cat = folder+f[i]

	# Name output file
	if Lag==1: 
		pkfile = str(Lbox)+'_n'+str(ncut)+'_pkL.'+str(seed)
        if (shuff==0) and (Lag==0): 
		pkfile = str(Lbox)+'_n'+str(ncut)+'_pk.'+str(seed)
        if (shuff==1) and (Lag==0):	
		pkfile = str(Lbox)+'_n'+str(ncut)+'_pk_shuffle.'+str(seed)
        if (fmt==1):	
		pkfile = str(Lbox)+'_pkF.'+str(seed)

       #If number cut pp file does not exist then make it
	if fmt==0:
		if os.path.isfile(pp_cat+"_"+str(ncut)+"_"+str(Lag))==False:
			cut_cat(pp_cat,ncut,Lag)

	#use number cut pp file
	if fmt==0:
		pp_cat = folder+f[i]+"_"+str(ncut)+"_"+str(Lag)

	if os.path.isfile(dirout+'power_'+pkfile)==False:
		print "Running on: ", pp_cat
       #powerspectra <mergedfile> <outfile> <Lbox in [Mpc]> 
       #             <nmesh> <fmt=: 0=peaks, 1=field>
		subprocess.call('mpirun -np '+str(nproc)+' powerspectra '+pp_cat+' '+pkfile+' '+str(Lbox)+' '+str(nmesh)+' '+str(fmt)+' 2>> '+dirout+str(ncut)+'_'+str(Lag)+'.stderr' + ' 1>> '+dirout+str(ncut)+'_'+str(Lag)+'.stdout', shell=True)
		subprocess.call('mv *'+pkfile+' '+dirout,shell=True)


