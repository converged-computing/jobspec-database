import numpy as np
import sys
import os 
import ntpath
import glob

dirin  = sys.argv[1]
ncut   = sys.argv[2]
print "\ndirin    = ",dirin

def get_data(filename):

    data = np.loadtxt(filename)
    xx = data[:,0]
    yy = data[:,1]

    return xx,yy

#GET PK FILES INTO ARRAY
nfiles = len(glob.glob(dirin+"/*power*"))
count  = 0
for f in glob.glob(dirin+"/*power*"):
        print f
        if count==0:
            k,pk     = get_data(f)
            kpk      = np.zeros((len(k),nfiles))
            kpk[:,count] = pk
        else:
            k,pk = get_data(f)
            kpk[:,count] = pk

        count+=1

print "\nNumber of pk files = ", count

#GET XI FILES INTO ARRAY
nfiles = len(glob.glob(dirin+"/*correlation*"))
count  = 0
for f in glob.glob(dirin+"/*correlation*"):
        print f
        if count==0:
            s,xi     = get_data(f)
            sxi      = np.zeros((len(s),nfiles))
            sxi[:,count] = xi
        else:
            s,xi = get_data(f)
            sxi[:,count] = xi

        count+=1
print "\nNumber of xi files = ", count

outdir = "numpy_data/"
np.savez(outdir+dirin+"_powerspec_"+str(count),k=k,pk=kpk)
np.savez(outdir+dirin+"_correltation_"+str(count),s=s,xi=sxi)
