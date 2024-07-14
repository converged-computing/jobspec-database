# -*- coing: utf-8 -*-
"""
Created on Tue April 11 11:42:00 2018
@author: solivieri and ianto-cannon
"""
import numpy as np
from math import ceil, floor
from pathlib import Path #IC
import os
import shutil
def combineOutputs():
  print('combineOutputs.py start')
  Path('output/').mkdir(parents=True, exist_ok=True) #IC make directory
  Path('mpiOutput/').mkdir(parents=True, exist_ok=True) #IC make directory
  snapshots = {}
  for inFile in os.listdir('output/'):
    if 'rank' not in inFile or 'DropsT00' not in inFile: continue
    if os.stat('output/'+inFile).st_size > 0:
      rank = int(inFile.split('rank')[1][:-4])
      if rank < 4: print('loading ' + 'output/' + inFile)
      with open('output/'+inFile[:-13]+'.dat','ab') as wfd:
        with open('output/'+inFile,'rb') as fd:
          shutil.copyfileobj(fd, wfd)
      if 'stat' in inFile:  
        time = int(inFile[10:-13])
        with open('output/'+inFile) as f:
          nDrops=0
          area=0.0
          vInv=0.0
          for line in f:
            nDrops += 1
            vInv += (1./float(line.split()[2])-vInv)/nDrops
            area += float(line.split()[3])
        if time in snapshots:
          nTot = snapshots[time][0]
          vInvAv = snapshots[time][1]
          vInvSqAv = snapshots[time][2]
          areaTot = snapshots[time][3]
        else:
          nTot = 0
          vInvAv = 0.0
          vInvSqAv = 0.0
          areaTot = 0
        vInvAv = (vInvAv*nTot+vInv*nDrops)/(nDrops+nTot)
        vInvSqAv = (vInvSqAv*nTot + vInv**2*nDrops)/(nDrops+nTot)
        nTot = nTot + nDrops
        areaTot = areaTot + area
        snapshots[time] = [nTot,vInvAv,vInvSqAv,areaTot]
    os.rename('output/'+inFile,'mpiOutput/'+inFile)
  with open('output/'+'avgVol.dat', 'w') as the_file:
    for time, vInv in sorted(snapshots.items()):
      vInvStdDev = np.sqrt(vInv[2]-vInv[1]**2)
      #use first order Taylor series expansion in the mean volume
      vStdDev = vInvStdDev/vInv[1]**2
      #time,  nTot, 1/vInvAv, stdDev(1/vInvAv), areaTot
      the_file.write(str(time)+'  '+str(vInv[0])+'  '+str(1.0/vInv[1])+'  '+str(vStdDev)+'  '+str(vInv[3])+'\n')
  print('combineOutputs.py end')
  print(' ')
  return

#prevent this code from running when imported:
if __name__ == "__main__":
  combineOutputs()
