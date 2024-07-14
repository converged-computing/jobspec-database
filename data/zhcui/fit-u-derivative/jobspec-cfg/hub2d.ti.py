import sys
import dmetTI as dmet
import numpy as np
import utils 
import numpy.linalg as la
import loggers.simpleLogger as log
import random


U = float(sys.argv[1]) 
#lam = float(sys.argv[2])
bdir = sys.argv[2] 
factor = float(sys.argv[3])  
cmtype = 'BCS'

#------------------------------------------------------------------------------------

v = str(U).replace('.','_')
#sys.stdout = log.Logger(bdir + '/log' + v + '.txt')
#sys.stderr = log.Logger(bdir + '/elog' + v + '.txt')

#------------------------------------------------------------------------------------
print
print "#########################################################"
print "Starting 2D Hubbard Model"
print "#########################################################"
print 

#2d Hubbard Model
Nx = 8
Ny = 8 
N = [Nx,Ny] 
Ix = 2
Iy = 2
I = [Ix,Iy]

#--------------------------------------------------------
Lx = Nx*Ix
Ly = Ny*Iy

norb = Lx*Ly
nimp = Ix*Iy

mtype = np.float64
#mtype = np.complex128
h1 = np.zeros([norb,norb],dtype=mtype)
#h1soc = np.zeros([norb,norb],dtype=mtype)
#g2e = np.zeros([norb,norb,norb,norb])

for j in range(Ny):
    for i in range(Nx):
        
        for y in range(Iy):
            for x in range(Ix):

                site = (Nx*j + i)*nimp + Ix*y + x
        
                if(y+1>=Iy):
                    nny = (j+1)%Ny
                    cy = 0
                else:
                    nny = j
                    cy = y+1

                if(x+1>=Ix):
                    nnx = (i+1)%Nx
                    cx = 0
                else:
                    nnx = i
                    cx = x+1

                if(x-1<0):
                    bnx = (i-1)%Nx
                    bx = Ix-1
                else:
                    bnx = i
                    bx = x-1

                if(y-1<0):
                    bny = (j-1)%Ny
                    by = Iy-1
                else:
                    bny = j
                    by = y-1

                siteu = (Nx*nny + i)*nimp + Ix*cy + x
                siter = (Nx*j + nnx)*nimp + Ix*y + cx
                sited = (Nx*bny + i)*nimp + Ix*by + x
                sitel = (Nx*j + bnx)*nimp + Ix*y + bx               

#               print i,j,x,y,site,siteu,siter,sited,sitel              
                    
                if(site!=siteu):     
                    h1[site,siteu] = -1.0
            
                if(site!=siter):
                    h1[site,siter] = -1.0
                
                if(site!=sited):
                    h1[site,sited] = -1.0
                
                if(site!=sitel):
                    h1[site,sitel] = -1.0
                
                '''
                h1soc[site,siteu] = lam*1j
                h1soc[site,siter] = lam*1j
                h1soc[site,sited] = lam*1j
                h1soc[site,sitel] = lam*1j
                '''
#               g2e[site,site,site,site] = 4.0


#Always check this for h1
assert(np.sum(np.matrix(h1) - np.matrix(h1).H)<=1.0e-16)
#--------------------------------------------------------
#gse,c = fci.kernel(h1,g2e,norb,norb)
#print "FCI energy per site: ",gse/norb
#--------------------------------------------------------

um = np.zeros([2*Ix*Iy,2*Ix*Iy])
if(True):
    hn = nimp/2
    #um = np.diag([0.1,-0.1]*hn + [-0.1,0.1]*hn)
    #um = np.diag([0.001,-0.001,-0.001,0.001] + [0.001,-0.001,-0.001,0.001])
    #um = np.diag([0.1,-0.1,-0.1,0.1] + [0.1,-0.1,-0.1,0.1])
    #um = np.diag([0.1,-0.1,-0.1,0.1] + [-0.1,0.1,0.1,-0.1])
    #um = np.diag([0.06525,0.02475,0.02475,0.06525] + [-0.02475,-0.06525,-0.06525,-0.02475])*U*10.0 
    if(U == 0.0):
        uf = 1.0e-02
    else:
        uf = U

    basevs = 0.75
    for y in range(Iy):
        basev = basevs
        for x in range(Ix):
            site = Ix*y+x
            um[site,site] = basev*uf*factor
            if(cmtype == 'BCS'):
                um[site+nimp,site+nimp] = (basev-1.0)*uf*factor
            elif(cmtype == 'UHF' or cmtype == 'SOC'):
                um[site+nimp,site+nimp] = (1.0-basev)*uf*factor
            else:
                um[site+nimp,site+nimp] = basev*uf*factor
            basev = 1.0 - basev
        basevs = 1.0 - basevs
    '''
    odg = np.zeros([Ix*Iy,Ix*Iy])
    for i in range(Ix*Iy):
        for j in range(i+1,Ix*Iy):
            odg[i,j] = random.random()*0.0e-1
    um[:Ix*Iy,:Ix*Iy] += odg + odg.conjugate().T
    um[Ix*Iy:,Ix*Iy:] += odg + odg.conjugate().T
    '''

if(True):
    if(cmtype == 'BCS'):
        #random.seed(10)
        vcorr = np.zeros([Ix*Iy,Ix*Iy]) 
        for i in range(Ix*Iy):
            for j in range(Ix*Iy):
                vcorr[i,j] = random.random()*1.0e-3

        for i in range(Ix*Iy):
            vcorr[i,i] = random.random()*0.0e-3

        um[:Ix*Iy,Ix*Iy:] = vcorr
        um[Ix*Iy:,:Ix*Iy] = vcorr.conjugate().T

nelec = int(norb*factor) 
g2e = np.zeros([1])
g2e[0] = U 

#print um
#print h1
#exit()

#--------------------------------------------------------
# WRITE GUESS TO FILE -- USEFUL FOR TESTS
if(0):
    np.save(bdir+'/um0',um,allow_pickle=False)
else:
    um = np.load(bdir+'/um0.npy',allow_pickle=False)
#--------------------------------------------------------

print "Interaction Strength (U/t):",U
print "Lattice dimensions:",N
print "Unit cell dimensions:",I
print "No. of electrons:",nelec
print "No. of sites:",norb
print

obj = dmet.dmet(N,nelec,I,h1,g2e, mtype = mtype, u_matrix = um,ctype=cmtype, SolverType='dmrg', loc_bath = 'pmfull')

obj.bdir = bdir

#Ceres Setup
ceres = False
if(ceres):
    obj.inpdata = 'ceres.inp.bin'
    obj.matrixdata = 'ceres.um.bin'

#output files
obj.rdmLog = 'rdmlog' + v + '.txt'
obj.dlog = log.Logger(bdir + '/' + obj.rdmLog) 
obj.tableFile = bdir + '/table' + v + '.txt'

#checkpoint options and files
obj.chkPointInterval=1
obj.chkPointFile= bdir + '/hub2d_chkpt_' + v + '.hdf5'
obj.resFile = bdir + '/hub2d_chkpt_' + v + '.hdf5'
obj.doRestart = True 
obj.resetStart = False 

#Iteration specifics
obj.fitBath = True
obj.iformalism = False 

obj.muSolverSimple = True 
obj.startMuDefault = True 
obj.startMu = 0.0 #-1.992233261e-01  

obj.doDAMP = False
obj.dampStart = 0
obj.dampTol = 0.1

obj.diisStart = 4 
obj.diisDim = 4

obj.traceFixStart = 0
obj.denFitStop = 0
obj.impFitStop = 0

obj.dmetitrmax = 200 
obj.corrIter = 20
#obj.corrTol = 1.0e-4
obj.corrTol = 1.0e-5
obj.corrTol_init = min(obj.corrTol * 10.0, 1.0e-3) # ZHC


obj.maxM = 400
obj.mu_per_sweep = True
obj.dm_DIIS = False

#--------------------------------------------------------
obj.solve_groundstate()
