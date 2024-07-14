#!/usr/bin/env python

#Author: Craig Lage, NYU; 
#Date: 14-Sep-15

#This program manages the running of multiple Poisson spots
import matplotlib
matplotlib.use("Agg")
from pylab import *
import sys, time, subprocess,h5py
from scipy.special import erf
from scipy.optimize import fmin_powell
from scipy import stats
sys.path.append('/g/g17/lage1/Software/forward_model_varying_i')
import forward
import mpi

#****************SUBROUTINES*****************

class Array2dSet:
    def __init__(self,xmin,xmax,nx,ymin,ymax,ny,nstamps):
        # This packages up a set of nstamps postage stamp images,
        # each image of which is nx * ny pixels
        self.nx=nx
        self.ny=ny
        self.nstamps=nstamps

        self.xmin=xmin
        self.ymin=ymin
        
        self.xmax=xmax
        self.ymax=ymax
        
        self.dx=(xmax-xmin)/nx
        self.dy=(ymax-ymin)/ny
        
        self.x=linspace(xmin+self.dx/2,xmax-self.dx/2,nx)
        self.y=linspace(ymin+self.dy/2,ymax-self.dy/2,ny)

        self.data=zeros([nx,ny,nstamps])
        self.xoffset=zeros([nstamps])
        self.yoffset=zeros([nstamps])
        self.imax=zeros([nstamps])

class Array3dHDF5Elec(object):
    def __init__(self, dir, filebase, n):
        elecfile = dir+'/'+filebase+'_'+str(n)+'_Elec.hdf5'
        hdfelec = h5py.File(elecfile,'r')
        Dimension = hdfelec[hdfelec.items()[0][0]].attrs[u'Dimension']
        self.nx=Dimension[0]
        self.ny=Dimension[1]
        self.nz=Dimension[2]
        
        Lower_Left = hdfelec[hdfelec.items()[0][0]].attrs[u'Lower_Left']
        self.xmin=Lower_Left[0]
        self.ymin=Lower_Left[1]
        self.zmin=Lower_Left[2]

        Upper_Right = hdfelec[hdfelec.items()[0][0]].attrs[u'Upper_Right']
        self.xmax=Upper_Right[0]
        self.ymax=Upper_Right[1]
        self.zmax=Upper_Right[2]
        
        self.dx=(self.xmax-self.xmin)/self.nx
        self.dy=(self.ymax-self.ymin)/self.ny
        self.dz=(self.zmax-self.zmin)/self.nz
        self.volume = self.dx * self.dy * self.dz
        
        self.x=linspace(self.xmin+self.dx/2,self.xmax-self.dx/2,self.nx)
        self.y=linspace(self.ymin+self.dy/2,self.ymax-self.dy/2,self.ny)
        self.z=linspace(self.zmin+self.dz/2,self.zmax-self.dz/2,self.nz)

        self.elec=array(hdfelec[hdfelec.items()[0][0]])

def Area(xl, xh, yl, yh, sigmax, sigmay, Imax):
    # Calculates how much of a 2D Gaussian falls within a rectangular box
    ssigx = sqrt(2) * sigmax
    ssigy = sqrt(2) * sigmay    
    I = (erf(xh/ssigx)-erf(xl/ssigx))*(erf(yh/ssigy)-erf(yl/ssigy))
    return Imax * I / 4.0

def FOM(params):
    global spotlist
    [sigmax, sigmay] = params
    result = forward.forward(spotlist,sigmax,sigmay)
    return result

def ReadConfigFile(filename):
    # This reads the config file for the necessary settings
    # and returns a dictionary with the values
    file = open(filename,'r')
    lines=file.readlines()
    file.close()
    ConfigData = {}
    try:
        for line in lines:
            ThisLine=line.strip().split()
            ThisLineLength=len(ThisLine)
            if ThisLineLength < 3:
                continue
            if list(ThisLine[0])[0]=='#' or ThisLine[0]=='\n':
                continue
            try:
                ParamName = ThisLine[0]
                ThisLine.remove(ThisLine[0])
                for counter,item in enumerate(ThisLine):
                    if list(item)[0] == '#':
                        del ThisLine[counter:] # Strip the rest of the line as a comment
                        continue
                    if item == '=':
                        ThisLine.remove(item)
                        continue
                if len(ThisLine) == 0:
                    continue
                elif len(ThisLine) == 1:
                    ThisParam = ThisLine[0]
                    try: ConfigData[ParamName] = int(ThisParam)
                    except ValueError:
                        try: ConfigData[ParamName] = float(ThisParam)
                        except ValueError:
                            try:
                                ConfigData[ParamName] = ThisParam
                            except ValueError:
                                print "Error reading .cfg file"
                else:
                    ThisParam = []
                    for item in ThisLine:
                        try: ThisParam.append(int(item))
                        except ValueError:
                            try: ThisParam.append(float(item))
                            except ValueError:
                                ThisParam.append(item)
                    ConfigData[ParamName] = ThisParam
            except:
                continue
    except:
        print "Error reading .cfg file"

    return ConfigData

def New_Cfg_File(incfgfile, outcfgfile, newrun):
    # This increments the run number to start a new spot
    # and assigns a random offset value within the central pixel
    ConfigData = ReadConfigFile(incfgfile)
    lines = OpenFile(incfgfile)

    dirbase = ConfigData['outputfiledir'].split('run')
    
    xoff = -5.0 + 10.0 * rand()
    yoff = -5.0 + 10.0 * rand()
    lines[114] = 'Xoffset = '+str(xoff)+'\n'
    lines[115] = 'Yoffset = '+str(yoff)+'\n'
    lines[140] = 'outputfiledir 	= '+dirbase[0]+'run'+newrun+'\n'
    newcfgfile = open(outcfgfile, 'w')
    for line in lines:
        newcfgfile.write(line)
    newcfgfile.close()
    
    return 


def OpenFile(filename):
    Numtries = 10
    tries = 0
    lines = []
    while tries < Numtries:
        try:
            file = open(filename, 'r')
            lines = file.readlines()
            file.close()
            break
        except:
            time.sleep(1.0)
            tries += 1

    return lines

def FillSpotlist(run, Numspots):
    global spotlist
    # Note sigmas and offset are in pixels, not microns.
    incfgfile = datadir+startdir+'bf.cfg'
    InConfigData = ReadConfigFile(incfgfile)
    # Postage Stamp size
    nx = InConfigData['PixelBoundaryNx']
    ny = InConfigData['PixelBoundaryNy']

    outputfiledir = InConfigData['outputfiledir']
    outputfilebase = InConfigData['outputfilebase']
    GridsPerPixel = InConfigData['GridsPerPixel'] * InConfigData['ScaleFactor']
    PixelSize = InConfigData['PixelSize']
    ChannelStopWidth = InConfigData['ChannelStopWidth']
    cspixels = int(ChannelStopWidth / PixelSize * float(GridsPerPixel / 2)) + 1
    stampxmin = -(int(nx/2)+0.5)
    stampxmax = -stampxmin
    stampymin = -(int(ny/2)+0.5)
    stampymax = -stampymin

    spotlist = Array2dSet(stampxmin,stampxmax,nx,stampymin,stampymax,ny,Numspots-1)

    dirbase = outputfiledir.split('bfrun')
    
    for spot in range(Numspots-1): 
        spotrun = spot + 1 # Don't include run 0 because it's different
        dat = Array3dHDF5Elec(dirbase[0]+'bfrun_%d'%spotrun, outputfilebase, run)
        cfgfile = dirbase[0]+'bfrun_%d'%spotrun+'/bf.cfg'
        ConfigData = ReadConfigFile(cfgfile)

        spotlist.xoffset[spot] = ConfigData['Xoffset'] / ConfigData['PixelSize']
        spotlist.yoffset[spot] = ConfigData['Yoffset'] / ConfigData['PixelSize']

        for i in range(nx):
            nxmin = ((ConfigData['PixelBoundaryLowerLeft'][0] - dat.xmin) / dat.dx) + GridsPerPixel * i
            nxmax = nxmin + GridsPerPixel
            for j in range(ny):
                nymin = ((ConfigData['PixelBoundaryLowerLeft'][1] - dat.ymin) / dat.dy) + GridsPerPixel * j
                nymax = nymin + GridsPerPixel
                electrons_in_pixel = dat.elec[(nxmin+cspixels):(nxmax-cspixels),nymin:nymax,:].sum()
                #print "i = %d, j = %d, nxmin = %d, nymin = %d, electron = %d"%(i,j,nxmin,nymin,electrons_in_pixel)
                spotlist.data[i,j,spot] = electrons_in_pixel

    param0 = [1.00, 1.00]
    args = ()
    Result = fmin_powell(FOM, param0, args)
    
    imax = spotlist.imax.mean()
    ADU_correction = Area(-0.5,0.5,-0.5,0.5,Result[0],Result[1],1.0)

    spotdata = [run, Result[0], Result[1], imax * ADU_correction]
    print spotdata
    mpi.send(spotdata, Numspots - 1, tag = run)
    return

def PlotSpotlist(Numruns, Numspots, imaxs, sigmaxs, sigmayx):
    global spotlist
    incfgfile = datadir+startdir+'bf.cfg'
    InConfigData = ReadConfigFile(incfgfile)
    # Postage Stamp size
    nx = InConfigData['PixelBoundaryNx']
    ny = InConfigData['PixelBoundaryNy']
    file = open("bf.txt","w")
    figure()
    title("Baseline - Sigmax = Sigmay = 1.0, Offsets=random, With Diffusion")
    scatter(imaxs, sigmaxs, color = 'green', lw = 2, label = 'Sigma-x')
    scatter(imaxs, sigmays, color = 'red', lw = 2, label = 'Sigma-y')
 
    slope, intercept, r_value, p_value, std_err = stats.linregress(imaxs[4:9],sigmaxs[4:9])
    xplot=linspace(0.0,150000.0,100)
    yplot = slope * xplot + intercept
    plot(xplot, yplot, color='blue', lw = 2, ls = '--')
    tslope = slope * 100.0 * 50000.0
    text(10000.0,0.98,"X Slope = %.2f %% per 50K e-, Intercept = %.3f"%(tslope,intercept))
    file.write("X Slope = %.2f %% per 50K e-, Intercept = %.3f\n"%(tslope,intercept))
    slope, intercept, r_value, p_value, std_err = stats.linregress(imaxs[4:9],sigmays[4:9])
    xplot=linspace(0.0,150000.0,100)
    yplot = slope * xplot + intercept
    plot(xplot, yplot, color='black', lw = 2, ls = '--')
    tslope = slope * 100.0 * 50000.0
    text(10000.0,0.97,"Y Slope = %.2f %% per 50K e-, Intercept = %.3f"%(tslope,intercept))
    file.write("Y Slope = %.2f %% per 50K e-, Intercept = %.3f\n"%(tslope,intercept))
    text(10000.0,0.99,"%d Simulated spots"%Numspots)
    xlabel('Central Peak(electrons)')
    ylabel('Sigma (Pixels)')
    legend(loc= 'lower right')
    ylim(0.95, 1.10)
    xlim(0.0,150000.0)
    xticks([0.0,50000,100000])

    savefig(datadir+startdir+"plots/BF_Sim_%d_%d.png"%(Numruns,Numspots))
    file.close()
    return

#****************MAIN PROGRAM*****************
global spotlist
datadir = 'data/'
startdir = 'bfrun1/'

rank = mpi.rank
Numspots = mpi.size

mpi.barrier() # Wait until everybody gets here

runname = '_%d'%rank
nextdir = datadir+'bfrun'+runname+'/'
newdir = subprocess.Popen('mkdir -p '+nextdir, shell=True) 
subprocess.Popen.wait(newdir)
incfgfile = datadir+startdir+'bf.cfg'
outcfgfile = nextdir+'bf.cfg'
ConfigData = ReadConfigFile(incfgfile)
Numruns = ConfigData['NumSteps']
SaveElec = ConfigData['SaveElec']

New_Cfg_File(incfgfile, outcfgfile, runname)

if rank == 0:  # Use run0 for the area shift with all of the charge in the central pixel.
    newdir = subprocess.Popen('cp data/bfrun1/bf0.cfg '+nextdir+'bf.cfg', shell=True) 
    subprocess.Popen.wait(newdir)
    newdir = subprocess.Popen('mkdir data/bfrun_0/plots', shell=True) 
    subprocess.Popen.wait(newdir)


cmd = '~/Software/Poisson_CCD_Hole10/src/Poisson '+outcfgfile
cpp_job = subprocess.Popen(cmd, shell=True)
subprocess.Popen.wait(cpp_job)
time.sleep(1.0)

mpi.barrier() # Wait until everybody gets here


NumPts = Numruns/SaveElec

if rank == Numspots - 1: # Use the last processor to gather the data and plot it
    imaxs = zeros([NumPts])
    sigmaxs = zeros([NumPts])
    sigmays = zeros([NumPts])

    for point in range(NumPts):
        run = point * SaveElec
        src = run % (Numspots - 1)
        spotdata, status = mpi.recv(src, tag = run)
        if run != spotdata[0]:
            print "Communication error! Run = %d, tag = %d"%(run, spotdata[0])
            continue
        else:
            print "Received data for run = %d"%run
            sigmaxs[point] = spotdata[1]
            sigmays[point] = spotdata[2]
            imaxs[point] = spotdata[3]

    PlotSpotlist(Numruns, Numspots, imaxs, sigmaxs, sigmays)

else:                    # Everybody else analyzes one or more runs with all of the spots.
    for point in range(NumPts):
        run = point * SaveElec
        if run % (Numspots - 1) == rank:
            FillSpotlist(run, Numspots)

if rank == 0:  # Use run0 to plot the area plots and charge distribution
    areaplot = subprocess.Popen('python AreaPlot_Corr.py data/bfrun_0/bf.cfg 80', shell=True) 
    subprocess.Popen.wait(areaplot)
    chargeplot = subprocess.Popen('python ChargeDistribution_XYZDist_N.py data/bfrun_0/bf.cfg 100 3', shell=True) 
    subprocess.Popen.wait(chargeplot)

