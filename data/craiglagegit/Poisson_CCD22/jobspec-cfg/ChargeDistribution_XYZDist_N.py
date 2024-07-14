#!/usr/bin/env python

#Author: Craig Lage, NYU;
#Date: 26-Jan-15

#This program plots the Poisson equation solutions from the C++ Poisson solver
import matplotlib
matplotlib.use("PDF")
from pylab import *
import os, sys, time, h5py

#****************SUBROUTINES*****************
class Array3dHDF5Elec(object):
    def __init__(self, dir, filebase, n):
        elecfile = dir+'/'+filebase+'_'+str(n)+'_Elec' + '.hdf5'
        hdfelec = h5py.File(elecfile,'r')
        holefile = dir+'/'+filebase+'_'+str(n)+'_Hole' + '.hdf5'
        hdfhole = h5py.File(holefile,'r')
        self.elec=array(hdfelec[hdfelec.items()[0][0]])
        self.hole=array(hdfhole[hdfhole.items()[0][0]])

        phifile = dir+'/'+filebase+'_'+str(run)+'_phi' + '.hdf5'
        rhofile = dir+'/'+filebase+'_'+str(run)+'_rho' + '.hdf5'
        xfile = dir+'/'+'grid_x.dat'
        yfile = dir+'/'+'grid_y.dat'
        zfile = dir+'/'+'grid_z.dat'        

        xgrid = loadtxt(xfile, skiprows=1)
        ygrid = loadtxt(yfile, skiprows=1)
        zgrid = loadtxt(zfile, skiprows=1)        

        self.nx=xgrid.shape[0]
        self.ny=ygrid.shape[0]

        Dimension = hdfelec[hdfelec.items()[0][0]].attrs[u'Dimension']
        self.nz=Dimension[2] # Note the Elec anf Hole grids are smaller than the phi,etc. grids.

        self.xmin=xgrid[0,1]
        self.ymin=ygrid[0,1]
        self.zmin=zgrid[0,1]

        self.xmax=xgrid[self.nx-1,3]
        self.ymax=ygrid[self.ny-1,3]
        self.zmax=zgrid[self.nz-1,3]

        self.x=xgrid[:,2]
        self.y=ygrid[:,2]
        self.z=zgrid[0:self.nz,2]

        EPSILON_SI = 11.7
        EPSILON_OX = 4.3

        self.Channelkmin = self.nz-1
        for k in range(self.nz):
            if ConfigData["GateOxide"] > zgrid[k,1] and ConfigData["GateOxide"] < zgrid[k,3]:
                self.Channelkmin = k + 1
                print "Channelkmin = %d"%self.Channelkmin
                break

def ReadConfigFile(filename):
    # This reads the Poisson simulator config file for
    # the settings that were run
    # and returns a dictionary with the values
    ConfigData = {}
    try:
        file = open(filename,'r')
        lines=file.readlines()
        file.close()
    except IOError:
        print "Configuration file %s not found"%filename
        return False, ConfigData 

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
                        try:
                            ConfigData[ParamName] = float(ThisParam)
                        except ValueError:
                            try:
                                ConfigData[ParamName] = ThisParam
                            except ValueError:
                                return False, ConfigData 
                else:
                    ThisParam = []
                    for item in ThisLine:
                        try: ThisParam.append(int(item))
                        except ValueError:
                            try: ThisParam.append(float(item))
                            except ValueError:
                                ThisParam.append(item)
                    ConfigData[ParamName] = ThisParam
            except (IOError, ValueError):
                continue
    except Exception as e:
        print "Error reading configuration file %s. Exception of type %s and args = \n"%(filename,type(e).__name__), e.args 
        return False, ConfigData 

    return True, ConfigData

#****************MAIN PROGRAM*****************

# First, read the .cfg file

configfile = sys.argv[1]
run = int(sys.argv[2])
cfg_success, ConfigData = ReadConfigFile(configfile)
if not cfg_success:
    print "Configuration file issue. Quitting"
    sys.exit()
outputfilebase = ConfigData["outputfilebase"]
outputfiledir = ConfigData["outputfiledir"]

# This holds all of the data
dat = Array3dHDF5Elec(outputfiledir, outputfilebase, run)

ScaleFactor = ConfigData["ScaleFactor"]
GridsPerPixel = ConfigData["GridsPerPixel"]

nxx = dat.nx - 1
nyy = dat.ny - 1
nzz = dat.nz - 1
NumPixelsPlotted = int(sys.argv[3])
nxcenter = nxx/2
nycenter = nyy/2
nxmin = nxcenter - (NumPixelsPlotted * ScaleFactor * GridsPerPixel)/2
nxmax = nxmin + (NumPixelsPlotted * ScaleFactor * GridsPerPixel)
nymin = nycenter - (NumPixelsPlotted * ScaleFactor * GridsPerPixel)/2
nymax = nymin + (NumPixelsPlotted * ScaleFactor * GridsPerPixel)
levels = linspace(-3.0, 7.0, 101)
file = open(outputfiledir+"/charge.txt","w")

nzxmin = nxcenter - (ScaleFactor * GridsPerPixel)/2
nzxmax = nzxmin + (ScaleFactor * GridsPerPixel)
nzymin = nycenter - (ScaleFactor * GridsPerPixel)/2
nzymax = nzymin + (ScaleFactor * GridsPerPixel)
ncenter = 0.0
nzcenter = 0.0

Total_elec = 0.0
Below_kmin = 0.0
for nx in range(nzxmin, nzxmax):
    for ny in range(nzymin, nzymax):
        for nz in range(nzz):
            if dat.elec[nx,ny,nz] > 0.0:
                ncenter += dat.elec[nx,ny,nz]
                nzcenter += dat.z[nz] * dat.elec[nx,ny,nz]
meanz = nzcenter / ncenter
print "Electrons in center Pixel = %.1f, Mean z = %.3f microns\n"%(ncenter, meanz)
print "Total Electrons = %.1f\n"%dat.elec.sum()

file.write("Electrons in center Pixel = %.1f, Mean z = %.3f microns\n"%(ncenter, meanz))
file.close()

carriers = ['Electron', 'Hole']
plotdatas = [dat.elec, dat.hole]

# Create the output directory if it doesn't exist
if not os.path.isdir(outputfiledir+"/plots"):
    os.mkdir(outputfiledir+"/plots")

kmax = dat.nz / 2

for i, plotdata in enumerate(plotdatas):

    if i == 1:
        nxcenter += ScaleFactor * GridsPerPixel / 2
        nycenter += ScaleFactor * GridsPerPixel / 2

    fig = figure(figsize = (12,12))
    suptitle("%s Charge Distribution"%carriers[i], fontsize = 36)

    ax1=axes([0.10,0.40,0.50,0.50],aspect=1)
    ax1.set_title("X-Y Slice")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plotarray = sum(plotdata[nxmin:nxmax,nymin:nymax,:],axis = 2)+0.1
    if i == 0:
        for nx in range(ScaleFactor*GridsPerPixel,nxmax-nxmin,ScaleFactor*GridsPerPixel):
            for ny in range(0,nymax-nymin):
                plotarray[nx,ny] = 0.001
        for ny in range(ScaleFactor*GridsPerPixel,nymax-nymin,ScaleFactor*GridsPerPixel):
            for nx in range(0,nxmax-nxmin):
                plotarray[nx,ny] = 0.001
        for nx in range(ScaleFactor*GridsPerPixel-1,nxmax-nxmin,ScaleFactor*GridsPerPixel):
            for ny in range(0,nymax-nymin):
                plotarray[nx,ny] = 0.001
        for ny in range(ScaleFactor*GridsPerPixel-1,nymax-nymin,ScaleFactor*GridsPerPixel):
            for nx in range(0,nxmax-nxmin):
                plotarray[nx,ny] = 0.001

    ax1.imshow(log10(transpose(plotarray)), interpolation = 'nearest')

    ax2=axes([0.10,0.20,0.50,0.20])
    ax2.set_title("X-Z Slice")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plotarray = sum(plotdata[nxmin:nxmax,:,dat.Channelkmin:kmax],axis = 1)+0.1

    ax2.imshow(log10(transpose(fliplr(plotarray))), interpolation = 'nearest')

    ax3=axes([0.10,0.10,0.50,0.10])
    ax3.set_title("X-Cut, Y = %.2f"%dat.y[nycenter])
    plotarray = sum(plotdata[nxmin:nxmax,nycenter,:],axis = 1)
    ax3.plot(dat.x[nxmin:nxmax],plotarray)
    ax3.set_xlabel("X ( Microns)")
    ax3.set_ylabel("Charge Density")


    ax4=axes([0.60,0.40,0.20,0.50])
    ax4.set_title("Y-Z Slice")
    ax4.set_xticks([])
    ax4.set_yticks([])
    plotarray = sum(plotdata[:,nymin:nymax,dat.Channelkmin:kmax],axis = 0)+0.1

    ax4.imshow(log10(fliplr(plotarray)), interpolation = 'nearest')

    ax5=axes([0.80,0.40,0.10,0.50])
    ax5.set_title("Y-Cut, X = %.2f"%dat.x[nxcenter])
    plotarray = sum(plotdata[nxcenter,nymin:nymax,0:20],axis = 1)
    ax5.plot(plotarray,dat.y[nymin:nymax])
    ax5.set_xlim(ax5.get_xlim()[::-1])
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_position("right")
    for tick in ax5.get_xticklabels():
        tick.set_rotation(90)
    ax5.set_ylabel("Y ( Microns)")
    ax5.set_xlabel("Charge Density")

    ax6=axes([0.65,0.25,0.10,0.10])
    ax6.set_title("Z-Cut")
    if i == 0:
        plotarray = log10(plotdata[nxcenter,nycenter,:]+0.01)
        ax6.plot(dat.z[:],plotarray, label = '(X,Y) = (%.2f,%.2f)'%(dat.x[nxcenter],dat.y[nycenter]))
    if i == 1:
        plotarray = log10(plotdata[nxcenter,nycenter,:]+0.01)
        ax6.plot(dat.z[:],plotarray, label = '(X,Y) = (%.2f,%.2f)'%(dat.x[nxcenter],dat.y[nycenter]))
        nycenter -= ScaleFactor * GridsPerPixel / 2
        plotarray = log10(plotdata[nxcenter,nycenter,:]+0.01)
        ax6.plot(dat.z[:],plotarray, label = '(X,Y) = (%.2f,%.2f)'%(dat.x[nxcenter],dat.y[nycenter]))
        legend(bbox_to_anchor=(1.20, 0.5), loc=2, fontsize = 9)
    ax6.set_ylabel("Log Charge Density")
    ax6.set_xlim(ax6.get_xlim()[::-1])
    ax6.set_xticks([0.0,1.0,2.0])
    ax6.set_xlabel("Z ( Microns)")
    ax6.yaxis.set_label_position("right")
    savefig(outputfiledir+"/plots/%sDistribution_XYZ_%d.pdf"%(carriers[i],run))
    close(fig)
