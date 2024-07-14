#!/usr/bin/env python
"""
Main model initialization script
Run as follows: PARAMDIR=tests/... $NRNHOME/$NRNARCH/bin/nrngui -python main.py

Written by Shyam Kumar Sudhakar, Ivan Raikov, Tom Close, Rodrigo Publio, Daqing Guo, and Sungho Hong
Computational Neuroscience Unit, Okinawa Institute of Science and Technology, Japan
Supervisor: Erik De Schutter

Correspondence: Sungho Hong (shhong@oist.jp)

September 16, 2017

"""

import sys, os, platform, string, subprocess
import os.path
import mmap
from neuron import h
from optparse import OptionParser

# NMODL Mechanisms used in this model:
mechanisms = [
              "Granule_CL",
              "Golgi_CL",
              "Synapses",
              "Presynaptic_spike_generator",
              "Event_stream",
              "gap"
              ]


def build_mechanism(modeldir, mechanism, home, arch, verbose=True, cleanup=True):
    """Runs the commands necessary to build an NMODL mechanism"""
    cleancmd = ['rm', '-rf', arch]
    buildcmd = [os.path.normpath(os.path.join(home,arch,'bin','nrnivmodl'))]
    mechdir = os.path.join(modeldir, 'mechanisms', mechanism)

    if verbose: print "building mechanisms in directory", mechdir
    os.chdir(mechdir)
    if verbose: print "running command", cleancmd, "in directory", mechdir
    # Erases any old compiled files
    if cleanup:
       subprocess.call(cleancmd)
    if verbose: print "running command", buildcmd, "in directory", mechdir
    # Runs the build command
    subprocess.call(['nrnivmodl'])
    os.chdir(modeldir)


def build(NRNHOME, NRNARCH, paramdir=".", verbose=True, cleanup=False):
    """Build all mechanisms"""
    for mechanism in mechanisms:
        build_mechanism(os.getcwd(),
                        mechanism,
                        NRNHOME,
                        NRNARCH,
                        verbose,
                        cleanup)


def load_mechanism(modeldir, mechanism, arch, verbose=True):
    """Runs the commands necessary to load an NMODL mechanism"""
    mechdir = os.path.join(modeldir, 'mechanisms', mechanism)
    os.chdir(mechdir)
    if verbose: print "loading mechanisms in directory", mechdir
    # Loads the resulting shared library
    if sys.platform.startswith('win'):
        h.nrn_load_dll(os.path.normpath(os.path.join(mechdir, arch, 'libs', 'libnrnmech.dll')))
    else:
        h.nrn_load_dll(os.path.normpath(os.path.join(mechdir, arch, '.libs', 'libnrnmech.so')))
    os.chdir(modeldir)


def init_populations(NRNHOME, NRNARCH, verbose=True, paramdir=".", iterparam=None, save_populations=False):
    """ Instantiate the populations"""
    for mechanism in mechanisms:
        load_mechanism(os.getcwd(), mechanism, NRNARCH, verbose)

    h.load_file(1, "nrngui.hoc")

    if (iterparam and verbose):
        print "Using iterparam:", iterparam

    # If iterparam is specified, it must be of the form:
    #
    # name:path:index
    #
    # Where name is the parameter name, path is a file containing
    # multiple values for this parameter, index is the row number of
    # the parameter value we wish to use for this run of the model.
    if iterparam:
        (iter_name, iter_path, iter_index) = string.split(iterparam, ':')
        iter_index = int(iter_index)
        with open(iter_path) as f:
            data = f.readlines()
        iter_cell_ids = string.split(data[iter_index], ' ')
        if verbose: print "Using cell ids %s from file %s row %d" % (iter_name, iter_path, iter_index)
        h('objref iter_cell_ids')
        h('iter_cell_ids = new Vector()')
        for cell_id in iter_cell_ids:
            h.iter_cell_ids.append(cell_id)

    # Define the locations of dat files
    h('strdef pathwidthz')
    h('pathwidthz="'+(os.path.normpath(paramdir+"/widthz.dat"))+'"')
    h('strdef pathwidthy')
    h('pathwidthy="'+(os.path.normpath(paramdir+"/widthy.dat"))+'"')
    h('strdef pathl')
    h('pathl="'+(os.path.normpath(paramdir+"/l.dat"))+'"')
    h('strdef pathGLpoints')
    h('pathGLpoints="'+(os.path.normpath(paramdir+"/GLpoints.dat"))+'"')
    h('strdef pathdatasp')
    h('pathdatasp="'+(os.path.normpath(paramdir+"/datasp.dat"))+'"')
    h('strdef pathactiveMfibres1')
    h('pathactiveMfibres1="'+(os.path.normpath(paramdir+"/activeMfibres1.dat"))+'"')
    h('strdef pathMFcoordinates')
    h('pathMFcoordinates="'+(os.path.normpath(paramdir+"/MFCr.dat"))+'"')


    # Model utility functions
    h.xopen("utilities.hoc")

    if verbose: print "Loading object declarations"
    h.xopen("objects.hoc")

    # Create Golgi cell population
    if 'Golgi_model' in h.__dict__.keys():
        Golgi_template = "%s_template.hoc" % (h.Golgi_model)
    else:
        Golgi_template = "Golgi_template_CL.hoc"
    if verbose: print "Loading Golgi cell template %s" % Golgi_template
    h.xopen(os.path.normpath(os.getcwd()+"/templates/%s" % Golgi_template))
    if verbose: print "Loading Golgi cell population"
    h.xopen(os.path.normpath(os.getcwd()+"/populations/GolgiPopulation.hoc"))

    # Create granule cell population
    if verbose: print "Loading Granule cell template"
#    h.xopen(os.path.normpath(os.getcwd()+"/templates/Granule_template.hoc"))
    h.xopen(os.path.normpath(os.getcwd()+"/templates/Granule_template_CL.hoc"))
    if verbose: print "Loading Granule cell population"
    h.xopen(os.path.normpath(os.getcwd()+"/populations/GranulePopulation.hoc"))

    # Create the mossy fiber input source
    if verbose: print "Loading Mossy fiber template"
    h.xopen(os.path.normpath(os.getcwd()+"/templates/MF_template.hoc"))
    if verbose: print "Creating Mossy fiber population"
    h.xopen(os.path.normpath(os.getcwd()+"/populations/MFPopulation.hoc"))

    if h.MLplug==1: # This part is reserved for future development
        # Create Stellate Cell population
        if verbose: print "Reading Stellate cell template"
        h.xopen("gap.hoc")
        h.xopen(os.path.normpath(os.getcwd()+"/templates/SC_template.hoc"))
        h.xopen(os.path.normpath(os.getcwd()+"/populations/StellatePopulation.hoc"))

        # Create Basket Cell population
        if verbose: print "Reading Basket cell template"

        h.xopen(os.path.normpath(os.getcwd()+"/templates/BC1_Template.hoc"))
        h.xopen(os.path.normpath(os.getcwd()+"/populations/BasketPopulation1.hoc"))

        if verbose: print "Cell populations created"

    #if verbose: print "Loading Purkinje cell template"
    #h.xopen(os.path.normpath(os.getcwd()+"/templates/Purkinje_template.hoc"))
    #h.xopen(os.path.normpath(os.getcwd()+"/templates/Pur.py"))

    # if verbose: print "Loading Purkinje cell population"
    #h.xopen(os.path.normpath(os.getcwd()+"/populations/PurkinjePopulation.hoc"))
    # h('objref PCcoordinates')
    #PCPopStartIndex = h.GolgiPop.nCells+h.GranulePop.nCells+h.MossyPop.nCells+h.StellatePop.nCells+h.BasketPop.nCells
    #h.PCcoordinates,PCs,PCPopEndIndex = make_PurkinjePop(h.gseed+4,PCPopStartIndex)

    print 'gseed =', h.gseed

    # Writing population coordinates data
    if save_populations:
        if verbose: print "Writing population data"
        h.xopen("save_population_data.hoc")


def init_connections(NRNHOME, NRNARCH, verbose=True):
    """ Create the network connections """
    if verbose: print "Creating Network Connections"

    pc = h.ParallelContext()

    if ((h.MF_GC_con > 0) | (h.PF_GoC_con > 0) | (h.GoC_GoC_inh_con > 0)) :

        h('objref vMFtoGCsources')
        h('strdef pathMFtoGCsources')
        h('pathMFtoGCsources="'+(os.path.normpath("MFtoGCsources%d.dat" % int(pc.id())))+'"')

        h('objref vMFtoGCtargets')
        h('strdef pathMFtoGCtargets')
        h('pathMFtoGCtargets="'+(os.path.normpath("MFtoGCtargets%d.dat" % int(pc.id())))+'"')

        h('objref vMFtoGCdistances')
        h('strdef pathMFtoGCdistances')
        h('pathMFtoGCdistances="'+(os.path.normpath("MFtoGCdistances%d.dat" % int(pc.id())))+'"')

        h('strdef pathPFtoGoCsources')
        h('pathPFtoGoCsources="'+(os.path.normpath("PFtoGoCsources%d.dat" % int(pc.id())))+'"')

        h('strdef pathPFtoGoCtargets')
        h('pathPFtoGoCtargets="'+(os.path.normpath("PFtoGoCtargets%d.dat" % int(pc.id())))+'"')

        h('strdef pathPFtoGoCdistances')
        h('pathPFtoGoCdistances="'+(os.path.normpath("PFtoGoCdistances%d.dat" % int(pc.id())))+'"')

        h('strdef pathPFtoGoCsegments')
        h('pathPFtoGoCsegments="'+(os.path.normpath("PFtoGoCsegments%d.dat" % int(pc.id())))+'"')

        h('strdef pathAAtoGoCsources')
        h('pathAAtoGoCsources="'+(os.path.normpath("AAtoGoCsources%d.dat" % int(pc.id())))+'"')

        h('strdef pathAAtoGoCtargets')
        h('pathAAtoGoCtargets="'+(os.path.normpath("AAtoGoCtargets%d.dat" % int(pc.id())))+'"')

        h('strdef pathAAtoGoCdistances')
        h('pathAAtoGoCdistances="'+(os.path.normpath("AAtoGoCdistances%d.dat" % int(pc.id())))+'"')

        h('strdef pathAAtoGoCsegments')
        h('pathAAtoGoCsegments="'+(os.path.normpath("AAtoGoCsegments%d.dat" % int(pc.id())))+'"')

        h('strdef pathGoCtoGoCsources')
        h('pathGoCtoGoCsources="'+(os.path.normpath("GoCtoGoCsources.dat"))+'"')

        h('strdef pathGoCtoGoCtargets')
        h('pathGoCtoGoCtargets="'+(os.path.normpath("GoCtoGoCtargets.dat"))+'"')

        h('strdef pathGoCtoGoCdistances')
        h('pathGoCtoGoCdistances="'+(os.path.normpath("GoCtoGoCdistances.dat"))+'"')

        h('objref vGoCtoGoCgapsources')
        h('strdef pathGoCtoGoCgapsources')
        h('pathGoCtoGoCgapsources="'+(os.path.normpath("GoCtoGoCgapsources.dat"))+'"')

        h('objref vGoCtoGoCgaptargets')
        h('strdef pathGoCtoGoCgaptargets')
        h('pathGoCtoGoCgaptargets="'+(os.path.normpath("GoCtoGoCgaptargets.dat"))+'"')

        h('objref vGoCtoGoCgapdistances')
        h('strdef pathGoCtoGoCgapdistances')
        h('pathGoCtoGoCgapdistances="'+(os.path.normpath("GoCtoGoCgapdistances.dat"))+'"')

        if verbose: print "Reloading the sorted GC and GoC coordinates"

        coordinate_files = ["GCcoordinates1",
                            "GoCcoordinates1",
                            "Tcoordinates1",
                            "Adendcoordinates1",
                            "Bdendcoordinates1",
                            ]

        for n in coordinate_files:
            h('strdef path%s' % n)
            h('objref file%s' % n)

        h('pathGCcoordinates1="'+(os.path.normpath("GCcoordinates.sorted.dat"))+'"')

        h('pathTcoordinates1="'+(os.path.normpath("GCTcoordinates.sorted.dat"))+'"')

        h('pathGoCcoordinates1="'+(os.path.normpath("GoCcoordinates.sorted.dat"))+'"')

        h('pathAdendcoordinates1="'+(os.path.normpath("GoCadendcoordinates.sorted.dat"))+'"')

        h('pathBdendcoordinates1="'+(os.path.normpath("GoCbdendcoordinates.sorted.dat"))+'"')


        for n in coordinate_files:
            h('file%s = new File(path%s)' % (n, n))
            h('file%s.ropen()' % n)

        h('GranulePop.GCcoordinates.scanf(fileGCcoordinates1,GranulePop.numGC,3)')
        h('GranulePop.Tcoordinates.scanf(fileTcoordinates1,GranulePop.numGC,3)')
        h('GolgiPop.GoCcoordinates.scanf(fileGoCcoordinates1,GolgiPop.numGoC,3)')
        h('GolgiPop.Adendcoordinates.scanf(fileAdendcoordinates1,GolgiPop.numGoC,3*(GoC_nDendML))')
        h('GolgiPop.Bdendcoordinates.scanf(fileBdendcoordinates1,GolgiPop.numGoC,3*(numDendGolgi-GoC_nDendML))')


        for n in coordinate_files:
            h('file%s.close()' % n)



    # conectivity rules and synaptic delays
    if verbose: print "Loading conectivity rules and synaptic delays"
    h.xopen("enPassage.hoc")

    if verbose: print "Creating MF to GoC"

    h.xopen("netconMFtoGoC.hoc")
    w = h.ncMFtoGoC[0].enP

    if verbose: print "Calling connectPops4"

     # w.connectPops4(h.MossyPop,h.GranulePop,h.GolgiPop,h.MFtoGoCzone,h.MFtoGCzone,h.TS,h.StellatePop,h.BasketPop,h.PCcoordinates,PCPopStartIndex)

    if verbose: print "Creating MF to GC"

    h.xopen("netconMFtoGC.hoc")

    if verbose: print "Creating GoC to GC"

    h.xopen("netconGoCtoGC.hoc")

    if verbose: print "Creating GC to GC"

    h.xopen("netconGCtoGoC.hoc")
    h.xopen("netconGoCtoGoC.hoc")

    if h.MLplug==1: # This part is reserved for future development
        h.xopen("trial.hoc")
        if verbose: print "Creating PF to SC"
        h.xopen("netconPFtoSC.hoc")
        if verbose: print "Creating PF to BC"
        h.xopen("netconPFtoBC.hoc")
        if verbose: print "Creating SC to PC"
        #h.xopen("netconSCtoPC.hoc")
        if verbose: print "Creating BC to PC"
        #h.xopen("netconBCtoPC.hoc")
        #ZcorPC = h.Vector()
        #ZcorPC = h.PCcoordinates.getcol(2)
        #XcorPC = h.Vector()
        #XcorPC = h.PCcoordinates.getcol(0)

    GcorZ = h.Vector()
    GcorZ = h.GranulePop.Tcoordinates.getcol(2)
    GcorX = h.Vector()
    GcorX = h.GranulePop.Tcoordinates.getcol(0)
    Coordinates = h.List()
    Coordinates.append(GcorX)
    Coordinates.append(GcorZ)
    #Coordinates.append(XcorPC)
    #Coordinates.append(ZcorPC)

    h('objref NC')
    h('objref Receplist')
    h('objref bundle1')
    h('objref bundle2')

    #h.NC,h.Receplist,h.bundle1,h.bundle2 = netconPFtoPC(h.Scale_factor,h.PCLdepth,Coordinates,h.GLdepth,h.step_time,h.GranulePop.startindex,PCPopStartIndex)

    if verbose: print "Connectivity patterns created"

    check ()
    return h.NC,h.Receplist,h.bundle1,h.bundle2


# Run the simulation
def run ():
    h.xopen("run.hoc")

def check ():
  pc1 = h.ParallelContext()
  nhost = int(pc1.nhost())
  for rank in range(nhost):
     if rank==pc1.id():
       fname = "CheckCoordinates.dat"+str(pc1.id())
       verify = open(fname,'w')
       nGC = int(h.GranulePop.nCells)
       for i in range(nGC):
		#GCx=h.GranulePop.GCcoordinates.x[i][0]
		#GCy=h.GranulePop.GCcoordinates.x[i][1]
		GCz=list()
		GCz=h.GranulePop.GCcoordinates.getrow(i)
		for item in GCz:

		 verify.write("%d\n" % item)
       verify.close()


def linecount(buf):
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines



optparser = OptionParser()
optparser.add_option("--iterparam", dest="iterparam",
                     help='specify parameter set to be iterated over',
                     metavar="ITERPARAM")
optparser.add_option("--paramdir", dest="paramdir",
                     help='specify directory from which to load parameter sets',
                     metavar="PARAMDIR")
optparser.add_option('--verbose', action='store_true',
                     help='verbose mode')
optparser.add_option('--cleanup', action='store_true',
                     help='remove compiled mechanisms before building them')
optparser.add_option('--build', action='store_true',
                     help='build mechanisms and exit')
optparser.add_option('--initpop', action='store_true',
                     help='instantiate the populations and exit')
optparser.add_option('--init', action='store_true',
                     help='instantiate the network and exit')
(options,args)=optparser.parse_args(sys.argv[3:])


paramdir=options.paramdir
if paramdir is None:
    paramdir=os.getenv('PARAMDIR')
    print 'paramdir:', paramdir
iterparam=options.iterparam
if iterparam is None:
    iterparam=os.getenv('ITERPARAM')
    print 'iterparam:', iterparam
verbose=options.verbose
if verbose is None:
    verbose=os.getenv('VERBOSE')
    print 'verbose:', verbose

NRNHOME=os.getenv('NRNHOME')
if NRNHOME==None:
    bindir=os.path.dirname(os.path.abspath(sys.argv[0]))
    NRNHOME=os.path.normpath(bindir+'/../..')

NRNARCH=os.getenv('NRNARCH')
if NRNARCH==None:
    NRNARCH=platform.machine()

if verbose: print "NEURON home is", NRNHOME
if verbose: print "NEURON arch is", NRNARCH
h('objref NCl')
h('objref RCl')
h('objref l2')
h('objref l1')

# Load global parameters
pathparams=os.path.normpath(os.path.join(paramdir, "Parameters.hoc"))
if verbose:
    print "Loading global parameters from", pathparams
h.xopen(pathparams)

if 'Golgi_model' in h.__dict__.keys():
    mechanisms.append(h.Golgi_model)

if options.build:
    build(NRNHOME,NRNARCH,paramdir=paramdir,verbose=verbose,cleanup=options.cleanup)
    sys.exit()
elif options.initpop:
    init_populations(NRNHOME,NRNARCH,verbose=verbose,paramdir=paramdir,iterparam=iterparam,save_populations=True)
    sys.exit()
elif options.init:
    init_populations(NRNHOME,NRNARCH,verbose=verbose,paramdir=paramdir,iterparam=iterparam)
    init_connections(NRNHOME,NRNARCH,verbose=verbose)
    sys.exit()
else:
    init_populations(NRNHOME,NRNARCH,verbose=verbose,paramdir=paramdir,iterparam=iterparam)
    h.NCl,h.RCl,h.l1,h.l2 = init_connections(NRNHOME,NRNARCH,verbose=verbose)
    #print h.NCl
    #tgt = h.NCl.o(1).syn
    #print tgt
    run()
