// a more general monitor, for things like <sinPhiH> or helicity
// - this reads DST files or skim files
// - can be run on slurm
// - note: search for 'CUT' to find which cuts are applied

import org.jlab.io.hipo.HipoDataSource
import org.jlab.clas.physics.Particle
import org.jlab.groot.data.H1F
import org.jlab.groot.data.H2F
import org.jlab.groot.data.GraphErrors
import org.jlab.groot.data.TDirectory
import org.jlab.clas.physics.LorentzVector
import org.jlab.detector.base.DetectorType
import java.lang.Math.*
import org.jlab.clas.timeline.util.Tools
Tools T = new Tools()

// CONSTANTS
def MIN_NUM_SCALERS = 2000   // at least this many scaler readouts per time bin // 2000 is roughly a DST 5-file
def NBINS           = 50     // number of bins in some histograms
def SECTORS         = 0..<6  // sector range
def ECAL_ID         = DetectorType.ECAL.getDetectorId() // ECAL detector ID
// debugging settings
def VERBOSE = false  // enable extra log messages, for debugging
def LIMITER = 0      // if nonzero, only analyze this many DST files (for quick testing and debugging)
def AUXFILE = false  // enable auxfile production, an event-by-event table (a large text file)

// function to print a debugging message
def printDebug = { msg -> if(VERBOSE) println "[DEBUG]: $msg" }

// ARGUMENTS
def inHipoType = "dst" // options: "dst", "skim"
def runnum = 0
if(args.length<2) {
  System.err.println """
  USAGE: run-groovy ${this.class.getSimpleName()}.groovy [HIPO directory or file] [output directory] [type(OPTIONAL)] [runnum(OPTIONAL)]
         REQUIRED parameters:
           - [HIPO directory or file] should be a directory of HIPO files
             or a single hipo file (depends on [type]: use 'dst' for directory
             or 'skim' for file)
           - [output directory] output directory for the produced files
         OPTIONAL parameters:
           - [type] can be 'dst' or 'skim' (default is '$inHipoType')
           - [runnum] the run number; if not specified, it will be obtained from RUN::config

  """
  System.exit(101)
}
def inHipo = args[0]
def outDir = args[1]
if(args.length>=3) inHipoType = args[2]
if(args.length>=4) runnum     = args[3].toInteger()
System.println """
inHipo     = $inHipo
outDir     = $outDir
inHipoType = $inHipoType"""

// get hipo file names
def inHipoList = []
if(inHipoType=="dst") {
  def inHipoDirObj = new File(inHipo)
  def inHipoFilter = ~/.*\.hipo/
  inHipoDirObj.traverse( type: groovy.io.FileType.FILES, nameFilter: inHipoFilter ) {
    if(it.size()>0) inHipoList << inHipo+"/"+it.getName()
  }
  inHipoList.sort(true)
  if(inHipoList.size()==0) {
    System.err.println "ERROR: no hipo files found in this directory"
    System.exit(100)
  }
}
else if(inHipoType=="skim") { inHipoList << inHipo }
else {
  System.err.println "ERROR: unknown inHipoType setting"
  System.exit(100)
}

// limiter: use this to only analyse a few DST files, for quicker testing
if(LIMITER>0) {
  inHipoList = inHipoList[0..(LIMITER-1)]
  System.err.println("WARNING WARNING WARNING: LIMITER ENABLED, we will only be analyzing ${LIMITER} DST files, and not all of them; this is for testing only!")
}

// get runnum; assumes all HIPO files have the same run number
if(runnum<=0)
  runnum = T.getRunNumber(inHipoList.first())
if(runnum<=0)
  System.exit(100)
System.println "runnum     = $runnum"


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
// RUN GROUP DEPENDENT SETTINGS ///////////////////////////////////////////////////////////////////

def RG = "unknown"
if(runnum>=4763 && runnum<=5001) RG="RGA" // early period
else if(runnum>=5032 && runnum<=5262) RG="RGA" // inbending1
else if(runnum>=5300 && runnum<=5666) RG="RGA" // inbending1 + outbending
else if(runnum>=5674 && runnum<=6000) RG="RGK" // 6.5+7.5 GeV
else if(runnum>=6120 && runnum<=6604) RG="RGB" // spring 19
else if(runnum>=6616 && runnum<=6783) RG="RGA" // spring 19
else if(runnum>=11093 && runnum<=11300) RG="RGB" // fall 19
else if(runnum>=11323 && runnum<=11571) RG="RGB" // winter 20
else if(runnum>=12210 && runnum<=12951) RG="RGF" // spring+summer 20
else if(runnum>=15019 && runnum<=15884) RG="RGM"
else if(runnum>=16043 && runnum<=16772) RG="RGC" // summer 22
else System.err.println "WARNING: unknown run group; using default run-group-dependent settings (see monitorRead.groovy)"
println "rungroup = $RG"

// beam energy
// - hard-coded; could instead get from RCDB, but sometimes it is incorrect
def EBEAM = 10.6041 // RGA default
if(RG=="RGA") {
  if(runnum>=6616 && runnum<=6783) EBEAM = 10.1998 // spring 19
  else EBEAM = 10.6041
}
else if(RG=="RGB") {
  if(runnum>=6120 && runnum<=6399) EBEAM = 10.5986 // spring
  else if(runnum>=6409 && runnum<=6604) EBEAM = 10.1998 // spring
  else if(runnum>=11093 && runnum<=11283) EBEAM = 10.4096 // fall
  else if(runnum>=11284 && runnum<=11300) EBEAM = 4.17179 // fall BAND_FT
  else if(runnum>=11323 && runnum<=11571) EBEAM = 10.3894 // winter (RCDB may still be incorrect)
  else System.err.println "ERROR: unknown beam energy"
}
else if(RG=="RGC") {
  if(runnum>=16010 && runnum<=16078) EBEAM = 2.21
  else if(runnum>=16079) EBEAM = 10.55
  else System.err.println "ERROR: unknown beam energy"
}
else if(RG=="RGK") {
  if(runnum>=5674 && runnum<=5870) EBEAM = 7.546
  else if(runnum>=5875 && runnum<=6000) EBEAM = 6.535
  else System.err.println "ERROR: unknown beam energy"
}
else if(RG=="RGF") {
  if     (runnum>=12210 && runnum<=12388) EBEAM = 10.389 // RCDB may still be incorrect
  else if(runnum>=12389 && runnum<=12443) EBEAM =  2.186 // RCDB may still be incorrect
  else if(runnum>=12444 && runnum<=12951) EBEAM = 10.389 // RCDB may still be incorrect
  else System.err.println "ERROR: unknown beam energy"
}
else if(RG=="RGM") {
  if     (runnum>=15013 && runnum<=15490) EBEAM = 5.98636
  else if(runnum>=15533 && runnum<=15727) EBEAM = 2.07052
  else if(runnum>=15728 && runnum<=15784) EBEAM = 4.02962
  else if(runnum>=15787 && runnum<=15884) EBEAM = 5.98636
  else System.err.println "ERROR: unknown beam energy"
}

/* gated FC charge determination: `FCmode`
 * - 0: DAQ-gated FC charge is incorrect
 *   - recharge option was likely OFF during cook, and should be turned on
 *   - re-calculates DAQ-gated FC charge as: `ungated FC charge * average livetime`
 *   - typically applies to data cooked with version 6.5.3 or below
 *   - typically used as a fallback if there are "spikes" in gated charge when `FCmode==1`
 * - 1: DAQ-gated FC charge can be trusted
 *   - recharge option was either ON or did not need to be ON
 *   - calculate DAQ-gated FC charge directly from `RUN::scaler:fcupgated`
 *   - if you find `fcupgated > fcup(un-gated)`, then most likely the recharge option was OFF
 *     but should have been ON, and `FCmode=0` should be used instead
 * - 2: special case
 *   - calculate DAQ-gated FC charge from `REC::Event:beamCharge`
 *   - useful if `RUN::scaler` is unavailable
 */
def FCmode = 1 // default assumes DAQ-gated FC charge can be trusted
if(RG=="RGM") {
  FCmode = 1
  if(runnum>=15015 && runnum<=15199) {
    FCmode = 2 // no scalars read out in this range probably dosen't work anyway
  }
}
/* PASS 1 FCmode settings:
if(RG=="RGA") {
  FCmode=1;
  if(runnum==6724) FCmode=0; // fcupgated charge spike in file 230
}
else if(RG=="RGB") {
  FCmode = 1
  if( runnum in [6263, 6350, 6599, 6601, 11119] ) FCmode=0 // fcupgated charge spikes
}
else if(RG=="RGC") FCmode = 1
else if(RG=="RGK") FCmode = 0
else if(RG=="RGF") FCmode = 0
else if(RG=="RGM") {
  FCmode = 1
  if(runnum>=15015 && runnum<=15199) {
    FCmode = 2 // no scalars read out in this range probably dosen't work anyway
  }
}
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// make outut directories and define output file
"mkdir -p $outDir".execute()
def outHipo = new TDirectory()
outHipo.mkdir("/$runnum")
outHipo.cd("/$runnum")

// prepare time-binned output table for electron count and FC charge
def datfileName   = "$outDir/data_table_${runnum}.dat"
def datfile       = new File(datfileName)
def datfileWriter = datfile.newWriter(false)
// prepare auxiliary, event-by-event output table (for debugging)
def auxfileName
def auxfile
def auxfileWriter
if(AUXFILE) {
  auxfileName   = "$outDir/aux_table_${runnum}.dat"
  auxfile       = new File(auxfileName)
  auxfileWriter = auxfile.newWriter(false)
  auxfileWriter << [
    "runnum/I",
    "binnum/I",
    "on_bin_boundary/I", // actually a boolean
    "has_run_scaler_bank/I", // actually a boolean
    "evnum/L",
    "timestamp/L",
    "fc/D",
    "ufc/D",
  ].join(':') << '\n'
}

// define shared variables
def hipoEvent
def timeBins = [:]
def pidList = []
def particleBank
def FTparticleBank
def configBank
def eventBank
def calBank
def scalerBank
def disEleFound
def caseCountNtrigGT1 = 0
def caseCountNFTwithTrig = 0
def disElectronInTrigger
def disElectronInFT

// DIS kinematics
def Q2
def W
def nu
def x
def y
def z
def vecBeam = new LorentzVector(0, 0, EBEAM, EBEAM)
def vecTarget = new LorentzVector(0, 0, 0, 0.938)
def vecEle = new LorentzVector()
def vecH = new LorentzVector()
def vecQ = new LorentzVector()
def vecW = new LorentzVector()


///////////////////////////////////////////////////////////////////////////////////////////////////
// SUBROUTINES
///////////////////////////////////////////////////////////////////////////////////////////////////

// subroutine which returns a list of Particle objects of a certain PID
// - if `pid==11`, it will count the trigger electrons in FD and/or FT
def findParticles = { pid, binNum ->

  // get list of bank rows and Particle objects corresponding to this PID
  def rowList = pidList.findIndexValues{ it == pid }.collect{it as Integer}
  def particleList = rowList.collect { row ->
    new Particle(pid,*['px','py','pz'].collect{particleBank.getFloat(it,row)})
  }
  //println "pid=$pid  found in rows $rowList"

  // if looking for electrons, also count the number of trigger electrons,
  // and find the DIS electron
  if(pid==11) {

    // reset some vars
    def Emax = 0
    def Etmp
    disElectronInTrigger = false
    disElectronInFT = false
    def nTrigger = 0
    def nFT = 0
    disEleFound = false
    def disElectron
    def eleSec

    // loop over electrons from REC::Particle
    if(rowList.size()>0) {
      rowList.eachWithIndex { row,ind ->

        def status = particleBank.getShort('status',row)
        def chi2pid = particleBank.getFloat('chi2pid',row)

        // TRIGGER ELECTRONS (FD or CD) CUT
        // - must have status<0 and FD or CD bit(s) set
        // - must have |chi2pid|<3
        // - must appear in ECAL, to obtain sector
        if( status<0 &&
            ( Math.abs(status/1000).toInteger() & 0x2 ||
              Math.abs(status/1000).toInteger() & 0x4 ) &&
            Math.abs(chi2pid)<3
        ) {

          // get sector
          def eleSecTmp = (0..calBank.rows()).collect{
            ( calBank.getShort('pindex',it).toInteger() == row &&
              calBank.getByte('detector',it).toInteger() == ECAL_ID ) ?
              calBank.getByte('sector',it).toInteger() : null
          }.find()

          // CUT for electron: sector must be defined
          if(eleSecTmp!=null) {

            nTrigger++ // count how many trigger electrons we looked at

            // CUT for electron: choose maximum energy electron (for triggers)
            // - choice is from both trigger and FT electron sets (see below)
            Etmp = particleList[ind].e()
            if(Etmp>Emax) {
              Emax = Etmp
              eleSec = eleSecTmp
              disElectronInTrigger = true
              disElectronInFT = false
              disElectron = particleList[ind]
            }

          } else {
            System.err.println "WARNING: found electron with unknown sector"
          }
        }

        // FT trigger electrons
        // - REC::Particle:status has FT bit
        // - must also appear in RECFT::Particle with status<0 and FT bit
        // - must have E > 300 MeV
        if( Math.abs(status/1000).toInteger() & 0x1 ) {
          if( FTparticleBank.rows() > row ) {
            def FTpid = FTparticleBank.getInt('pid',row)
            def FTstatus = FTparticleBank.getShort('status',row)
            if( FTpid==11 &&
                FTstatus<0 &&
                Math.abs(FTstatus/1000).toInteger() & 0x1 &&
                particleList[ind].e() > 0.3
            ) {

              nFT++ // count how many FT electrons we looked at

              // CUT for electron: maximum energy electron (for FT)
              // - choice is from both trigger and FT electron sets (see above)
              Etmp = particleList[ind].e()
              if(Etmp>Emax) {
                Emax = Etmp
                disElectronInFT = true
                disElectronInTrigger = false
                disElectron = particleList[ind]
              }

            }
          }
        }

      } // eo loop through REC::Particle
    } // eo if nonempty REC::Particle

    // calculate DIS kinematics and increment counters
    if(disElectronInTrigger || disElectronInFT) {

      // - calculate DIS kinematics
      // calculate Q2
      vecQ.copy(vecBeam)
      vecEle.copy(disElectron.vector())
      vecQ.sub(vecEle)
      Q2 = -1*vecQ.mass2()

      // calculate W
      vecW.copy(vecBeam)
      vecW.add(vecTarget)
      vecW.sub(vecEle)
      W = vecW.mass()

      // calculate x and y
      nu = vecBeam.e() - vecEle.e()
      x = Q2 / ( 2 * 0.938272 * nu )
      y = nu / EBEAM

      // CUT for electron: Q2 cut
      //if(Q2<2.5) return

      // - increment counters, and set `disEleFound`
      if(disElectronInTrigger && disElectronInFT) { // can never happen (failsafe)
        System.err.println "ERROR: disElectronInTrigger && disElectronInFT == 1; skip event"
        return
      }
      else if(disElectronInTrigger) {
        timeBins[binNum].nElec[eleSec-1]++
        disEleFound = true
      }
      else if(disElectronInFT) {
        timeBins[binNum].nElecFT++
        disEleFound = true
      }

      // increment 'case counters' (for studying overlap of trigger/FT cuts)
      // - case where there are more than one trigger electron in FD
      if(disElectronInTrigger && nTrigger>1)
        caseCountNtrigGT1 += nTrigger-1 // count number of unanalyzed extra electrons
      // - case where disElectron is in FT, but there are trigger electrons in FD
      if(disElectronInFT && nTrigger>0)
        caseCountNFTwithTrig += nTrigger // count number of unanalyzed trigger (FD) electrons

    } // eo if(disElectronInTrigger || disElectronInFT)
  } // eo if(pid==11)

  // return list of Particle objects
  return particleList
}



// subroutine to build a histogram
def buildHist(histName, histTitle, propList, runn, nb, lb, ub, nb2=0, lb2=0, ub2=0) {

  def propT = [
    'pip': 'pi+',
    'pim': 'pi-',
    'hp':  'hel+',
    'hm':  'hel-',
    'hu':  'hel?',
  ]

  def pn = propList.join('_')
  def pt = propList.collect{ propT.containsKey(it) ? propT[it] : it }.join(' ')
  if(propList.size()>0) { pn+='_'; }

  def sn = propList.size()>0 ? '_':''
  def st = propList.size()>0 ? ' ':''
  def hn = "${histName}_${pn}${runn}"
  def ht = "${pt} ${histTitle}"

  if(nb2==0) return new H1F(hn,ht,nb,lb,ub)
  else return new H2F(hn,ht,nb,lb,ub,nb2,lb2,ub2)
}


// subroutine to find the EARLIEST time bin for a given event number
// - if the event number is on a time-bin boundary, the earlier time bin will be returned
def findTimeBin = { evnum ->
  def s = timeBins.find{ evnum >= it.value.eventNumMin && evnum <= it.value.eventNumMax }
  if(s==null) {
    System.err.println "ERROR: cannot find time bin for event number $evnum"
    return -1
  }
  [ s.key, s.value ]
}


// subroutine to update a min and/or max value in a time bin (viz. FC charge start and stop)
def setMinMaxInTimeBin = { binNum, key, val ->
  valOld = timeBins[binNum][key]
  timeBins[binNum][key] = [
    valOld[0] == "init" ? val : [valOld[0], val].min(),
    valOld[1] == "init" ? val : [valOld[1], val].max()
  ]
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// DEFINE TIME BINS
///////////////////////////////////////////////////////////////////////////////////////////////////

defineTimeBins = { // in its own closure, so giant data structures are garbage collected
  // get list of tag1 event numbers
  printDebug "Begin tag1 event loop"
  def tag1events = []
  inHipoList.each { inHipoFile ->
    printDebug "Open HIPO file $inHipoFile"
    def reader = new HipoDataSource()
    reader.getReader().setTags(1)
    reader.open(inHipoFile)
    while(reader.hasEvent()) {
      hipoEvent = reader.getNextEvent()
      // printDebug "tag1 event bank list: ${hipoEvent.getBankList()}"
      if(hipoEvent.hasBank("RUN::scaler") && hipoEvent.getBank("RUN::scaler").rows()>0 &&
         hipoEvent.hasBank("RUN::config") && hipoEvent.getBank("RUN::config").rows()>0)
      {
        tag1events << [
          BigInteger.valueOf(hipoEvent.getBank("RUN::config").getInt('event',0)),
          BigInteger.valueOf(hipoEvent.getBank("RUN::config").getLong('timestamp',0))
        ]
      }
    }
    reader.close()
  }
  printDebug "Number of tag1 events: ${tag1events.size()}"

  // sort the events by event number
  tag1eventNumList = tag1events.sort(false){it[0]}.collect{it[0]}
  // check that we would get the same result, if we instead sorted by timestamp
  if(tag1eventNumList != tag1events.sort(false){it[1]}.collect{it[0]}) {
    System.err.println "ERROR: sorting tag1 events by event number is DIFFERENT than sorting by timestamp"
    System.exit(100)
  }

  // define the time bin boundaries: first, some sorting and transformations
  def timeBinBounds = tag1eventNumList
    .collate(MIN_NUM_SCALERS)   // partition into subsets, each with cardinality MIN_NUM_SCALERS (the last subset may be smaller)
    .collect{ it[0] }           // take the first event number of each subset
    .plus(tag1eventNumList[-1]) // append the final tag1 event number...
    .unique()                   // ...and make sure it's not just repeating the previous event number
    .collect{ [it, it] }        // double each element (since upper bound of bin N = lower bound of bin N+1)...
    .flatten()                  // ...and flatten it, since we are going to re-collate it below after adding the final bin boundaries
  // set the first bin boundary to 0; we'll fix it later after the main event loop
  timeBinBounds = [0] + timeBinBounds
  // set the last bin boundary to a high number, because the true highest event
  // number is not yet known; we'll fix it later after the main event loop
  timeBinBounds = timeBinBounds + [10**(Math.log10(timeBinBounds[-1]).toInteger()+2)] // two orders of magnitude above largest known event number
  // pair the elements to define the bin boundaries
  timeBinBounds = timeBinBounds.collate(2)
  // define the time bin objects, initializing additional fields
  timeBinBounds.eachWithIndex{ bounds, binNum ->
    timeBins[binNum] = [
      eventNumMin:  bounds[0],           // event number range
      eventNumMax:  bounds[-1],
      timestampMin: "init",              // timestamp range
      timestampMax: "init",
      nElec:        SECTORS.collect{0},  // number of electrons for each FD sector
      nElecFT:      0,                   // number of electrons for FT
      fcRange:      ["init", "init"],    // gated FC charge at the bin boundaries
      ufcRange:     ["init", "init"],    // ungated  ""             ""
      fcMinMax:     ["init", "init"],    // gated FC charge min,max (to check if they are within the boundaries set in `fcRange`)
      ufcMinMax:    ["init", "init"],    // ungated  ""             ""
      LTlist:       [],
      histTree:     [:],
    ]
  }
}
defineTimeBins()

// debug `timeBins` logging function (call it where you need it)
print_timeBinBounds = {
  println "TIME BINS =============================="
  println "@ #runnum/I:binnum/I:number_of_bins/I:evnum_min/L:evnum_max/L:timestamp_min/L:timestamp_max/L:num_events/L"
  timeBins.each{ binNum, timeBin ->
    def num_events = timeBin.eventNumMax - timeBin.eventNumMin
    if(binNum==0) {
      num_events++ // since first bin has no lower bound
    }
    println "@ ${runnum} ${binNum} ${timeBins.size()} ${timeBin.eventNumMin} ${timeBin.eventNumMax} ${timeBin.timestampMin} ${timeBin.timestampMax} ${num_events}"
  }
  println "END TIME BINS =========================="
}
// print_timeBinBounds()

// initialize min and max overall event numbers and timestamps
def overallMinEventNumber = timeBins[0].eventNumMax  // it will be smaller than first bin's max
def overallMaxEventNumber = timeBins[timeBins.size()-1].eventNumMin // it will be larger than last bin's min
def overallMinTimestamp   = "init"
def overallMaxTimestamp   = "init"

// initialize histograms for each time bin
printDebug "Initialize histograms for each time bin"
timeBins.each{ binNum, timeBin ->
  def partList = [ 'pip', 'pim' ]
  T.buildTree(timeBin.histTree, 'helic',     [['sinPhi'],partList,['hp','hm']],        { new H1F() })
  T.buildTree(timeBin.histTree, 'helic',     [['dist']],                               { new H1F() })
  T.buildTree(timeBin.histTree, 'helic',     [['scaler'],['chargeWeighted']],          { new H1F() })
  T.buildTree(timeBin.histTree, 'DIS',       [['Q2','W','x','y']],                     { new H1F() })
  T.buildTree(timeBin.histTree, "DIS",       [['Q2VsW']],                              { new H2F() })
  T.buildTree(timeBin.histTree, "inclusive", [partList,['p','pT','z','theta','phiH']], { new H1F() })
  // if(binNum==0) T.printTree(timeBin.histTree,{T.leaf.getClass()});

  timeBin.histTree.helic.dist         = buildHist('helic_dist','helicity',[],runnum,3,-1,2)
  timeBin.histTree.DIS.Q2             = buildHist('DIS_Q2','Q^2',[],runnum,2*NBINS,0,12)
  timeBin.histTree.DIS.W              = buildHist('DIS_W','W',[],runnum,2*NBINS,0,6)
  timeBin.histTree.DIS.x              = buildHist('DIS_x','x',[],runnum,2*NBINS,0,1)
  timeBin.histTree.DIS.y              = buildHist('DIS_y','y',[],runnum,2*NBINS,0,1)
  timeBin.histTree.DIS.Q2VsW          = buildHist('DIS_Q2VsW','Q^2 vs W',[],runnum,NBINS,0,6,NBINS,0,12)
  T.exeLeaves( timeBin.histTree.helic.sinPhi, {
    T.leaf = buildHist('helic_sinPhi','sinPhiH',T.leafPath,runnum,NBINS,-1,1)
  })
  timeBin.histTree.helic.scaler.chargeWeighted = buildHist('helic_scaler_chargeWeighted','FC-charge-weighted helicity',[],runnum,3,-1,2)
  T.exeLeaves( timeBin.histTree.inclusive, {
    def lbound=0
    def ubound=0
    if(T.key=='p')          { lbound=0; ubound=10 }
    else if(T.key=='pT')    { lbound=0; ubound=4 }
    else if(T.key=='z')     { lbound=0; ubound=1 }
    else if(T.key=='theta') { lbound=0; ubound=Math.toRadians(90.0) }
    else if(T.key=='phiH')  { lbound=-3.15; ubound=3.15 }
    T.leaf = buildHist('inclusive','',T.leafPath,runnum,NBINS,lbound,ubound)
  })

  T.exeLeaves( timeBin.histTree, {
    def histN = T.leaf.getName() + "_${binNum}"
    def histT = T.leaf.getTitle() + " :: timeBinNum=${binNum}"
    T.leaf.setName(histN)
    T.leaf.setTitle(histT)
  })

  // print the histogram names and titles
  // if(binNum==0) {
    println "---\nhistogram names and titles:"
    T.printTree(timeBin.histTree,{ T.leaf.getName() +" ::: "+ T.leaf.getTitle() })
    println "---"
  // }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN EVENT LOOP
///////////////////////////////////////////////////////////////////////////////////////////////////

def evCount = 0
def countEvent
printDebug "Begin main event loop"
inHipoList.each { inHipoFile ->

  // open HIPO file
  printDebug "Open HIPO file $inHipoFile"
  def reader = new HipoDataSource()
  reader.open(inHipoFile)

  // EVENT LOOP
  while(reader.hasEvent()) {
    hipoEvent = reader.getNextEvent()

    // get required banks
    particleBank   = hipoEvent.getBank("REC::Particle")
    eventBank      = hipoEvent.getBank("REC::Event")
    configBank     = hipoEvent.getBank("RUN::config")
    FTparticleBank = hipoEvent.getBank("RECFT::Particle")
    calBank        = hipoEvent.getBank("REC::Calorimeter")
    scalerBank     = hipoEvent.getBank("RUN::scaler")
    helScalerBank  = hipoEvent.getBank("HEL::scaler")

    // get event number
    def eventNum
    def timestamp
    if(configBank.rows()>0) {
      eventNum = BigInteger.valueOf(configBank.getInt('event',0))
      timestamp = BigInteger.valueOf(configBank.getLong('timestamp',0))
    }
    else if(hipoEvent.getBankList().length==1 && hipoEvent.getBankList().contains("COAT::config")) {
      printDebug "Skipping event which has only 'COAT::config' bank"
      continue
    }
    else {
      // System.err.println "WARNING: cannot get event number for event with no RUN::config bank; skipping this event; available banks: ${hipoEvent.getBankList()}"
      continue
    }
    if(eventNum==0) {
      // System.err.println "WARNING: found event with eventNum=0; banks: ${hipoEvent.getBankList()}"
      continue
    }

    // set overall min and max event numbers and timestamps
    overallMinEventNumber = [ overallMinEventNumber, eventNum].min()
    overallMaxEventNumber = [ overallMaxEventNumber, eventNum].max()
    if(overallMinTimestamp == "init") overallMinTimestamp = timestamp
    else overallMinTimestamp = [ overallMinTimestamp, timestamp ].min()
    if(overallMaxTimestamp == "init") overallMaxTimestamp = timestamp
    else overallMaxTimestamp = [ overallMaxTimestamp, timestamp ].max()

    // find the time bin that contains this event
    def (thisTimeBinNum, thisTimeBin) = findTimeBin(eventNum)
    if(thisTimeBinNum == -1) continue

    // get list of PIDs, with list index corresponding to bank row
    pidList = (0..<particleBank.rows()).collect{ particleBank.getInt('pid',it) }
    //println "pidList = $pidList"


    // get the FC charge and livetime, depending on `FCmode`
    // - also sets the min and max FC charges, for this time bin
    def lt
    def fc  = "init"
    def ufc = "init"
    if(scalerBank.rows()>0) {
      // ungated charge
      ufc = scalerBank.getFloat("fcup",0)
      setMinMaxInTimeBin(thisTimeBinNum, "ufcMinMax", ufc)
      // livetime
      lt = scalerBank.getFloat("livetime",0)
      if(lt>=0) { thisTimeBin.LTlist << lt }
      // gated charge (if trustworthy)
      if(FCmode==1) {
        fc = scalerBank.getFloat("fcupgated",0)
        setMinMaxInTimeBin(thisTimeBinNum, "fcMinMax", fc)
      }
    }
    if(FCmode==2 && eventBank.rows()>0) {
      // gated charge only
      fc = eventBank.getFloat("beamCharge",0)
      setMinMaxInTimeBin(thisTimeBinNum, "fcMinMax", fc)
    }

    // if this event is on a bin boundary, and it has `scalerBank`, update `fcRange` and `ufcRange`
    // FIXME: this will only work for FCmode==1; need to figure out how to handle the others
    def onBinBoundary = false
    if(eventNum == thisTimeBin.eventNumMax) {
      onBinBoundary = true
      if(scalerBank.rows()>0) { // must have scalerBank, so `fc` and `ufc` are set (we'll check if any `(u)fcRange` values are still "init" later)
        // events on the boundary are assigned to earlier bin; this FC charge is that bin's max charge
        thisTimeBin.fcRange[1]   = fc
        thisTimeBin.ufcRange[1]  = ufc
        thisTimeBin.timestampMax = timestamp
        // this FC charge is also the next bin's min charge
        def nextTimeBin = timeBins[thisTimeBinNum+1]
        if(nextTimeBin==null) { System.err.println "ERROR: found a time bin that has no subsequent bin, and is not the latest bin" }
        nextTimeBin.fcRange[0]   = fc
        nextTimeBin.ufcRange[0]  = ufc
        nextTimeBin.timestampMin = timestamp
        printDebug "event number ${eventNum} on upper boundary of bin ${thisTimeBinNum}, and assigned to that bin:"
        printDebug "  - gated charge:   ${fc}"
        printDebug "  - ungated charge: ${ufc}"
        printDebug "  - banks: ${hipoEvent.getBankList()}"
      }
    }
    if(eventNum == thisTimeBin.eventNumMin) {
      onBinBoundary = true
      System.err.println "ERROR: event number ${eventNum} on lower boundary of bin ${thisTimeBinNum}, and assigned to that bin; this shouldn't happen in the current binning scheme."
    }

    // dump event-level info to a text file
    if(AUXFILE) {
      auxfileWriter << [
        runnum,
        thisTimeBinNum,
        onBinBoundary ? 1 : 0,
        scalerBank.rows() > 0 ? 1 : 0,
        eventNum,
        timestamp,
        fc,
        ufc,
      ].join(' ') << '\n'
    }

    // get helicity and fill helicity distribution
    def helicity = 0 // if undefined, default to 0
    if(hipoEvent.hasBank("REC::Event") && eventBank.rows()>0) {
      helicity = eventBank.getByte('helicity',0)
      thisTimeBin.histTree.helic.dist.fill(helicity)
    }
    def helStr
    def helDefined
    switch(helicity) {
      case 1:  helStr='hp'; helDefined=true; break
      case -1: helStr='hm'; helDefined=true; break
      default: helDefined = false; helicity = 0; break
    }
    // get scaler helicity from `HEL::scaler`, and fill its charge-weighted distribution
    if(hipoEvent.hasBank("HEL::scaler")) {
      helScalerBank.rows().times{ row -> // HEL::scaler readouts "pile up", so there are multiple bank rows in an event
        def sc_helicity = helScalerBank.getByte("helicity", row)
        def sc_fc       = helScalerBank.getFloat("fcupgated", row) // helicity-latched FC charge
        thisTimeBin.histTree.helic.scaler.chargeWeighted.fill(sc_helicity, sc_fc)
      }
    }

    // get electron list, and increment the number of trigger electrons
    // - also finds the DIS electron, and calculates x,Q2,W,y,nu
    findParticles(11, thisTimeBinNum)

    // CUT: if a DIS electron was found by `findParticles`
    if(disEleFound) {

      // CUT for pions: Q2 and W and y and helicity
      if( Q2>1 && W>2 && y<0.8 && helDefined) {

        // get pions, calculate their kinematics and fill histograms
        countEvent = false
        [
          [ findParticles(211, thisTimeBinNum),  'pip' ],
          [ findParticles(-211, thisTimeBinNum), 'pim' ],
        ].each{ pionList, pionName ->

          pionList.each { part ->

            // calculate z
            vecH.copy(part.vector())
            z = T.lorentzDot(vecTarget,vecH) / T.lorentzDot(vecTarget,vecQ)

            // CUT for pions: particle z
            if(z>0.3 && z<1) {

              // calculate momenta, theta, phiH
              def p     = vecH.p()
              def pT    = Math.hypot( vecH.px(), vecH.py() )
              def theta = vecH.theta()
              def phiH  = T.planeAngle( vecQ.vect(), vecEle.vect(), vecQ.vect(), vecH.vect() )

              // CUT for pions: if phiH is defined
              if(phiH>-10000) {

                // fill histograms
                if(helDefined) {
                  thisTimeBin["histTree"]['helic']['sinPhi'][pionName][helStr].fill(Math.sin(phiH))
                }
                thisTimeBin["histTree"]['inclusive'][pionName]['p'].fill(p)
                thisTimeBin["histTree"]['inclusive'][pionName]['pT'].fill(pT)
                thisTimeBin["histTree"]['inclusive'][pionName]['z'].fill(z)
                thisTimeBin["histTree"]['inclusive'][pionName]['theta'].fill(theta)
                thisTimeBin["histTree"]['inclusive'][pionName]['phiH'].fill(phiH)

                // tell event counter that this event has at least one particle added to histos
                countEvent = true
              }
            }
          }
        }

        if(countEvent) {

          // fill event-level histograms
          thisTimeBin.histTree.DIS.Q2.fill(Q2)
          thisTimeBin.histTree.DIS.W.fill(W)
          thisTimeBin.histTree.DIS.x.fill(x)
          thisTimeBin.histTree.DIS.y.fill(y)
          thisTimeBin.histTree.DIS.Q2VsW.fill(W,Q2)

          // increment event counter
          evCount++
          if(evCount % 100 == 0) printDebug "found $evCount events which contain a pion"

        }
      }
    }

  } // end event loop
  reader.close()

} // end loop over hipo files


// correct the first and last time bins' event number ranges, and their FC charge ranges
def firstTimeBin = timeBins[0]
def lastTimeBin  = timeBins[timeBins.size()-1]
firstTimeBin.eventNumMin  = overallMinEventNumber
firstTimeBin.timestampMin = overallMinTimestamp
firstTimeBin.fcRange      = [ 0, 0 ] // unknown accumulated charge (NOTE: first scaler readout may have NEGATIVE FC charge); just set it to zero
firstTimeBin.fcMinMax     = [ 0, 0 ]
firstTimeBin.ufcRange     = [ 0, 0 ]
firstTimeBin.ufcMinMax    = [ 0, 0 ]
lastTimeBin.eventNumMax   = overallMaxEventNumber
lastTimeBin.timestampMax  = overallMaxTimestamp
lastTimeBin.fcRange       = [ 0, 0 ] // unknown absolute maximum of FC charge; just set it to zero
lastTimeBin.fcMinMax      = [ 0, 0 ]
lastTimeBin.ufcRange      = [ 0, 0 ]
lastTimeBin.ufcMinMax     = [ 0, 0 ]

// write final time bin's histograms
timeBins.each{ itBinNum, itBin ->

  // loop through histTree, adding histos to the hipo file;
  T.exeLeaves( itBin.histTree, {
    outHipo.addDataSet(T.leaf)
  })
  //println "write histograms:"; T.printTree(itBin.histTree,{T.leaf.getName()})


  // get accumulated ungated FC charge
  def ufcStart = 0
  def ufcStop  = 0
  if(itBinNum+1<timeBins.size()) { // unknown for last time bin, just let the charge be "zero"
    if(!itBin.ufcRange.contains("init")) {
      ufcStart = itBin.ufcRange[0]
      ufcStop  = itBin.ufcRange[1]
    } else {
      System.err.println "ERROR: no ungated FC charge for run ${runnum} time bin ${itBinNum}"
    }
  }

  // get accumulated gated FC charge
  def fcStart = 0
  def fcStop  = 0
  def aveLivetime = itBin.LTlist.size()>0 ? itBin.LTlist.sum() / itBin.LTlist.size() : 0
  if(itBinNum+1<timeBins.size()) { // unknown for last time bin, just let the charge be "zero"
    if(FCmode==0) {
      fcStart = ufcStart * aveLivetime // workaround method
      fcStop  = ufcStop  * aveLivetime // workaround method
    } else if(FCmode==1 || FCmode==2) {
      if(!itBin.fcRange.contains("init")) {
        fcStart = itBin.fcRange[0]
        fcStop  = itBin.fcRange[1]
      } else {
        System.err.println "ERROR: no gated FC charge for run ${runnum} time bin ${itBinNum}"
      }
    }
    if(fcStart>fcStop || ufcStart>ufcStop) {
      System.err.println "ERROR: faraday cup start > stop for run ${runnum} time bin ${itBinNum}"
    }
  }

  // write number of electrons and FC charge to datfile
  SECTORS.each{ sec ->
    datfileWriter << [ runnum, itBinNum ].join(' ') << ' '
    datfileWriter << [ itBin.eventNumMin, itBin.eventNumMax ].join(' ') << ' '
    datfileWriter << [ itBin.timestampMin, itBin.timestampMax ].join(' ') << ' '
    datfileWriter << [ sec+1, itBin.nElec[sec], itBin.nElecFT ].join(' ') << ' '
    datfileWriter << [ fcStart, fcStop, ufcStart, ufcStop, aveLivetime ].join(' ') << '\n'
  }
  printDebug " - charge for timeBin $itBinNum:"
  printDebug "   - event number range: [ ${itBin.eventNumMin}, ${itBin.eventNumMax} ]"
  printDebug "   - gated-FC charge:    [ $fcStart, $fcStop ]"
  printDebug "   - ungated-FC charge:  [ $ufcStart, $ufcStop ]"

  // print some stats
  /*
  def nElecTotal = itBin.nElec*.value.sum()
  println "\nnumber of trigger electrons: $nElecTotal"
  println """number of electrons that satisified FD trigger cuts, but were not analyzed...
  ...because they had subdominant E: $caseCountNtrigGT1
  ...because there was a higher-E electron satisfying FT cuts: $caseCountNFTwithTrig"""
  caseCountNtrigGT1=0
  caseCountNFTwithTrig=0
  */
}


// print the time bins
print_timeBinBounds()


// cross check: is each time bin's min and max FC charge within the FC charge values at the bin boundaries?
/*
prior to RGC, we had a 1 second clock such that the FC charge as a function of event number
is a bit non-monotonic, looking like

    charge
     |             /
     |            /
     |         /\/
     |        /
     |       /
     |    /\/
     |   /
     |  /
     +---------------- event num

If the bin boundary is on one of these non-monotonic jumps, the min or max FC charge within a bin
may be smaller or larger than the FC charge values at the bin boundaries
*/
def nonMonotonicityGr = new GraphErrors("nonMonotonicity_${runnum}_0") // one graph for all time bins, so just set time bin number to `0`
nonMonotonicityGr.setTitle("FC charge non-monotonicity vs. time bin")
timeBins.each{ itBinNum, itBin ->
  if(itBinNum+1 == timeBins.size()) return // can't cross check the last bin
  def (fc_lb,   fc_ub)   = itBin.fcRange
  def (fc_min,  fc_max)  = itBin.fcMinMax
  def (ufc_lb,  ufc_ub)  = itBin.ufcRange
  def (ufc_min, ufc_max) = itBin.ufcMinMax
  printDebug "FC charge cross check for bin ${itBinNum}"
  printDebug "  gated:   (lb,ub)   = [${fc_lb}, ${fc_ub}]"
  printDebug "           (min,max) = [${fc_min}, ${fc_max}]"
  printDebug "  ungated: (lb,ub)   = [${ufc_lb}, ${ufc_ub}]"
  printDebug "           (min,max) = [${ufc_min}, ${ufc_max}]"
  def chargeViaBounds = fc_ub  - fc_lb
  def chargeViaMinMax = fc_max - fc_min
  // not clear whether `chargeViaBounds` or `chargeViaMinMax` is the real charge, so we calculate their percent difference w.r.t. their mean:
  def ave = (chargeViaBounds + chargeViaMinMax) / 2.0
  def nonMonotonicity = ave==0 ? 0 : Math.abs(chargeViaBounds - chargeViaMinMax) / ave
  nonMonotonicityGr.addPoint(itBinNum, nonMonotonicity, 0.0, 0.0)
}
outHipo.addDataSet(nonMonotonicityGr)

// close output text files
datfileWriter.flush()
datfileWriter.close()
if(AUXFILE) {
  auxfileWriter.flush()
  auxfileWriter.close()
}

// write outHipo file
outHipoN = "$outDir/monitor_${runnum}.hipo"
File outHipoFile = new File(outHipoN)
if(outHipoFile.exists()) outHipoFile.delete()
outHipo.writeFile(outHipoN)
println("Wrote the following files:")
println(" - $outHipoN")
println(" - $datfileName")
if(AUXFILE) {
  println(" - $auxfileName")
}
