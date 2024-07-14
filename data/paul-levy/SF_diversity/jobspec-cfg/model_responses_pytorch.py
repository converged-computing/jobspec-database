import torch
import torch.nn as nn
from torch.utils import data as torchdata
from sklearn.model_selection import KFold

import numpy as np
import scipy.stats as ss
import time
import datetime
import sys, os
import warnings
from functools import partial
import pdb

import helper_fcns as hf
import helper_fcns_sfBB as hf_sfBB

torch.autograd.set_detect_anomaly(True)

#########
### setting to avoid overtaxing CPU?
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
###
#########

#########
### Some global things...
#########
torch.set_num_threads(1) # to reduce CPU usage - 20.01.26
force_earlyNoise = 0; # if None, allow it as parameter; otherwise, force it to this value (e.g. 0)
#force_earlyNoise = None; # if None, allow it as parameter; otherwise, force it to this value (e.g. 0)
recenter_norm = 0;
_schedule = False; # use scheduler or not??? True or False
singleGratsOnly = False; # True

fall2020_adj = 1; # 210121, 210206, 210222, 210226, 210304, 210308/11/12/14, 210321
spring2021_adj = 1; # further adjustment to make scale a sigmoid rather than abs; 2102622
if fall2020_adj:
  globalMin = 1e-10 # what do we "cut off" the model response at? should be >0 but small
  #globalMin = 1e-1 # what do we "cut off" the model response at? should be >0 but small
else:
  globalMin = 1e-6 # what do we "cut off" the model response at? should be >0 but small

#globalMin = 0;
globalMinDiv = 1e-10; # just for terms with division, to avoid by zero
fftMin = 1e-10; # was previously 1e-10 to avoid NaN/failure of backward pass?

modRecov = 0;
# --- for parameters that are transformed with sigmoids, what's the scalar in front of the sigmoid??
### WARNING: If you adjust _sigmoidRespExp, adjust the corresponding line in sfNormMod.forward (search for _sigmoidRespExp, used in ratio = ...)
_sigmoidRespExp = None; # 3 or None, as of 21.03.14
### WARNING: See above
_sigmoidSigma = 5; # if None, then we just use raw value; otherwise, we'll do a sigmoid transform
_sigmoidScale = 10
_sigmoidDord = 5;
_sigmoidGainNorm = 5;
# --- and a flag for whether or not to include the LGN filter for the gain control
#_LGNforNorm = 1;
### force_full datalist?
try:
  expDir = sys.argv[2];
  force_full = 0 if expDir == 'V1_BB/' else 1;
except:
  force_full = 1;

try:
  cellNum = int(sys.argv[1]);
except:
  cellNum = np.nan;
try:
  dataListName = hf.get_datalist(expDir, force_full=force_full, new_v1=True); # argv[2] is expDir
except:
  dataListName = None;

### Helper -- from Billy
def _cast_as_tensor(x, device='cpu', dtype=torch.float32):
    # needs to be float32 to work with the Hessian calculations
    #return torch.tensor(x, dtype=dtype, device=device) # per bill broderick
    return torch.as_tensor(x, dtype=dtype, device=device) # updated for expected lower CPU usage

def _cast_as_param(x, requires_grad=True):
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)

### Helper -- other

## rvc
def naka_rushton(params, cons):

  base, gain, expon, c50 = params;
  return torch.add(base, gain * torch.div(torch.pow(cons, expon), torch.pow(cons, expon) + torch.pow(c50, expon)));

def get_rvc_model(params, cons, whichRVC=0):
  ''' simply return the rvc model used in the fits (type 0; should be used only for LGN)
      --- from Eq. 3 of Movshon, Kiorpes, Hawken, Cavanaugh; 2005
  '''

  if whichRVC==1:
    return naka_rushton(params, cons);
  else: # default
    b = params[0]; k = params[1]; c0 = params[2];
    return torch.add(b, torch.mul(k, torch.log(1+torch.div(cons, c0))));

## sf tuning
def flexible_Gauss(params, stim_sf, minThresh=0.1, sigmoidValue=_sigmoidSigma):
    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3] if sigmoidValue is None else torch.mul(_cast_as_tensor(sigmoidValue), torch.sigmoid(params[3]));
    sigmaHigh       = params[4] if sigmoidValue is None else torch.mul(_cast_as_tensor(sigmoidValue), torch.sigmoid(params[4]));

    # Tuning function
    sf0   = torch.div(stim_sf, sfPref);
    sigma = sigmaLow.repeat(sf0.shape)
    #sigma = torch.full_like(sf0, _cast_as_tensor(sigmaLow.item())); # was sigmaLow.detach
    whereSigHigh = torch.where(sf0>1);
    sigma[whereSigHigh] = sigmaHigh;

    shape = torch.exp(-torch.div(torch.pow(torch.log(sf0),2), 2*torch.pow(sigma,2)))
                
    return torch.max(_cast_as_tensor(minThresh), respFloor + respRelFloor*shape);

def DoGsach(gain_c, r_c, gain_s, r_s, stim_sf):
  ''' Difference of gaussians as described in Sach's thesis
  [0] gain_c    - gain of the center mechanism
  [1] r_c       - radius of the center
  [2] gain_s    - gain of surround mechanism
  [3] r_s       - radius of surround
  '''
  dog = gain_c*np.pi*torch.pow(r_c,2)*torch.exp(-torch.pow(stim_sf*np.pi*r_c,2)) - gain_s*np.pi*torch.pow(r_s,2)*torch.exp(-torch.pow(stim_sf*np.pi*r_s,2));

  return dog;

def DiffOfGauss(gain, f_c, gain_s, j_s, stim_sf):
  ''' Difference of gaussians - as formulated in Levitt et al, 2001
  gain      - overall gain term
  f_c       - characteristic frequency of the center, i.e. freq at which response is 1/e of maximum
  gain_s    - relative gain of surround (e.g. gain_s of 0.5 says peak surround response is half of peak center response
  j_s       - relative characteristic freq. of surround (i.e. char_surround = f_c * j_s)
  '''

  ctr = torch.mul(gain, torch.exp(-torch.pow(stim_sf/f_c, 2)));
  sur = torch.mul(gain*gain_s, torch.exp(-torch.pow(stim_sf/(f_c*j_s),2)));
  dog = torch.sub(ctr, sur);

  return dog;

def get_descrResp(params, stim_sf, DoGmodel, minThresh=0.1):
  # returns only pred_spikes
  if DoGmodel == 0:
    #pred_spikes = flexible_Gauss(params, stim_sf=stim_sf, minThresh=minThresh);
    pred_spikes = flexible_Gauss(params, stim_sf=stim_sf, minThresh=minThresh);
  elif DoGmodel == 1:
    pred_spikes = DoGsach(*params, stim_sf=stim_sf);
  elif DoGmodel == 2:
    pred_spikes = DiffOfGauss(*params, stim_sf=stim_sf);
  return pred_spikes;

def organize_mean_perCond(trInf, resp):
  ''' Given trInf and resp, organize the mean per condition
      Design this to work for all experiments...
  '''
  nConds = 0;
  means = np.nan * np.zeros_like(resp);
  conRounds = np.round(trInf['con'], 3);
  unique_conSet = np.unique(conRounds, axis=0);
  sfRounds = np.round(trInf['sf'], 3);
  unique_sfSet = np.unique(sfRounds, axis=0);
  for conSet in unique_conSet:
    con_rows = np.where((conRounds == conSet).all(axis=1))[0];
    if conSet[0] == 0: # i.e. if the first stimulus is blank, then the first SF doesn't matter
      # Yes, this is just written for expInd == -1 but could be generalized to avoid needing an if statement...to do later on
      unique_sfsubSet = np.unique(sfRounds[:, 1:]);
      for sfSet in unique_sfsubSet:
        sf_rows = np.where((sfRounds[:, 1:] == sfSet).all(axis=1))[0]
        to_mean = np.intersect1d(con_rows, sf_rows);
        if len(to_mean) == 0:
          continue;
        means[to_mean, :] = np.mean(resp[to_mean, :], axis=0)
        nConds += 1;
    else:
      for sfSet in unique_sfSet:
        sf_rows = np.where((sfRounds == sfSet).all(axis=1))[0];
        to_mean = np.intersect1d(con_rows, sf_rows);
        if len(to_mean) == 0:
          continue;
        means[to_mean, :] = np.mean(resp[to_mean, :], axis=0)
        nConds += 1;
  #print('# conditions: %d' % nConds);
  return means;

## FFT
def fft_amplitude(fftSpectrum, stimDur):
    ''' given an fftSpectrum (and assuming all other normalization has taken place), we double the non-DC frequencies and return
        only the DC and positive frequencies; we also convert these values into rates (i.e. spikes or power per second)

        normalization: since this code is vectorized, the length of each signal passed into np.fft.fft is 1
        i.e. an array whose length is one, with the array[0] having length stimDur/binWidth
        Thus, the DC amplitude is already correct, i.e. not in need of normalization by nSamples

        But, remember that for a real signal like this, non-DC amplitudes need to be doubled - we take care of that here       
    '''
    # Note that fftSpectrum will be [nTr x nFr]
    nyquist = [np.int32(x.shape[1]/2) for x in fftSpectrum];
    correctFFT = [];
    for i, spect in enumerate(fftSpectrum):
      allFFT = spect[:, 0:nyquist[i]+1]; # include nyquist 
      # since we cannot modify in-place, we simply double all of the non-DC amplitudes, then grab the unmodified DC, and create a new tensor accordingly
      nonDCvals = 2*allFFT[:, 1:nyquist[i]+1];
      currFFT = torch.cat((allFFT[:,0].unsqueeze(dim=1), nonDCvals), dim=1) # will be [nTr x nyquist+1]
      # note the divison by stimDur; our packaging of the psth when we call np.fft.fft means that the baseline is each trial
      # is one second; i.e. amplitudes are rates IFF stimDur = 1; here, we divide by stimDur to ensure all trials/psth
      # are true rates
      currFFT = torch.div(currFFT, _cast_as_tensor(stimDur));
      correctFFT.append(currFFT);
    
    return correctFFT;   

def spike_fft(psth, tfs = None, stimDur = None, binWidth=1e-3, inclPhase=0):
    ''' given a psth (and optional list of component TFs), compute the fourier transform of the PSTH
        if the component TFs are given, return the FT power at the DC, and at all component TFs
        NOTE: spectrum, rel_amp are rates (spks/s)
              full_fourier is unprocessed, in that regard
        
        normalization: since this code is vectorized, the length of each signal passed into np.fft.fft is 1
        i.e. an array whose length is one, with the array[0] having length stimDur/binWidth
        Thus, the DC amplitude is already correct, i.e. not in need of normalization by nSamples
        But, remember that for a real signal like this, non-DC amplitudes need to be doubled - we take care of that here 

    '''

    full_fourier = [torch.rfft(x, signal_ndim=1, onesided=False) for x in psth];
    epsil = _cast_as_tensor(fftMin); #1e-10;
    if inclPhase:
      # NOTE: I have checked that the amplitudes (i.e. just "R" in polar coordinates, 
      # -- computed as sqrt(x^2 + y^2)) are ...
      # -- equivalent when derived from with_phase as in spectrum, below
      with_phase = fft_amplitude(full_fourier, stimDur); # passing in while still keeping the imaginary component (so that we can back out phase)
    # Frustrating --> However, sqrt(0) fails in the backward pass, so we add a small value, and then subtract if off...
    full_fourier = [torch.sqrt(epsil + torch.add(torch.pow(x[:,:,0], 2), torch.pow(x[:,:,1], 2))) - torch.sqrt(epsil) for x in full_fourier]; # just get the amplitude
    spectrum = fft_amplitude(full_fourier, stimDur);

    if tfs is not None:
      try:
        if 'torch' in str(type(tfs)):
          tf_as_ind = _cast_as_tensor(hf.tf_to_ind(tfs.numpy(), stimDur), dtype=torch.long); # if 1s, then TF corresponds to index; if stimDur is 2 seconds, then we can resolve half-integer frequencies -- i.e. 0.5 Hz = 1st index, 1 Hz = 2nd index, ...; CAST to integer
        else:
          tf_as_ind = _cast_as_tensor(hf.tf_to_ind(tfs, stimDur), dtype=torch.long); # if 1s, then TF corresponds to index; if stimDur is 2 seconds, then we can resolve half-integer frequencies -- i.e. 0.5 Hz = 1st index, 1 Hz = 2nd index, ...; CAST to integer
        if len(spectrum[0]) == len(tf_as_ind): # i.e. we have separate tfs for each trial
          spec_curr = spectrum[0];
          rel_amp = [torch.gather(spec_curr, dim=1, index=tf_as_ind)];
          #rel_amp = [torch.stack([spec_curr[i][tf_as_ind[i]] for i in range(len(tf_as_ind))])]; # DEPRECATED --> SLOW!
        else: # i.e. it's just a fixed set of TFs that applies for all trials (expInd=-1)
          rel_amp = [spectrum[i][:, tf_as_ind[i]] for i in range(len(tf_as_ind))];
      except:
        warnings.warn('In spike_fft: if accessing power at particular frequencies, you must also include the stimulation duration!');
        rel_amp = [];
    else:
      rel_amp = [];

    if inclPhase:
      return spectrum, rel_amp, full_fourier, with_phase;
    else:
      return spectrum, rel_amp, full_fourier;

### Datawrapper/loader

def process_data(coreExp, expInd=-1, respMeasure=0, respOverwrite=None, whichTrials=None, shufflePh=False, shuffleTf=False, singleGratsOnly=False, simulate=False):
  ''' Process the trial-by-trial stimulus information for ease of use with the model
      Specifically, we stack the stimuli to be [nTr x nStimComp], where 
      - [:,0] is base, [:,1] is mask, respectively for sfBB
      - If respOverwrite is not None, it will overwrite the responses from coreExp (only used for expInd>=1)
  
      Note: for expInd>=1, pass in X['sfm']['exp']['trial']
  '''
  ### TODO: Make sure you are consistent with returning either rates or counts (I think we should do counts, at least for DC)
  trInf = dict();

  ### first, find out which trials we should be analyzing
  if expInd == -1: # i.e. sfBB
    if simulate:
      trialInf = coreExp;
    else:
      trialInf = coreExp['trial'];
    ######
    # First, get the valid trials (here either mask and/or base present)
    ######
    valTrials = np.where(np.logical_or(trialInf['maskOn'], trialInf['baseOn']))[0]
    if whichTrials is None: # if whichTrials is None, then we're using ALL non-blank trials (i.e. fitting 100% of data)
      whichTrials = valTrials;
    else: # if we did specify whichTrials, then just filter out trials that do not have either mask or base on!
      whichTrials = np.intersect1d(valTrials, whichTrials);
      #### TODO: verify this!
    if not simulate:
      if respMeasure == 0:
        resp = np.expand_dims(coreExp['spikeCounts'][whichTrials], axis=1); # make (nTr, 1)
      elif respMeasure == 1: # then we're getting F1 -- first at baseTF, then maskTF
        # NOTE: CORRECTED TO MASK, then BASE on 20.11.15
        # -- the tranpose turns it from [2, nTr] to [nTr, 2], but keeps [:,0] as mask; [:,1] as base
        resp = np.vstack((coreExp['f1_mask'][whichTrials], coreExp['f1_base'][whichTrials])).transpose();
        # NOTE: Here, if spikeCount == 0 for that trial, then f1_mask/base will be NaN -- replace with zero
        to_repl = np.where(coreExp['spikeCounts'][whichTrials] == 0)[0];
        resp[to_repl, :] = 0;
  # --- first step, but for older experiments    
  elif expInd == 1: # i.e. sfMix original
    trialInf = coreExp;
    ######
    # First, filter out the orientation trials and blank/NaN trials!
    ######
    mask = np.ones_like(coreExp['ori'][0], dtype=bool); # i.e. true
    #mask = np.ones_like(coreExp['num'], dtype=bool); # i.e. true
    #mask = np.ones_like(coreExp['spikeCount'], dtype=bool); # i.e. true
    # and get rid of orientation tuning curve trials
    oriBlockIDs = np.hstack((np.arange(131, 155+1, 2), np.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab
    oriInds = np.empty((0,)); 
    if not simulate: # if we're simulating, no need to check if it's ori condition --> already know that it's not
      for iB in oriBlockIDs:
          indCond = np.where(coreExp['blockID'] == iB);
          if len(indCond[0]) > 0:
              oriInds = np.append(oriInds, indCond);
    mask[oriInds.astype(np.int64)] = False;
    # and also blank/NaN trials
    whereOriNan = np.where(np.isnan(coreExp['ori'][0]))[0];
    mask[whereOriNan] = False;
    if whichTrials is None: # if whichTrials is None, then we're using ALL non-blank trials (i.e. fitting 100% of data)
      whichTrials = np.where(mask)[0];
    else: # then take the specified trials that also meet the mask!
      whichTrials = np.intersect1d(np.where(mask)[0], whichTrials);
      # TODO: verify this!
    # now, finally get ths responses
    if not simulate:
      if respOverwrite is not None:
        resp = respOverwrite;
      else:
        if respMeasure == 0:
          resp = coreExp['spikeCount'].astype(int); # cast as int, since some are as uint
        else:
          resp = coreExp['f1'];
        resp = np.expand_dims(resp, axis=1); # expand dimensions to make it (nTr, 1)
      resp = resp[whichTrials];
  elif expInd >= 2: # the updated sfMix experiments
    trialInf = coreExp;
    ######
    # First, filter the correct/valid trials!
    ######
    # BUT, if we pass in trialSubset, then use this as our mask (i.e. overwrite the above mask)
    if singleGratsOnly:
      # the first line (commented out now) is single gratings AND just at one SF (1.7321, for ex.)
      #valTrials = np.where(np.logical_and(np.isclose(coreExp['sf'][0], 2.460, atol=0.1), np.logical_and(~np.isnan(np.sum(coreExp['ori'], 0)), coreExp['con'][1]==0)))[0]; # this will force singlegrats only!
      # single gratings AND 1.5<=sf<=4
      #valTrials = np.where(np.logical_and(np.logical_and(coreExp['sf'][0]>1.5, coreExp['sf'][0]<4), np.logical_and(~np.isnan(np.sum(coreExp['ori'], 0)), coreExp['con'][1]==0)))[0]; # this will force singlegrats only!
      # DEFAULT: just single gratings
      #valTrials = np.where(np.logical_and(~np.isnan(np.sum(coreExp['ori'], 0)), coreExp['con'][1]==0))[0]; # this will force singlegrats only!
      # 23.01.07 try: only mixtures
      valTrials = np.where(np.logical_and(~np.isnan(np.sum(coreExp['ori'], 0)), coreExp['con'][1]>0))[0]; # this will force singlegrats only!
    else:
      valTrials = np.where(~np.isnan(np.sum(coreExp['ori'], 0)))[0];

    if whichTrials is None: # take all valid trials
      whichTrials = valTrials;
    else: # then we need to find the intersection of valid trials (currently called whichTrials) and 
      whichTrials = np.intersect1d(valTrials, whichTrials);
    if not simulate:
      # TODO -- put in the proper responses... NOTE AS OF 22.11.19 -- I think these responses below are OK, already?
      if respOverwrite is not None:
        resp = respOverwrite;
      else:
        if respMeasure == 0:
          resp = coreExp['spikeCount'].astype(int); # cast as int, since some are as uint
        elif respMeasure == 1:
          resp = coreExp['f1'];
        # 23.02.01 TODO/SUSPICION --> I *think* this line should only apply for DC (i.e. respMeasure==0)
        resp = np.expand_dims(resp, axis=1);
      resp = resp[whichTrials];

  # then, process the raw data such that trInf is [nTr x nComp]
  # for expInd == -1, this means mask [ind=0], then base [ind=1]
  trInf['num'] = whichTrials;
  trInf['ori'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['ori']), (1,0))[whichTrials, :])
  if shuffleTf:
    #arr = np.transpose(np.vstack(trialInf['tf']), (1,0))[whichTrials, :]
    #np.random.shuffle(arr);
    #trInf['tf'] = _cast_as_tensor(arr)
    #trInf['tf'] = _cast_as_tensor(0.1)*_cast_as_tensor(arr)
    trInf['tf'] = _cast_as_tensor(5)*torch.ones_like(_cast_as_tensor(np.transpose(np.vstack(trialInf['tf']), (1,0))[whichTrials, :]));
  else:
    trInf['tf'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['tf']), (1,0))[whichTrials, :])
  if shufflePh:
    arr = np.transpose(np.vstack(trialInf['ph']), (1,0))[whichTrials, :]
    np.random.shuffle(arr);
    trInf['ph'] = _cast_as_tensor(arr)
    #trInf['ph'] = torch.zeros_like(_cast_as_tensor(np.transpose(np.vstack(trialInf['ph']), (1,0))[whichTrials, :]));
  else:
    trInf['ph'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['ph']), (1,0))[whichTrials, :])
  trInf['sf'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['sf']), (1,0))[whichTrials, :])
  trInf['con'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['con']), (1,0))[whichTrials, :])
  #low_isol = np.where(trInf['con'][:,0]<0.06);
  #trInf['con'][low_isol, 0] = trInf['con'][low_isol,0]=0
  if not simulate:
    resp = _cast_as_tensor(resp);
  else: # if simulation, then we haven't processed responses...
    resp = None;

  return trInf, resp;

class dataWrapper(torchdata.Dataset):
    def __init__(self, expInfo, expInd=-1, respMeasure=0, device='cpu', whichTrials=None, respOverwrite=None, shufflePh=False, shuffleTf=False, singleGratsOnly=False, simulate=False):
        # if respMeasure == 0, then we're getting DC; otherwise, F1
        # respOverwrite means overwrite the responses; used only for expInd>=0 for now
        # --- if simulate==True, then we ignore responses...

        super().__init__();
        trInf, resp = process_data(expInfo, expInd, respMeasure, whichTrials=whichTrials, respOverwrite=respOverwrite, shufflePh=shufflePh, shuffleTf=shuffleTf, singleGratsOnly=singleGratsOnly, simulate=simulate)

        self.trInf = trInf;
        if not simulate:
          self.resp = resp;
        self.device = device;
        self.expInd = expInd;
        self.respMeasure = respMeasure
        
    def get_single_item(self, idx):
        # NOTE: This assumes that trInf['ori', 'tf', etc...] are already [nTr, nStimComp]
        feature = dict();
        feature['ori'] = self.trInf['ori'][idx, :]
        feature['tf'] = self.trInf['tf'][idx, :]
        feature['sf'] = self.trInf['sf'][idx, :]
        feature['con'] = self.trInf['con'][idx, :]
        feature['ph'] = self.trInf['ph'][idx, :]
        feature['num'] = idx; # which trials are part of this
        
        target = dict();
        try:
          target['resp'] = self.resp[idx];
          #target['resp'] = self.resp[idx, :];
        except:
          print('resp [rm=%d] has shape:' % self.respMeasure);
          print(self.resp.shape);

        if self.expInd == -1:
          maskInd, baseInd = hf_sfBB.get_mask_base_inds();
          target['maskCon'] = self.trInf['con'][idx, maskInd]
          target['baseCon'] = self.trInf['con'][idx, baseInd]
        else:
          target['cons'] = self.trInf['con'][idx, :];
        
        return (feature, target);

    def __getitem__(self, idx):
        return self.get_single_item(idx)

    def __len__(self):
        return len(self.resp)

### The model
class sfNormMod(torch.nn.Module):
  # inherit methods/fields from torch.nn.Module(); [12,15] and [0.75, 1.5] are defaults!
  def __init__(self, modParams, expInd=-1, excType=2, normType=1, lossType=1, lgnFrontEnd=0, newMethod=1, lgnConType=1, applyLGNtoNorm=1, device='cpu', pSfBound=14.9, pSfBound_low=0.1, fixRespExp=False, normToOne=True, norm_nFilters=[12,15], norm_dOrd=[0.75, 1.5], norm_gain=[0.57, 0.614], norm_range=[0.1,30], useFullNormResp=True, fullDataset=True, toFit=True, normFiltersToOne=False, forceLateNoise=None, _lgnOctDiff=np.log2(9/3), dgNormFunc=False, rvcMod=0):

    super().__init__();

    ### meta/fit parameters
    self.expInd = expInd;
    self.excType = excType
    self.normType = normType;
    self.lossType = lossType;
    self.lgnFrontEnd = lgnFrontEnd;
    self.lgnConType = lgnConType;
    self.applyLGNtoNorm = applyLGNtoNorm; # do we also apply the LGN front-end to the gain control tuning? Default is 1 for backwards compatability, but should be 0 (i.e., DON'T; per Eero, 21.02.26)
    self.device = device;
    self.newMethod = newMethod;
    self.maxPrefSf = _cast_as_tensor(pSfBound); # don't allow the max prefSf to exceed this value (will enforce via sigmoid)
    self.minPrefSf = _cast_as_tensor(pSfBound_low); # don't allow the prefSf to go below this value (will avoid div by 0)
    self.normToOne = normToOne;
    self.normFiltersToOne = normFiltersToOne; # normalize the quadrature filters?
    self.fullDataset = fullDataset; # are we fitting the full dataset at once?
    self.dgNormFunc = dgNormFunc

    ### all modparams
    self.modParams = modParams;

    ### Keep an empty space for calculations which are independent of model (i.e. stimulus only) --> that way we don't re-compute
    # ---- todo: make sure this doesn't hurt memory too much? 22.10.13
    self.stimRealImag = None; # defaults to None so that we know to compute it the first time around
    self.lgnCalcDone  = False;

    ### now, establish the parameters to optimize
    # Get parameter values
    nParams = hf.nParamsByType(self.normType, excType, lgnFrontEnd);

    # handle the possibility of a multi fitting first
    if self.lgnFrontEnd == 99:
      self.mWeight = _cast_as_param(modParams[nParams-1]);
    elif self.lgnFrontEnd == 1 and self.lgnConType == 1:
      self.mWeight = _cast_as_param(modParams[-1]);
    elif (self.lgnFrontEnd == 3 or self.lgnFrontEnd == 4) and self.lgnConType == 1: # yoked LGN; mWeight is still treated the same way
      self.mWeight = _cast_as_param(modParams[-1]);
    elif self.lgnConType == 2 or self.lgnConType == 5: # FORCE at 0.5
      self.mWeight = _cast_as_tensor(0); # then it's NOT a parameter, and is fixed at 0, since as input to sigmoid, this gives 0.5
    elif self.lgnConType == 3: # still not a parameter, but use the value in modParams; this case shouldn't really come up...
      self.mWeight = _cast_as_tensor(modParams[-1]);
    elif self.lgnConType == 4: # FORCE at -1, i.e. all parvo
      self.mWeight = _cast_as_tensor(-np.Inf); # then it's NOT a parameter, and is fixed at -Inf, since as input to sigmoid, this gives 0 (i.e. all parvo)
    # make sure mWeight is 0 if LGN is not on (this will also ensure it's defined)
    if lgnFrontEnd<=0: # i.e. no LGN, then overwrite this!
      self.mWeight = _cast_as_tensor(-np.Inf); # not used anyway!
    # LGN front end - separate center SF for LGN?
    if self.lgnFrontEnd == 4:
      self.lgnCtrSf    = _cast_as_param(modParams[7]);  # multiplicative noise
    else: # ignored in these cases, anyway
      self.lgnCtrSf    = _cast_as_tensor(modParams[7]);  # NOT optimized and fully ignored, as of 22.12.26
      
    self.prefSf = _cast_as_param(modParams[0]);
    if self.excType == 1:
      self.dordSp = _cast_as_param(modParams[1]);
    if self.excType == 2:
      self.sigLow = _cast_as_param(modParams[1]);
      highInd = -1-np.sign(lgnFrontEnd);
      self.sigHigh = _cast_as_param(modParams[highInd]);

    # Other (nonlinear) model components
    self.sigma    = _cast_as_param(modParams[2]); # normalization constant
    #self.respExp  = _cast_as_tensor(1); # response exponent -- fixed at one (not a free param)
    self.respExp  = _cast_as_param(modParams[3]) if fixRespExp is None else _cast_as_tensor(modParams[3]); # response exponent
    self.scale    = _cast_as_param(modParams[4]); # response scalar

    # Noise parameters
    if force_earlyNoise is None or not toFit:
      self.noiseEarly = _cast_as_param(modParams[5]);   # early additive noise
    else:
      self.noiseEarly = _cast_as_tensor(force_earlyNoise); # early additive noise - fixed value, i.e. NOT optimized in this case
    if forceLateNoise is None or not toFit:
      self.noiseLate  = _cast_as_param(modParams[6]);  # late additive noise
    else:
      self.noiseLate  = _cast_as_param(forceLateNoise); # could be set to 0 (eg. for F1)

    ### Normalization parameters
    normParams = hf.getNormParams(modParams, normType, forceAsymZero=toFit);
    if self.normType == 1 or self.normType == 0:
      self.inhAsym = _cast_as_tensor(normParams) if self.normType==1 else _cast_as_param(normParams); # then it's not really meant to be optimized, should be just zero
      self.gs_mean = None; self.gs_std = None; # replacing the "else" in commented out 'if normType == 2 or normType == 4' below
    elif self.normType == 2 or self.normType == 5 or self.normType == 6 or self.normType == 7:
      if self.normType == 6: # just cast as tensor, since we will NOT optimize for it [yoked to filter prefSf]
        self.gs_mean = _cast_as_tensor(torch.log(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf)));
      else:
        self.gs_mean = _cast_as_param(normParams[0]);
      self.gs_std  = _cast_as_param(normParams[1]);
      if self.normType == 5:
        self.gs_gain = _cast_as_param(normParams[2]);
      else:
        self.gs_gain = None;
      # overwrite the above if:
      if self.normType == 7 and self.excType == 1:
        self.gs_mean = _cast_as_tensor(self.prefSf);
        self.gs_std = _cast_as_tensor(self.dordSp);
    elif self.normType == 3:
      # sigma calculation
      self.offset_sigma = _cast_as_param(normParams[0]);  # c50 filter will range between [v_sigOffset, 1]
      self.stdLeft      = _cast_as_param(normParams[1]);  # std of the gaussian to the left of the peak
      self.stdRight     = _cast_as_param(normParams[2]); # '' to the right '' 
      self.sfPeak       = _cast_as_param(normParams[3]); # where is the gaussian peak?
    elif self.normType == 4: # NOT DEPRECATED, but not really used...
      self.gs_mean = _cast_as_param(normParams[0]); # mean
      self.gs_std_low = _cast_as_param(normParams[1][0]); # really the deriv. order!
      self.gs_std_high = _cast_as_param(normParams[1][1]); # really the deriv. order!
    else:
      self.inhAsym = _cast_as_param(normParams);

    # also initialize the full normalizaton model/pool!
    self.useFullNormResp = useFullNormResp; # are we computing full normalization?
    self.normFull = dict({'nFilters': norm_nFilters, 'dOrd': _cast_as_tensor(norm_dOrd), 'norm_gain': _cast_as_tensor(norm_gain)})
    # and make the prefSfs
    psfs = None;
    for i,nFilt in enumerate(norm_nFilters):
      psfs_curr = _cast_as_tensor(np.geomspace(norm_range[0], norm_range[1], nFilt));
      if psfs is None:
        psfs = [psfs_curr];
      else:
        psfs.append(psfs_curr);
    self.normFull['prefSfs'] = psfs;
    self.normFilters = None; # have we computed the filters?
    self.normFiltersBasic = None;
    self.normCalc = None; # have we made the full calculation (w/stimulus?)

    ### LGN front parameters
    self.LGNmodel = 2; # parameterized as DiffOfGauss (not DoGsach)
    # prepopulate DoG parameters -- will overwrite, if needed
    # STANDARDS, CURRENT AS OF 23.01.29 --> [1,3,0.3,0.4] for M /// [1,9,0.5,0.4] for P
    self.M_k = _cast_as_tensor(1); # gain of 1
    self.M_fc = _cast_as_tensor(3); # char. freq of 3
    self.M_ks = _cast_as_tensor(0.3); # surround gain rel. to center
    self.M_js = _cast_as_tensor(0.4); # relative char. frequency of surround
    self.P_k = _cast_as_tensor(1); # gain of 1 [we handle M vs. P sensitivity in the RVC]
    self.P_fc = _cast_as_tensor(9); # char. freq of 9 [3x is a big discrepancy]
    self.P_ks = _cast_as_tensor(0.5); # surround gain rel. to center
    self.P_js = _cast_as_tensor(0.4); # relative char. freq of surround
    # --- below are more exaggerated M&P parameters --> namely much stronger surround
    # - attempts on 23.01.30, however, suggest that this just allows the model to use LGN as a tuned gain control (more bandpass than the above)
    #self.M_ks = _cast_as_tensor(0.7); # surround gain rel. to center
    #self.M_js = _cast_as_tensor(0.3); # relative char. frequency of surround
    #self.P_ks = _cast_as_tensor(0.55); # surround gain rel. to center
    #self.P_js = _cast_as_tensor(0.3); # relative char. freq of surround

    if self.lgnFrontEnd == 2:
      self.M_fc = _cast_as_tensor(6); # different variant (make magno f_c=6, not 3)
    elif self.lgnFrontEnd == 3 or self.lgnFrontEnd == 4:
      self.lgnOctDiff = _cast_as_tensor(_lgnOctDiff);
      if self.lgnFrontEnd == 3:
        ctrSf = self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf).item();
      elif self.lgnFrontEnd == 4:
        ctrSf = self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.lgnCtrSf).item();
      # With the preferred SF, we'll make a somewhat convoluted calculation:
      # - 1. We can analatically determine the relationship between f_c and psf
      # ---- Note that this relationship depends on surround strength and gain for the DoG model
      # - 2. Using, the pre-described relationship between M and P char. freq (3x, or np.log2(3) in log2)...
      # ---- ...get the equivalent M and P pSf
      # ---- Note that based on the note in 1., we use a different calculation for M and P
      # - THUS: Here, we compute the mapping for pSf to f_c, which is fixed based on surround gain/radius
      sfs_test = np.geomspace(1e-10, 100, 100); # over what range of Sfs do we allow for a preferred SF; since it's model, say wide
      f_cs = np.geomspace(0.5, 15, 100);
      p_psfs = np.array([hf.descr_prefSf([self.P_k.item(), fc_curr, self.P_ks.item(), self.P_js.item()], self.LGNmodel, all_sfs=sfs_test) for fc_curr in f_cs]);
      m_psfs = np.array([hf.descr_prefSf([self.M_k.item(), fc_curr, self.M_ks.item(), self.M_js.item()], self.LGNmodel, all_sfs=sfs_test) for fc_curr in f_cs]);
      m_mapping = ss.linregress(f_cs, m_psfs);
      p_mapping = ss.linregress(f_cs, p_psfs);
      # now, with the slope/intercept from mapping, make lambda to get the equiv. psf from the f_c
      get_eqv_psf_m = lambda fc: m_mapping.intercept + fc*m_mapping.slope
      get_eqv_psf_p = lambda fc: p_mapping.intercept + fc*p_mapping.slope
      # --- plus the mapping from pSf to desired fc
      self.get_eqv_cf_m = lambda psf: (psf - m_mapping.intercept)/m_mapping.slope
      self.get_eqv_cf_p = lambda psf: (psf - p_mapping.intercept)/p_mapping.slope
      # --- the below (oct_range_in_psf) gives the equivalent range in PSF for lgnOctDiff-fold fc difference
      # ------ why? P_fc is np.power(2, lgnOctDiff) while M_fc is 1, thus P_fc is 2^lgnOctDiff times M_fc
      m_fc_at_psf = self.get_eqv_cf_m(ctrSf);
      p_fc_at_psf = self.get_eqv_cf_m(ctrSf);
      log_mn_fc_at_psf = m_fc_at_psf * np.power(2, np.log2(p_fc_at_psf/m_fc_at_psf));
      self.oct_range_in_psf = np.log2(get_eqv_psf_p(log_mn_fc_at_psf*np.power(2, self.lgnOctDiff/2))/get_eqv_psf_m(log_mn_fc_at_psf*np.power(2, -self.lgnOctDiff/2)))
      # Now, get the desired pSf/mPsf, and go back to charFreq
      p_sf, m_sf = [ctrSf * np.power(2, self.oct_range_in_psf/2), ctrSf * np.power(2, -self.oct_range_in_psf/2)]
      # now, back out the f_c from the expected psf
      self.M_fc = self.get_eqv_cf_m(m_sf)
      self.P_fc = self.get_eqv_cf_p(p_sf)
    elif self.lgnFrontEnd == 99: # 99 is code for fitting an LGN front end which is common across all cells in the dataset...
      # parameters are passed as [..., m_fc, p_fc, m_ks, p_ks, m_js, p_js]
      self.M_fc = _cast_as_param(self.modParams[-6]);
      self.M_ks = _cast_as_param(self.modParams[-4]);
      self.M_js = _cast_as_param(self.modParams[-2]);
      ## TODO: Will need to update P_fc in the code rather than rely on this one-time writing of self.P_fc (since self.M_fc will update)
      self.P_fc = torch.mul(self.M_fc, _cast_as_param(self.modParams[-5]));
      self.P_ks = _cast_as_param(self.modParams[-3]);
      self.P_js = _cast_as_param(self.modParams[-1]);
    # specify rvc parameters (not true "parameters", i.e. not optimized)
    if self.lgnFrontEnd > 0:
      self.rvcMod = rvcMod;
      if self.rvcMod==1: # Naka-Rushton (base, gain, expon, c50)
        self.rvc_m = _cast_as_tensor([0, 1, 1.5, 0.15]); # magno has lower c50
        self.rvc_p = _cast_as_tensor([0, 1, 1.5, 0.55]);
      else: # Tony formulation of RVC
        self.rvc_m = _cast_as_tensor([0, 12.5, 0.05]); # magno has lower gain, c50
        self.rvc_p = _cast_as_tensor([0, 17.5, 0.50]);
      # --- and pack the DoG parameters we specified above
      self.dog_m = [self.M_k, self.M_fc, self.M_ks, self.M_js] 
      self.dog_p = [self.P_k, self.P_fc, self.P_ks, self.P_js]
    ### END OF INIT

  #######
  def update_manual(self, verbose=False):
    # update fields (NOT PARAMETERS) which are dep. on parameters
    # --- two of the possible updates (as of 22.12.20) rely on prefSf
    prefSf = self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf);

    if self.lgnFrontEnd == 3 or self.lgnFrontEnd == 4: # Then we have to update the M/P_fc
      if self.lgnFrontEnd == 4: # then actually LGN filters placed on their own
        ctrSf = self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.lgnCtrSf);
      else:
        ctrSf = prefSf;

      # Now, get the desired pSf/mPsf, and go back to charFreq
      p_sf, m_sf = [ctrSf * np.power(2, self.oct_range_in_psf/2), ctrSf * np.power(2, -self.oct_range_in_psf/2)]
      # now, back out the f_c from the expected psf
      self.M_fc = self.get_eqv_cf_m(m_sf)
      self.P_fc = self.get_eqv_cf_p(p_sf)
      # --- and pack the DoG parameters we specified above
      self.dog_m = [self.M_k, self.M_fc, self.M_ks, self.M_js] 
      self.dog_p = [self.P_k, self.P_fc, self.P_ks, self.P_js]
      # --- finally, set lgnCalcDone to False (we'll need to recalculate)
      self.lgnCalcDone = False;
      if verbose:
        print('M/P fc are %.2f/%.2f [done? %r]' % (self.M_fc, self.P_fc, self.lgnCalcDone))
    if self.normType == 6: # norm pool mn eq. to prefSf
      self.gs_mean = prefSf
    if self.normType == 7 and self.excType == 1: # only works if norm weights are given as deriv. of Gaussian AND dG exc filter
      # in this case, same filter shape!
      self.gs_mean = self.prefSf;
      self.gs_std = self.dordSp

  def transform_sigmoid_param(self, whichPrm, _sigmoidDord=_sigmoidDord, _sigmoidSigma=_sigmoidSigma, _sigmoidScale=_sigmoidScale, overwriteValue=None):
    # used for outputs (e.g. prints, plots) where we don't want the sigmoided value...
    if whichPrm == 'prefSf' or whichPrm=='lgnCtrSf':
      if overwriteValue is not None:
        val_to_use = _cast_as_tensor(overwriteValue);
      else:
        val_to_use = self.prefSf if whichPrm=='prefSf' else 'lgnCtrSf';
      curr_val = self.minPrefSf + self.maxPrefSf*torch.sigmoid(val_to_use)
    elif whichPrm == 'dordSp':
      val_to_use = self.dordSp if overwriteValue is None else _cast_as_tensor(overwriteValue);
      curr_val = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(val_to_use));
    elif whichPrm == 'gs_mean':
      val_to_use = self.gs_mean if overwriteValue is None else _cast_as_tensor(overwriteValue);
      if self.dgNormFunc:
        curr_val = self.minPrefSf + self.maxPrefSf*torch.sigmoid(val_to_use)
      else:
        curr_val = val_to_use
    elif whichPrm == 'gs_std':
      val_to_use = self.gs_std if overwriteValue is None else _cast_as_tensor(overwriteValue);
      if self.dgNormFunc:
        curr_val = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(val_to_use))
      else:
        curr_val = val_to_use;
    elif whichPrm == 'mWt':
      val_to_use = self.mWeight if overwriteValue is None else _cast_as_tensor(overwriteValue);
      curr_val = torch.sigmoid(val_to_use);
    return curr_val.detach().numpy();
      
  def clear_saved_calcs(self):
    # reset the images/pre-computed calculations in case the dataset has changed (i.e. cross-validation!)
    self.stimRealImag = None; # defaults to None so that we know to compute it the first time around
    self.lgnCalcDone  = False;
    self.normFilters  = None;
    self.normFiltersBasic  = None;
    self.normCalc     = None;

  def print_params(self, transformed=1):
    # return a list of the parameters
    print('\n********MODEL PARAMETERS********');
    print('prefSf: %.2f' % (self.minPrefSf.item() + self.maxPrefSf.item()*torch.sigmoid(self.prefSf).item())); # was just self.prefSf.item()
    # only if lgnFrontEnd==4
    if self.lgnFrontEnd==4:
      print('\tLGN filters centered around: %.2f' % (self.minPrefSf.item() + self.maxPrefSf.item()*torch.sigmoid(self.lgnCtrSf).item()));
    if self.excType == 1:
      dord = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(self.dordSp)) if transformed else self.dordSp.item();
      print('deriv. order: %.2f' % dord);
    elif self.excType == 2:
      print('sigma l|r: %.2f|%.2f' % (self.sigLow.item(), self.sigHigh.item()));
    if self.lgnFrontEnd > 0:
      mWt = torch.sigmoid(self.mWeight).item() if transformed else self.mWeight.item();
      print('mWeight: %.2f (orig %.2f)' % (mWt, self.mWeight.item()));
      print('\tapplying the LGN filter for the gain control' if self.applyLGNtoNorm else 'No LGN for GC')
    else:
      print('No LGN!');
    scale = torch.mul(_cast_as_tensor(_sigmoidScale), torch.sigmoid(self.scale)).item() if transformed else self.scale.item();
    if self.normToOne==1 and self.newMethod==1: # then scale is simpler!
      scale = self.scale.item();
    print('scalar|early|late: %.3f|%.3f|%.3f' % (scale, self.noiseEarly.item(), self.noiseLate.item()));
    print('norm. const.: %.2f' % self.sigma.item());
    if self.normType == 2 or self.normType == 5 or self.normType == 6:
      normMn = torch.exp(self.gs_mean).item() if transformed else self.gs_mean.item();
      print('tuned norm mn|std: %.2f|%.2f' % (normMn, torch.abs(self.gs_std).item()));
      if self.normType == 5:
        print('\tAnd the norm gain (transformed|untransformed) is: %.2f|%.2f' % (torch.mul(_cast_as_tensor(_sigmoidGainNorm), torch.sigmoid(self.gs_gain)).item(), self.gs_gain.item()));
    elif self.normType == 0:
      print('inhAsym: %.2f' % self.inhAsym.item());
    elif self.normType == 7 and self.excType == 1:
      print('norm. tuning matched to exc. filter!')
    print('********END OF MODEL PARAMETERS********\n');

    return None;

  def return_params(self):
    # return a list of the parameters
    if self.normType <= 1:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.inhAsym.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.inhAsym.item(), self.sigHigh.item(), self.mWeight.item()];
    elif self.normType == 2 or self.normType == 6 or self.normType == 7:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.gs_mean.item(), self.gs_std.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.gs_mean.item(), self.gs_std.item(), self.sigHigh.item(), self.mWeight.item()];
    elif self.normType == 5:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.gs_mean.item(), self.gs_std.item(), self.gs_gain.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.gs_mean.item(), self.gs_std.item(), self.gs_gain.item(), self.sigHigh.item(), self.mWeight.item()];
    # NOT really used, but keeping for posterity (in case we try to revive this flex. Gauss normalization weighting)
    elif self.normType == 4:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.gs_mean.item(), self.gs_std_low.item(), self.gs_std_high.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.lgnCtrSf.item(), self.gs_mean.item(), self.gs_std_low.item(), self.gs_std_high.item(), self.sigHigh.item(), self.mWeight.item()];
    # after all possible model configs...
    if self.lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
      param_list = param_list[0:-1];

    return param_list

  def simpleResp_matMul(self, trialInf, stimParams = [], sigmoidSigma=_sigmoidSigma, preCompOri=None, debug=False, fps=120, quadrature=False):
    # returns object with simpleResp and other things
    # --- Created 20.10.12 --- provides ~4x speed up compared to SFMSimpleResp() without need to explicit parallelization
    # --- Updated 20.10.29 --- created new method 

    # simpleResp_matMul       Computes response of simple cell for sfmix experiment

    # simpleResp_matMul(varargin) returns a complex cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # NOTE: For first pass, not building in make_own_stim, since this will be for
    # - optimizing the model, not for evaluating at arbitrary stimuli

    # Get spatial coordinates
    xCo = 0; # in visual degrees, centered on stimulus center
    yCo = 0; # in visual degrees, centered on stimulus center

    # Store some results in M

    # Pre-allocate memory
    z             = trialInf;
    nStimComp     = hf.get_exp_params(self.expInd).nStimComp;
    nFrames       = hf.num_frames(self.expInd);
    try:
      nTrials = len(z['num']);
    except:
      nTrials = len(z['con'][0]); 
    
    ####
    # Set stim parameters
    stimTf = _cast_as_tensor(z['tf'], self.device);
    stimCo = _cast_as_tensor(z['con'], self.device);
    stimSf = _cast_as_tensor(z['sf'], self.device);
    stimOr = _cast_as_tensor((np.pi/180) * z['ori'], self.device);
    stimPh = _cast_as_tensor((np.pi/180) * z['ph'], self.device);

    ### LGN filtering stage
    ### Assumptions: No interaction between SF/con -- which we know is not true...
    # - first, SF tuning: model 2 (Tony)
    if self.lgnFrontEnd > 0:
      if self.lgnCalcDone and self.fullDataset: # only skip if already done AND full dataset
        selCon_p = self.selCon_p;
        selCon_m = self.selCon_m;
        selSf_p = self.selSf_p;
        selSf_m = self.selSf_m;
      else:
        resps_m = get_descrResp(self.dog_m, stimSf, self.LGNmodel, minThresh=globalMin)
        resps_p = get_descrResp(self.dog_p, stimSf, self.LGNmodel, minThresh=globalMin)
        # -- make sure we normalize by the true max response:
        sfTest = _cast_as_tensor(np.geomspace(0.1, 15, 1000));
        max_m = torch.max(get_descrResp(self.dog_m, sfTest, self.LGNmodel, minThresh=globalMin));
        max_p = torch.max(get_descrResp(self.dog_p, sfTest, self.LGNmodel, minThresh=globalMin));
        # -- then here's our selectivity per component for the current stimulus
        selSf_m = torch.div(resps_m, max_m);
        selSf_p = torch.div(resps_p, max_p);
        # - then RVC response: # ASSUMES rvcMod 0 (Movshon)
        # --- the following commented out lines show how we could evaluate the rvc at one contrast for all components -->
        # -----> HOWEVER, this is how we factor in contrast, so likely this will be useful only for eval. SF at diff. con, if we introduce LGN shifts [22.10.03]
        #scm = torch.max(stimCo, axis=1)[0];
        #selCon_m = get_rvc_model(self.rvc_m, scm).unsqueeze(dim=1);
        #selCon_p = get_rvc_model(self.rvc_p, scm).unsqueeze(dim=1);
        selCon_m = get_rvc_model(self.rvc_m, stimCo, self.rvcMod); # could evaluate at torch.max(stimCo,axis=1)[0] rather than stimCo, i.e. highest grating con, not per grating
        selCon_p = get_rvc_model(self.rvc_p, stimCo, self.rvcMod);

        # and save the values/flag that we 've done it
        self.selSf_p = selSf_p;
        self.selSf_m = selSf_m;
        self.selCon_p = selCon_p;
        self.selCon_m = selCon_m;
        self.lgnCalcDone = True; # save the LGN calc --> but fear not, we'll overwrite if not full dataset
        # NOTE: We will turn lgnCalcDone back to False when we update the m/p params (i.e. lgnFrontEnd==3 or 4)

      if self.lgnConType == 1 or self.lgnConType == 5: # DEFAULT
        # -- then here's our final responses per component for the current stimulus
        # ---- NOTE: The real mWeight will be sigmoid(mWeight), such that it's bounded between 0 and 1
        lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_m)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_p)));
      elif self.lgnConType == 2 or self.lgnConType == 3 or self.lgnConType == 4:
        # -- Unlike the above (default) case, we don't allow for a separate M & P RVC - instead we just take the average of the two
        avgWt = torch.sigmoid(self.mWeight);
        selCon_avg = avgWt*selCon_m + (1-avgWt)*selCon_p;
        lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_avg)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_avg)));

    if self.excType == 1:
      # Compute spatial frequency tuning - Deriv. order Gaussian
      sfRel = torch.div(stimSf, self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf));
      effDord = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(self.dordSp));
      s     = torch.pow(stimSf, effDord) * torch.exp(-effDord/2 * torch.pow(sfRel, 2));
      sMax  = torch.pow(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf), effDord) * torch.exp(-effDord/2);
      selSf   = torch.div(s, sMax);
    elif self.excType == 2:
      selSf = flexible_Gauss([0,1,self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf),self.sigLow,self.sigHigh], stimSf, minThresh=0, sigmoidValue=sigmoidSigma);
 
    if self.lgnFrontEnd > 0:
      selSi = torch.mul(selSf, lgnSel); # filter sensitivity for the sinusoid in the frequency domain
    else:
      selSi = selSf;

    # II. Phase, space and time
    if self.stimRealImag is None or not self.fullDataset: # i.e. if it's not full dataset, then we need to overwrite the dataset
      if preCompOri is None:
        omegaX = torch.mul(stimSf, torch.cos(stimOr)); # the stimulus in frequency space
        omegaY = torch.mul(stimSf, torch.sin(stimOr));
      else: # preCompOri is the same for all trials/comps --> cos(stimOr) is [0], sin(-) is [1]
        omegaX = torch.mul(stimSf, preCompOri[0]); # the stimulus in frequency space
        omegaY = torch.mul(stimSf, preCompOri[1]);
      #omegaT = 5*stimTf/stimTf; # make them all 1??? # USED FOR DEBUGGING ODDITIES
      omegaT = stimTf;

      P = torch.empty((nTrials, nFrames, nStimComp, 3)); # nTrials x nFrames for number of frames x nStimComp x [two for x and y coordinate, one for time]
      P[:,:,:,0] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*xCo);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
      P[:,:,:,1] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*yCo);  # P(:,0) and p(:,1) describe location of the filters in space

      # Use the effective number of frames displayed/stimulus duration
      # phase calculation -- 
      stimFr = torch.div(torch.arange(nFrames), float(fps));
      #stimFr = torch.div(torch.arange(nFrames), float(nFrames));
      phOffset = torch.div(stimPh, torch.mul(2*np.pi, stimTf));
      #phOffset = torch.div(100*torch.ones_like(stimPh), 2*np.pi); # USED FOR DEBUGGING ODDITIES
      # fast way?
      P3Temp = torch.add(phOffset.unsqueeze(-1), stimFr.unsqueeze(0).unsqueeze(0)).permute(0,2,1); # result is [nTrials x nFrames x nStimComp], so transpose
      P[:,:,:,2]  = 2*np.pi*P3Temp; # P(:,2) describes relative location of the filters in time.

      # per LCV code: preallocation and then filling in is much more efficient than using stack
      omegas = torch.empty((*omegaX.shape,3), device=omegaX.device);
      omegas[..., 0] = omegaX;
      omegas[..., 1] = omegaY;
      omegas[..., 2] = omegaT;
      dotprod = torch.einsum('ijkl,ikl->ijk',P,omegas); # dotproduct over the "3" to get [nTr x nSC x nFr]

      # as of 20.10.14, torch doesn't handle complex numbers
      # since we're just computing e^(iX), we can simply code the real (cos(x)) and imag (sin(x)) parts separately
      # -- by virtue of e^(iX) = cos(x) + i*sin(x) // Euler's identity
      realPart = torch.cos(dotprod);
      imagPart = torch.sin(dotprod);
      # per LCV code: preallocation and then filling in is much more efficient than using stack
      realImag = torch.empty((*realPart.shape,2), device=realPart.device);
      realImag[...,0] = realPart;
      realImag[...,1] = imagPart;
      self.stimRealImag = realImag; # write the current image REGARDLESS of 
    else:
      realImag = self.stimRealImag;

    # NOTE: here, I use the term complex to denote that it is a complex number NOT
    # - that it reflects a complex cell (i.e. this is still a simple cell response)
    if self.lgnFrontEnd > 0: # contrast already included: TRY 22.10.12
      # selSi is [nTr x nComp], realImag is [nTr x nComp x nFrames x 2]
      rComplex = torch.einsum('ij,ikjz->ikz', selSi, realImag) # mult. to get [nTr x nFr x 2] response
    else: # need to include contrast here
      rComplex = torch.einsum('ij,ikjz->ikz', torch.mul(selSi,stimCo), realImag) # mult. to get [nTr x nFr x 2] response
    # The above line takes care of summing over stimulus components
    if self.normToOne == 1 and self.newMethod == 1:
      # since in new method, we just return [...,0], normalize the response to the max of that component
      # --- this ensures that the amplitude is the same regardless of whether LGN is ON or OFF
      # --- added 22.10.10
      if self.normFiltersToOne:
        rComplex = torch.div(rComplex, torch.max(_cast_as_tensor(globalMinDiv), torch.max(rComplex[...,0]))); # max = 1
        if quadrature:
          # used for DC (i.e. if respMeasure==0)
          rComplexA = torch.div(rComplex[...,1], torch.max(_cast_as_tensor(globalMinDiv), torch.max(rComplex[...,1]))); # max = 1
          rComplexB = torch.div(_cast_as_tensor(-1)*rComplex[...,0], torch.max(_cast_as_tensor(globalMinDiv), torch.max(_cast_as_tensor(-1)*rComplex[...,0]))); # max = 1
          rComplexC = torch.div(_cast_as_tensor(-1)*rComplex[...,1], torch.max(_cast_as_tensor(globalMinDiv), torch.max(_cast_as_tensor(-1)*rComplex[...,1]))); # max = 1
      else: # no intermediate norm, just keep the filters as is
        rComplex = rComplex;
        if quadrature:
          # used for DC (i.e. if respMeasure==0)
          rComplexA = rComplex[...,1];
          rComplexB = _cast_as_tensor(-1)*rComplex[...,0];
          rComplexC = _cast_as_tensor(-1)*rComplex[...,1];


    if debug: # TEMPORARY?
      return realImag,selSi,torch.mul(selSi,stimCo);

    # Store response in desired format - which is actually [nFr x nTr], so transpose it!
    if self.newMethod == 1:
      # NOTE: 22.10.24 -- SHOULD ADD noiseEarly HERE before/at rectification, NOT afterward
      if quadrature:
        # i.e. complex cell
        respSimple1 = torch.max(_cast_as_tensor(globalMin), self.noiseEarly + rComplex[...,0]); # half-wave rectification,...
        respSimple2 = torch.max(_cast_as_tensor(globalMin), self.noiseEarly + rComplexB);
        respSimple3 = torch.max(_cast_as_tensor(globalMin), self.noiseEarly + rComplexA);
        respSimple4 = torch.max(_cast_as_tensor(globalMin), self.noiseEarly + rComplexC);

        rsall = torch.div(torch.pow(respSimple1, self.respExp) + torch.pow(respSimple2, self.respExp) + torch.pow(respSimple3, self.respExp) + torch.pow(respSimple4, self.respExp), 4);
        #rsall = torch.sqrt(torch.div(torch.pow(respSimple1, 2) + torch.pow(respSimple2, 2) + torch.pow(respSimple3, 2) + torch.pow(respSimple4, 2), 4));
        return torch.transpose(rsall,0,1);
        # the below line is useful for debugging purposes
        #return torch.transpose(respSimple1, 0, 1), torch.transpose(rComplexA,0,1), torch.transpose(rComplexB,0,1), torch.transpose(rComplexC,0,1);
      else:
        respSimple1 = torch.pow(torch.max(_cast_as_tensor(globalMin), rComplex[...,0]), self.respExp);
        #respSimple1 = rComplex[...,0]; # we'll keep the half-wave rectification for the end...
        return torch.transpose(respSimple1, 0, 1)

    else: # Old approach, in which we return the complex response
      # four filters placed in quadrature (only if self.newMethod == 0, which is default)
      respSimple1 = torch.max(_cast_as_tensor(globalMin), rComplex[...,0]); # half-wave rectification,...
      respSimple2 = torch.max(_cast_as_tensor(globalMin), torch.mul(_cast_as_tensor(-1),rComplex[...,0]));
      respSimple3 = torch.max(_cast_as_tensor(globalMin), rComplex[...,1]);
      respSimple4 = torch.max(_cast_as_tensor(globalMin), torch.mul(_cast_as_tensor(-1),rComplex[...,1]));

      # if channel is tuned, it is phase selective...
      # NOTE: 19.05.14 - made response always complex...(wow)! See git for previous version
      respComplex = torch.pow(respSimple1, 2) + torch.pow(respSimple2, 2) \
          + torch.pow(respSimple3, 2) + torch.pow(respSimple4, 2);
      respAvg = torch.div(respComplex, 4);
      respComp = torch.sqrt(respAvg); # div by 4 to avg across all filters

      return torch.transpose(respComp, 0, 1);

  #def genNormWeightsSimple(self, trialInf, recenter_norm=recenter_norm, threshWeights=1e-6, avg_sfs=_cast_as_tensor(np.geomspace(0.1,30,51))):
  def genNormWeightsSimple(self, trialInf, recenter_norm=recenter_norm, threshWeights=1e-6, avg_sfs=None, gs_std_min=0.3):
    ''' simply evaluates the usual normalization weighting but at the frequencies of the stimuli directly
    i.e. in effect, we are eliminating the bank of filters in the norm. pool
        --- if threshWeights is None, then we won't threshold the norm. weights; 
        --- if it's not None, we do max(threshWeights, calculatedWeights)
        --- gs_std_min=0.3 gives ~1 octave for norm. filter at a minimum
    '''

    sfs = _cast_as_tensor(trialInf['sf']); # [nComps x nTrials]
    cons = _cast_as_tensor(trialInf['con']); # [nComps x nTrials]

    # apply LGN stage -
    # -- NOTE: previously, we applied equal M and P weight, since this is across a population of neurons, not just the one one neuron under consideration
    if self.lgnFrontEnd > 0 and self.applyLGNtoNorm:
      if self.lgnCalcDone:
        selCon_p = self.selCon_p;
        selCon_m = self.selCon_m;
        selSf_p = self.selSf_p;
        selSf_m = self.selSf_m;
      else:
        resps_m = get_descrResp(self.dog_m, sfs, self.LGNmodel, minThresh=globalMin)
        resps_p = get_descrResp(self.dog_p, sfs, self.LGNmodel, minThresh=globalMin)
        # -- make sure we normalize by the true max response:
        sfTest = _cast_as_tensor(np.geomspace(0.1, 15, 1000));
        max_m = torch.max(get_descrResp(self.dog_m, sfTest, self.LGNmodel, minThresh=globalMin));
        max_p = torch.max(get_descrResp(self.dog_p, sfTest, self.LGNmodel, minThresh=globalMin));
        # -- then here's our selectivity per component for the current stimulus
        selSf_m = torch.div(resps_m, max_m);
        selSf_p = torch.div(resps_p, max_p);
        # - then RVC response: # ASSUMES rvcMod 0 (Movshon)
        selCon_m = get_rvc_model(self.rvc_m, cons, self.rvcMod);
        selCon_p = get_rvc_model(self.rvc_p, cons, self.rvcMod);
      # -- then here's our final responses per component for the current stimulus
      if self.lgnConType == 1 or self.lgnConType == 5: # DEFAULT
        # -- then here's our final responses per component for the current stimulus
        # ---- NOTE: The real mWeight will be sigmoid(mWeight), such that it's bounded between 0 and 1
        lgnStage = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_m)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_p)));
      elif self.lgnConType == 2 or self.lgnConType == 3 or self.lgnConType == 4:
        # -- Unlike the above (default) case, we don't allow for a separate M & P RVC - instead we just take some average in-between of the two (depending on lgnConType)
        selCon_avg = torch.sigmoid(self.mWeight)*selCon_m + (1-torch.sigmoid(self.mWeight))*selCon_p;
        lgnStage = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_avg)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_avg)));
      #if avg_sfs is not None:
      #  # also compute the LGN weights across
      #  lgnStageForAvg = 
    else:
      lgnStage = torch.ones_like(sfs);
    lgnStage = torch.div(lgnStage, torch.mean(lgnStage)); # make average=1 for LGN stage
      
    if self.gs_mean is None or self.gs_std is None: # we assume inhAsym is 0
      #self.inhAsym = _cast_as_tensor(0);
      new_weights = 1 + self.inhAsym*(torch.log(sfs) - torch.mean(torch.log(sfs)));
      new_weights = torch.mul(lgnStage, new_weights);
      # AS of 22.10.26 -- stop doing the normalization of weights!
      # new change on 22.10.24
      #new_weights = new_weights/torch.max(_cast_as_tensor(0.001), torch.mean(new_weights)); # ensures no div by zero
    elif self.normType == 2 or self.normType == 5 or self.normType == 6:
      # Relying on https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal.log_prob
      log_sfs = torch.log(sfs);
      if self.normType == 6:
        # simply compute the prefSf, and take the log of it
        weight_distr = torch.distributions.normal.Normal(torch.log(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf)), torch.clamp(self.gs_std, min=_cast_as_tensor(gs_std_min)));
      else:
        weight_distr = torch.distributions.normal.Normal(self.gs_mean, torch.clamp(self.gs_std, min=_cast_as_tensor(gs_std_min)));
      #weight_distr = torch.distributions.normal.Normal(self.gs_mean, torch.abs(self.gs_std))
      new_weights = torch.exp(weight_distr.log_prob(log_sfs));
      # --- 221026a --> normalize by the average across a reasonable range?
      avg_weights = torch.exp(weight_distr.log_prob(torch.log(_cast_as_tensor(np.geomspace(0.1,30,31)))))
      new_weights = torch.div(new_weights, torch.mean(avg_weights));
      # adding min. to ensure we don't div.by 0
      gain_curr = torch.max(_cast_as_tensor(0.0001), torch.mul(_cast_as_tensor(_sigmoidGainNorm), torch.sigmoid(self.gs_gain))) if self.normType == 5 else _cast_as_tensor(1);
      #gain_curr = torch.mul(_cast_as_tensor(_sigmoidGainNorm), torch.sigmoid(self.gs_gain)) if self.normType == 5 else _cast_as_tensor(1);
      new_weights = torch.mul(gain_curr, torch.mul(lgnStage, new_weights));
    #'''
    if recenter_norm == 1: # we'll recenter this weighted normalization around 1 (by addition)
      normMin, normMax = torch.min(new_weights), torch.max(new_weights)
      centerVal = (normMax-normMin)/2
      toAdd = 1-centerVal-normMin; # to make the center of these weights at 1
      new_weights = toAdd + new_weights
      if threshWeights is not None:
        new_weights = torch.max(_cast_as_tensor(threshWeights), new_weights);
    elif recenter_norm == 2: # we'll recenter this weighted normalization by division, s.t. the AVERAGE is 1
      if avg_sfs is None:
        new_weights = new_weights/torch.max(_cast_as_tensor(0.001), torch.mean(new_weights)); # ensures no div by zero
      else: # THIS ONLY WORKS IF NO LGN!!! TO-DO (22.10.23): Remedy this - either compute lgn-front-end for avg_sfs OR abandon OR just wait for "proper" normalization (filters with defined bandwidth)
        # --- furthermore -- evaluate if we really need this normalization?
        weights_for_avg = torch.exp(weight_distr.log_prob(torch.log(avg_sfs)));
        new_weights = new_weights/torch.max(_cast_as_tensor(0.001), torch.mean(weights_for_avg)); # Compute the avg. 
      #new_weights = new_weights/torch.mean(new_weights);
      #if threshWeights is not None:
      #  new_weights = torch.max(_cast_as_tensor(threshWeights), new_weights);
    #'''
    elif recenter_norm == 3: # max will be 1
        new_weights = new_weights/torch.max(_cast_as_tensor(0.001), torch.max(new_weights)); # ensures no div by zero
    if threshWeights is not None:
      new_weights = torch.max(_cast_as_tensor(threshWeights), new_weights);
        
    return new_weights;

  def SimpleNormResp(self, trialInf, trialArtificial=None, recenter_norm=recenter_norm):
    if trialArtificial is not None:
      trialInf = trialArtificial;
    else:
      trialInf = trialInf;
    # cons (and wghts) will be (nComps x nTrials)
    wghts = self.genNormWeightsSimple(trialInf, recenter_norm=recenter_norm);
    # UPDATE 22.10.20
    # --- if we apply LGN to norm, then the weights already incorporate the contrast, so don't re-apply 
    if self.lgnFrontEnd > 0 and self.applyLGNtoNorm:
      resp = wghts;
    else:
      incl_cons = _cast_as_tensor(trialInf['con'])
      #incl_cons = torch.pow(_cast_as_tensor(trialInf['con']), self.respExp); 
      resp = torch.mul(wghts, incl_cons);
    resp = torch.pow(resp, self.respExp);

    # now put it all together
    respPerTr = resp.sum(1); # i.e. sum over components
    #respPerTr = torch.div(respPerTr, torch.mean(respPerTr));
    #respPerTr = torch.pow(resp.sum(1), 1./self.respExp); # i.e. sum over components, then sqrt
    return respPerTr; # will be [nTrials] -- later, will ensure right output size during operation    

  # gs_std_min = 0.05 [default value]
  def FullNormResp(self, trialInf, trialArtificial=None, debugFilters=False, debugQuadrature=False, debugFilterTemporal=False, gs_std_min=_cast_as_tensor(0.05), forceExpAt2=False, normOverwrite=False, minWeight=_cast_as_tensor(5e-8)): # minWeight was 0.005
    ''' Per discussions with Tony and Eero (Oct. 2022), need to re-incorporate a more realistic normalization signal
        --- 1. Including temporal dynamics
        --- 2. Allow for interactions between stimulus components if they appear within the filter pass-band
        --- 3. Multiple (or at least flexibly defined) filters rather than just one filter!
        As of 22.10.28, will assume that the image is already calculated!
        --- also start with filter construction from Robbe

        Calculation is as follows:
        - a: compute normalization pool filters (applies always, but only need to compute first time)
        - b: apply weights to each filter (all equal if untuned, otherwise compute current weight)
        - --- (b) only needed each time if tuned normalization
        - --- result of a&b is self.normFilters
        - c: apply LGN if applicable (TODO: CAN SKIP IF LGN is tensor)
        - d: finally, multiply the filters with the stimulus

        How we proceed through this function depends on whether or not the normalization is tuned AND whether LGN front end is on
        - 1. if untuned norm and no LGN, we only need to compute once
    '''

    assert self.stimRealImag is not None; # temporary?
    assert self.normFull is not None;
    assert self.lgnCalcDone is not None;

    any_debug = debugFilters or debugQuadrature or debugFilterTemporal;

    # get the current SFs and Cons (all other stimulus features are in stimRealImag or are irrelevant to this calculation
    stimSf = _cast_as_tensor(trialInf['sf'], self.device);
    stimCo = _cast_as_tensor(trialInf['con'], self.device);
    nStimComp     = hf.get_exp_params(self.expInd).nStimComp;

    if trialArtificial is not None:
      trialInf = trialArtificial;
    else:
      trialInf = trialInf;

    if self.normFilters is None or normOverwrite: 
      ########
      # a. First, construct the filter bank if it's not already constructed
      # --- independent of LGN and gain control tuning
      ########
      # if we don't have the full calculation, then we'll do the following:
      # --- compute the underlying filters, if needed (otherwise skip that calculation)
      # --- finally, apply the weights
      if self.normFiltersBasic is None or normOverwrite:
        basic_filters = [];
        # if the norm. is untuned, then we'll just compute once
        # Compute SF tuning
        for iB in range(len(self.normFull['nFilters'])):
          sfRel = stimSf.unsqueeze(-1) / self.normFull['prefSfs'][iB].unsqueeze(0).unsqueeze(0)
          s     = torch.pow(stimSf.unsqueeze(-1), self.normFull['dOrd'][iB]) \
                      * torch.exp(-self.normFull['dOrd'][iB]/2 * torch.pow(sfRel, 2));
          sMax  = torch.pow(self.normFull['prefSfs'][iB], self.normFull['dOrd'][iB]) * torch.exp(-self.normFull['dOrd'][iB]/2);
          if basic_filters == []:
            basic_filters = [s/sMax.unsqueeze(0).unsqueeze(0)];
          else:
            basic_filters.append(s/sMax.unsqueeze(0).unsqueeze(0));
        if self.fullDataset: # as always, only save calculations if full dataset
          self.normFiltersBasic = basic_filters;
      else:
        basic_filters = self.normFiltersBasic;
      ########
      # b. at this stage, the basic filters are already calculated! apply the weights (per bank)
      # ---- the per-filter weights will be applied at the end
      ########
      # - NOTE: This stage will be done every time if tuned normalization
      for iB, filts in zip(range(len(self.normFull['nFilters'])), basic_filters):
        curr_resp = self.normFull['norm_gain'][iB] * filts;
        if iB == 0:
            selSf = curr_resp
        else:
            selSf = [selSf, curr_resp];
      # unfold the selSf into [nTr x nComp x nFilters] --> but permute to [nFilters x nTr x nComp]
      selSf = torch.cat(selSf, dim=-1).permute((-1,0,1));
      if self.fullDataset: # we can only save the full filters w/initial gain if full dataset
        self.normFilters = selSf;
    else:
      selSf = self.normFilters;

    ########
    # c. Now, apply the LGN, if needed --> or we're done!
    ########
    if self.applyLGNtoNorm and ((self.lgnFrontEnd > 0 and self.normCalc is None) or normOverwrite): # why if self.normCalc is None, we've already saved the calc we need
      # unpack/fully compute LGN front end:
      # --- NOTE: Even if the LGN is being updated (i.e. lgnFrontEnd==3 or 4), we'll have called simpleResp_matMul before getting here
      # ----- this, these values will be updated to reflected any shift in the SF=
      selCon_p = self.selCon_p;
      selCon_m = self.selCon_m;
      selSf_p = self.selSf_p;
      selSf_m = self.selSf_m;

      if self.lgnConType == 1 or self.lgnConType == 5: # DEFAULT
        # -- then here's our final responses per component for the current stimulus
        # ---- NOTE: The real mWeight will be sigmoid(mWeight), such that it's bounded between 0 and 1
        if self.lgnConType==1:
          lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_m)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_p)));
        elif self.lgnConType==5:
          #### TEMP: Only apply RVC, not SF filtering from LGN
          lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), selCon_m), torch.mul(1-torch.sigmoid(self.mWeight), selCon_p));
      elif self.lgnConType == 2 or self.lgnConType == 3 or self.lgnConType == 4:
        # -- Unlike the above (default) case, we don't allow for a separate M & P RVC - instead we just take the average of the two
        avgWt = torch.sigmoid(self.mWeight);
        selCon_avg = avgWt*selCon_m + (1-avgWt)*selCon_p;
        lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_avg)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_avg)));

      if self.normFiltersToOne:
        # normalize LGN s.t. max is 1
        lgnSel = torch.div(lgnSel, torch.max(lgnSel));
        #lgnSel = torch.div(lgnSel, torch.median(lgnSel)); # weird --> ignore
      # and apply to selSf
      # --- note: selSf is [nFilt x nTr x nComps] while lgnSel is [nTr x nComps] --> unsqueeze in zero dim to [1 x nTr x nComps]
      selSi = torch.mul(selSf, lgnSel.unsqueeze(0)); # filter sensitivity for the sinusoid in the frequency domain
    else:
      selSi = selSf;
    # all of LGN filtering, if applicable, is done!

    if debugFilters:
      return selSi, stimSf;
    
    ########
    # d. Final calculation (applying the filters to the image, then the weights to the responses per filter)
    ########
    if self.normCalc is None or any_debug or normOverwrite:
      # Here is the core computation - multiply the filter(s) with the stimulus image
      # --- however, because each filter will have it's own rectification, we have to do this operation separately per filter
      if self.lgnFrontEnd > 0: # contrast already included: TRY 22.10.12
        # selSf is now [nTr x nFr x nFilt]
        rComplex = torch.einsum('fij,ikjz->fikz', selSi, self.stimRealImag) # mult. to get [nFilt x nTr x nFr x 2] response
      else: # need to include contrast here
        rComplex = torch.einsum('fij,ikjz->fikz', torch.mul(selSi,stimCo.unsqueeze(0)), self.stimRealImag)

      # Now, get each filter as in quadrature set, rectify, apply non-linearity
      if self.normFiltersToOne:
        # --- here is (hopefully) slightly faster!
        to_div = torch.max(torch.max(rComplex), -torch.min(rComplex)); # 22.10.30 - might not need to divide by max of each, separately?
        # --- clamp is faster than max
        rsAlt = torch.div(rComplex, torch.max(_cast_as_tensor(globalMinDiv), to_div));
      else: # no intermediate norm.
        rsAlt = rComplex;

      if forceExpAt2: # this is faster, even though respExp is just one value
        rsAlt_pos = torch.pow(torch.clamp(rsAlt, min=_cast_as_tensor(globalMin)), 2);
        rsAlt_neg = torch.pow(torch.clamp(_cast_as_tensor(-1)*rsAlt, min=_cast_as_tensor(globalMin)), 2);
      else:
        rsAlt_pos = torch.pow(torch.clamp(rsAlt, min=_cast_as_tensor(globalMin)), self.respExp);
        rsAlt_neg = torch.pow(torch.clamp(_cast_as_tensor(-1)*rsAlt, min=_cast_as_tensor(globalMin)), self.respExp);
      rSimple1 = rsAlt_pos[...,0];
      rSimple2 = rsAlt_pos[...,1];
      rSimple3 = rsAlt_neg[...,0];
      rSimple4 = rsAlt_neg[...,1];

      if debugQuadrature:
        return rSimple1, rSimple2, rSimple3, rSimple4;

      if debugFilterTemporal:
        rsall = torch.div(torch.pow(rSimple1, self.respExp) + torch.pow(rSimple2, self.respExp) + torch.pow(rSimple3, self.respExp) + torch.pow(rSimple4, self.respExp), 4);
      else:
        rsall = torch.div(rSimple1 + rSimple2 + rSimple3 + rSimple4, 4);
        # now, we have [nFilt x nTr x nFr] --> apply the weights HERE
        if self.normType == 0: # i.e. inhAsym
          filt_sfs = torch.cat([x for x in self.normFull['prefSfs']]).flatten();
          # if inhAsym, apply the weights at this stage
          # weights relative to mean of pool
          #curr_resp = (1 + torch.clamp(self.inhAsym,-0.3,0.3) * (torch.log(filt_sfs) - torch.mean(torch.log(self.normFull['prefSfs'][iB])))) * filts
          # weights relative to cell preference
          #all_weights = 1 + torch.clamp(self.inhAsym,-0.3,0.3) * (torch.log(filt_sfs) - torch.log(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf)))
          # --- but not clipped
          all_weights = 1 + self.inhAsym * (torch.log(filt_sfs) - torch.log(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf)))

        elif self.normType == 2 or self.normType == 5 or self.normType == 6 or self.normType == 4 or self.normType == 7: # i.e. tuned weights
          all_weights = [];
          for iB, filt_sfs in zip(range(len(self.normFull['nFilters'])), self.normFull['prefSfs']):
            log_sfs = torch.log(filt_sfs);
            # NOTE: 22.10.31 --> clamp the gs_mean at the 1st/last filters of the 1st bank of filters
            if self.normType == 6: # here, norm yoked to filter --> compute prefSf, and take the log of it
              weight_distr = torch.distributions.normal.Normal(torch.clamp(torch.log(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf)), min=torch.log(self.normFull['prefSfs'][0][0]), max=torch.log(self.normFull['prefSfs'][0][-1])), torch.clamp(self.gs_std, min=gs_std_min));
            elif self.normType == 4: # two-half Gaussian?
              weight_distr = flexible_Gauss([_cast_as_tensor(0), _cast_as_tensor(1), torch.clamp(torch.exp(self.gs_mean), min=self.normFull['prefSfs'][0][0], max=self.normFull['prefSfs'][0][-1]), self.gs_std_low, self.gs_std_high], torch.exp(log_sfs), minThresh=0);
            else:
              if self.dgNormFunc:
                sfRel = torch.div(filt_sfs, self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.gs_mean));
                #effDord = torch.pow(torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(self.gs_std)), 2);
                effDord = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(self.gs_std));
                s     = torch.pow(filt_sfs, effDord) * torch.exp(-effDord/2 * torch.pow(sfRel, 2));
                sMax  = torch.pow(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.gs_mean), effDord) * torch.exp(-effDord/2);
                weight_distr   = torch.div(s, sMax);
              else: # log-Gaussian
                weight_distr = torch.distributions.normal.Normal(torch.clamp(self.gs_mean, min=torch.log(self.normFull['prefSfs'][0][0]), max=torch.log(self.normFull['prefSfs'][0][-1])), torch.clamp(self.gs_std, min=gs_std_min));

            # clamp the weights to avoid near-zero values
            if self.normType == 4:
              new_weights = torch.clamp(weight_distr, min=minWeight)
            else:
              if self.dgNormFunc:
                new_weights = torch.clamp(weight_distr, min=minWeight)
              else:
                new_weights = torch.clamp(torch.exp(weight_distr.log_prob(log_sfs)), min=minWeight);
            if self.normType == 5: 
              new_weights = torch.pow(new_weights, self.gs_gain); # make the gs_gain act as a power on the weights?
            avg_weights = new_weights/torch.mean(new_weights); # make avg. weight = 1
            if all_weights == []:
              all_weights = [avg_weights];
            else:
              all_weights.append(avg_weights);
          # flatten the weights into a [nFilt x 1] from the lists
          all_weights = torch.cat([aw for aw in all_weights]).flatten();

        # now, make the weights 3dim using view or unsqueeze
        # apply weights:
        if self.normType != 1: # apply weights as needed
          # --- including a thresholding s.t. no negative weights
          rsall = torch.mul(torch.max(minWeight, all_weights).unsqueeze(-1).unsqueeze(-1), rsall);
        rsall = rsall.mean(0); # take the avg. across filters
      # at this point, rsall is [nTr x nFrames]
      if self.normFiltersToOne:
        rsall = rsall/torch.max(torch.mean(rsall, axis=1)); # take avg. across frames (i.e. per trial) and normalize all to the max avg.

      rsall = rsall.transpose(1,0); # transpose to make [nFr x nTr]
      if not any_debug and not self.mWeight.requires_grad and self.normType==1 and self.fullDataset and (self.lgnFrontEnd != 3 or self.lgnFrontEnd != 4): # we can only save the full calc IF the norm. is untuned AND the lgnFrontEnd is fixed
        # only overwrite/save if LGN and filter weights are not optimized-for (and if no debugging)!
        self.normCalc = rsall;
      return rsall;

    else: # we've already computed the normalization pool (only applies if norm. is flat - incl. LGN stage)!
      return self.normCalc;

  def respPerCell(self, trialInf, debug=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm, preCompOri=None, quadrature=False, normOverwrite=False):
    import cProfile
    # excitatory filter, first
    simpleResp = self.simpleResp_matMul(trialInf, sigmoidSigma=sigmoidSigma, preCompOri=preCompOri, quadrature=quadrature);

    # use below block for debugging speed of FullNormResp!
    #normResp = self.FullNormResp(trialInf); # [nFrames x nTrials]
    #pdb.set_trace();
    #cProfile.runctx('self.FullNormResp(trialInf)', {'self':self}, locals())

    if self.useFullNormResp:
      normResp = self.FullNormResp(trialInf, normOverwrite=normOverwrite); # [nFrames x nTrials]
    else:
      normResp = self.SimpleNormResp(trialInf, recenter_norm=recenter_norm); # [nFrames x nTrials]

    if self.newMethod == 1:
      Lexc = simpleResp; # [nFrames x nTrials]
      Linh = normResp; # un-normalized...
    else:
      Lexc = torch.div(simpleResp, torch.max(simpleResp)); # [nFrames x nTrials]
      Linh = torch.div(normResp, torch.max(normResp)); # normalize the normResp...

    # the below line assumes normType != 3 (which was only briefly used/explored, anyway...)
    sigmaFilt = torch.pow(torch.pow(_cast_as_tensor(10), self.sigma), 2); # i.e. square the normalization constant

    if debug:
      return Lexc, Linh, sigmaFilt;

    # naka-rushton style?
    numerator     = Lexc; # [nFrames x nTrials]
    denominator   = sigmaFilt + Linh; # NOTE 22.10.24 - already did respExp there
    if self.useFullNormResp:
      rawResp       = torch.div(numerator, denominator);
    else:
      rawResp       = torch.div(numerator, denominator.unsqueeze(0)); # unsqueeze(0) to account for the missing leading dimension (nFrames)
    ratio         = _cast_as_tensor(globalMin) + rawResp;

    if self.newMethod == 1 and self.normToOne == 1:
      # in this case, only apply the noiseLate and self.scale AFTER the FT
      # --- 22.10.13 --> don't thresh! That introduces oddities!!!
      respModel = ratio;
      return torch.transpose(respModel, 1, 0);
    elif self.newMethod == 1 and self.normToOne != 1:
      if fall2020_adj:
        if spring2021_adj:
          respModel     = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, torch.mul(torch.mul(_cast_as_tensor(_sigmoidScale), torch.sigmoid(self.scale)), ratio))); # why 10 as the scalar? From multiple fits, the value is never over 1 or 2, so 10 gives the parameter enough operating range
        else:
          respModel     = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, torch.mul(torch.abs(self.scale), ratio)));
      else:
        respModel     = torch.add(self.noiseLate, torch.mul(torch.abs(self.scale), ratio));
      return torch.transpose(respModel, 1, 0);
    else:
      meanRate      = ratio.mean(0);
      respModel     = torch.add(self.noiseLate, torch.mul(torch.abs(self.scale), meanRate));
      return respModel; # I don't think we need to transpose here...

  def forward(self, trialInf, respMeasure=0, returnPsth=0, debug=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm, preCompOri=None, normOverwrite=False): # expInd=-1 for sfBB
    # respModel is the psth! [nTr x nFr]
    use_quadr = True if respMeasure==0 else False;
    respModel = self.respPerCell(trialInf, sigmoidSigma=sigmoidSigma, recenter_norm=recenter_norm, preCompOri=preCompOri, debug=debug, quadrature=use_quadr, normOverwrite=False);

    if debug:
      return respModel

    stimDur = hf.get_exp_params(self.expInd).stimDur
    nFrames = respModel.shape[1]/stimDur; # convert to frames per second...
    # We bifurcate here based on sfBB experiment or not - why? I originally designed the code with only the sfBB experiments in mind
    # - and it takes advantage of the TF for all (2) stimulus components being the same for all trials
    # - for sfMix experiments, we have to handle it differently, but spike_fft works accordingly
    if self.expInd == -1:
      # then, get the base & mask TF
      maskInd, baseInd = hf_sfBB.get_mask_base_inds();
      maskTf, baseTf = trialInf['tf'][0, maskInd], trialInf['tf'][0, baseInd] # relies on tf being same for all trials (i.e. maskTf always same, baseTf always same)!
      tfAsInts = [np.array([int(maskTf), int(baseTf)])] if respMeasure==1 else None;
      # important to transpose the respModel before passing in to spike_fft
      amps, rel_amps, full_fourier = spike_fft([respModel], tfs=tfAsInts, stimDur=stimDur, binWidth=1.0/nFrames)
    else:
      tfAsInts = trialInf['tf'] if respMeasure==1 else None;
      # important to transpose the respModel before passing in to spike_fft
      amps, rel_amps, full_fourier = spike_fft([respModel], tfs=tfAsInts, stimDur=stimDur, binWidth=1.0/nFrames)
    # NOTE: In the above, we pass in None for tfs if getting DC (won't use F1 anyway)!

    if respMeasure == 1: # i.e. F1
      to_use = rel_amps[0]; 
    else: # i.e. DC
      to_use = amps[0][:,0];
    if self.normToOne==1:
      to_use = _cast_as_tensor(globalMin) + torch.add(self.noiseLate, self.scale*to_use);
      #to_use = _cast_as_tensor(globalMin) + torch.add(self.noiseLate, self.scale*to_use/torch.max(to_use));
      #to_use = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, self.scale*to_use/torch.max(to_use)));
    if returnPsth == 1:
      return to_use, respModel;
    else:
      return to_use;

  def simulate(self, coreExp, respMeasure, con, sf, disp=None, nRepeats=None, baseOn=False, debug=False):
    ''' Simulate the model for stimuli which were not necessarily presented (i.e. freqs/cons not presented!)
        Procedure is:
        - 1. Generate new stimuli
        - 2. Call dataWrapper to package for model evaluation
        - 3. Clear saved model calculations
        - 4. Evaluate! (self.forward)
        - 5. Clear saved model calculations
        NOTE: We return responses as numpy array, not tensor
    '''
    # DETERMINE IF WE NEED TO DIVIDE BY stimDur
    stimDur = hf.get_exp_params(self.expInd).stimDur

    # We bifurcate here based on sfBB experiment or not
    if self.expInd == -1:
      new_stims = hf_sfBB.makeStimulusRef(coreExp, con, sf, nRepeats=nRepeats, baseOn=baseOn);
      new_wrap  = dataWrapper(new_stims, respMeasure=respMeasure, expInd=self.expInd, simulate=True)
      self.clear_saved_calcs();
      if debug:
        resps_sim = self.forward(new_wrap.trInf, respMeasure=respMeasure, normOverwrite=True, debug=debug)
        resps_sim = [x.detach().numpy() for x in resps_sim]
      else:
        resps_sim = self.forward(new_wrap.trInf, respMeasure=respMeasure, normOverwrite=True, debug=debug).detach().numpy();
      self.clear_saved_calcs();
      return resps_sim; # either nTr or [nTr, [mask,base]]
    else: # then we can use hf.makeStimulusRef
      new_stims = hf.makeStimulusRef(coreExp, disp, con, sf, expInd=self.expInd, nRepeats=nRepeats)
      new_wrap  = dataWrapper(new_stims, respMeasure=respMeasure, expInd=self.expInd, simulate=True)
      self.clear_saved_calcs();
      if debug:
        resps_sim = self.forward(new_wrap.trInf, respMeasure=respMeasure, normOverwrite=True, debug=debug)
        resps_sim = [x.detach().numpy() for x in resps_sim]
      else:
        resps_sim = self.forward(new_wrap.trInf, respMeasure=respMeasure, normOverwrite=True, debug=debug).detach().numpy();
      self.clear_saved_calcs();
      div_factor = stimDur if respMeasure==0 else 1./stimDur; # OK --> seemingly works with plot_diagnose_vLGN?
      if debug: # ignore div factor, at least for now...
        return resps_sim;
      else:
        return resps_sim/div_factor;

### End of class (sfNormMod)
    
def loss_sfNormMod(respModel, respData, lossType=1, debug=0, nbinomCalc=2, varGain=None):
  # nbinomCalc, varGain used only in lossType == 3

  if lossType == 1: # sqrt
      lsq = torch.pow(torch.sign(respModel)*torch.sqrt(torch.abs(respModel)) - torch.sign(respData)*torch.sqrt(torch.abs(respData)), 2);

      per_cond = lsq;
      NLL = torch.mean(lsq);

  if lossType == 2: # poiss TODO FIX TODO
      poiss_loss = torch.nn.PoissonNLLLoss(log_input=False);
      if debug == 1: # only eval if debug, since this is slow
        per_cond = _cast_as_tensor([poiss_loss(x,y) for x,y in zip(respModel, respData)]);
      NLL = poiss_loss(respModel, respData); # previously was respData, respModel

  if lossType == 3: # DEPRECATED AS OF 22.12.26 -- why? Used varGain 'place' in param_list for lgnCtfSf
      # varGain, respData, respMeans
      # -- all_counts is the spike count from every trial
      # -- count_mean is averaged across condition
      # - How does it work? Mu will need to be broadcast to be the same length as all_counts
      # - p is similarly broadcast, while r is just one value
      mu = torch.max(_cast_as_tensor(.1), respModel); # The predicted mean spike count
      # -- sigmoid(varGain) to ensure it's non-negative
      var = mu + (torch.sigmoid(varGain)*torch.pow(mu, 2)); # The corresponding variance of the spike count
      # Note: Two differeing versions of r - the first (doesn't use var) is from Robbe's code shared through Hasse/Mariana
      # -- the second is from the code that was in early code of my V1 model as written by Robbe
      if nbinomCalc == 1:
          r  = 1/varGain;
      elif nbinomCalc == 2:
          r  = torch.pow(mu, 2) / (var - mu); # The parameters r and p of the negative binomial distribution
      pSucc  = r / (r + mu);
      p = 1-pSucc; # why? Well, compared to scipy.stats.nbinom, the "p" here is for failure, not success, so it should be 1-p
      nbinomDistr = torch.distributions.negative_binomial.NegativeBinomial(r,p);
      # -- Evaluate the model
      # ---- but first, make any negative values just greater than zero (if F1 responses)
      if 'flo' in str(respData.dtype):
        respData = torch.max(_cast_as_tensor(1e-6), respData);
      llh = nbinomDistr.log_prob(respData); # The likelihood for each pass under the doubly stochastic model
      per_cond = -llh;
      NLL = torch.mean(-llh);

  if debug:
    return NLL, per_cond;
  else:
    return NLL;

### Now, actually do the optimization!

# ---def setModel(cellNum, expDir=-1, excType=1, lossType=1, fitType=1, lgnFrontEnd=0, lgnConType=1, applyLGNtoNorm=1, max_epochs=500, learning_rate=0.001, batch_size=128, scheduler=True
# ---def setModel(cellNum, expDir=-1, excType=1, lossType=1, fitType=1, lgnFrontEnd=0, lgnConType=1, applyLGNtoNorm=1, max_epochs=1000, learning_rate=0.01, batch_size=3000, scheduler=True
# learning_rate guide, as of 22.11.16:
# --- 0.01 if all data (e.g. batch_size=3000)
# --- 0.002 if batch_size = 256
######
## 23.01.04 and beyond: max_epochs should be 2500, learning_rate 0.0175, batch_size=3k
######
def setModel(cellNum, expDir=-1, excType=1, lossType=1, fitType=1, lgnFrontEnd=0, lgnConType=1, applyLGNtoNorm=1, max_epochs=2500, learning_rate=0.0175, batch_size=3000, scheduler=True, initFromCurr=0, kMult=0.1, newMethod=0, fixRespExp=None, trackSteps=True, trackStepsReduced=True, fL_name=None, respMeasure=0, vecCorrected=0, whichTrials=None, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm, to_save=True, pSfBound=14.9, pSfFloor=0.1, allCommonOri=True, rvcName = 'rvcFitsHPC_220928', rvcMod=1, rvcDir=1, returnOnlyInits=False, normToOne=True, verbose=True, singleGratsOnly=False, useFullNormResp=True, normFiltersToOne=False, preLoadDataList=None, k_fold=None, k_fold_shuff=True, k_fold_state=None, testingNames=False, dgNormFunc=False): # learning rate 0.04 on 22.10.01 (0.15 seems too high - 21.01.26); was 0.10 on 21.03.31;
  '''
  # --- max_epochs usually 7500; learning rate _usually_ 0.04-0.05
  # --- to_save should be set to False if calling setModel in parallel!
  # --- normToOne: if True, then the maximum response in selSi will be 1
  '''
  global dataListName
  global force_full
  
  ### Load the cell, set up the naming
  ########
  # Load cell
  ########
  loc_base = os.getcwd() + '/';
  loc_data = loc_base + expDir + 'structures/';

  if 'pl1465' in loc_base:
    loc_str = 'HPC';
  else:
    loc_str = '';
  #loc_str = 'HPC'; # use to override (mostly for debugging locally for HPC-based fits)

  if fL_name is None: # otherwise, it's already defined...
    if modRecov == 1:
      fL_name = 'mr_fitList%s_190516cA' % loc_str
    else:
      fL_name = 'fitList%s_pyt_nr230118a%s%s%s' % (loc_str, '_noRE' if fixRespExp is not None else '', '_noSched' if scheduler==False else '', '_sg' if singleGratsOnly else '');

  k_fold = 1 if k_fold is None else k_fold; # i.e. default to one "fold"
  todoCV = 1 if whichTrials is not None or k_fold>1 else 0;

  testingInfo = [max_epochs,learning_rate,batch_size]; # wrapping for naming purposes...
  fitListName = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=todoCV, excType=excType, lgnForNorm=applyLGNtoNorm, testingNames=testingNames, testingInfo=testingInfo, dgNormFunc=dgNormFunc);
  if todoCV and initFromCurr == 1: # i.e. we want to pre-initialize, but we're doing C-V...
    fitListName_nonCV = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=0, excType=excType, lgnForNorm=applyLGNtoNorm, testingNames=testingNames, testingInfo=testingInfo, dgNormFunc=dgNormFunc);
  else:
    fitListName_nonCV = None;
  print('applying LGN to norm? %d [fitList %s]' % (applyLGNtoNorm, fitListName))
  fitType_simpler, lgnFrontEnd_simpler = hf.get_simpler_mod(fitType, lgnFrontEnd)
  if fitType==fitType_simpler and lgnFrontEnd==lgnFrontEnd_simpler: # i.e. we were already at the simplest model
    if initFromCurr == -1: # then don't initialize bother initializing with simpler model (i.e. initFromCurr = -1)
      print('***Cancelling initFromCurr == -1***')
      initFromCurr = 0;
  fitListName_simpler = hf.fitList_name(base=fL_name, fitType=fitType_simpler, lossType=lossType, lgnType=lgnFrontEnd_simpler, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=todoCV, excType=excType, lgnForNorm=applyLGNtoNorm, testingNames=testingNames, testingInfo=testingInfo, dgNormFunc=dgNormFunc);
  # get the name for the stepList name, regardless of whether or not we keep this now
  stepListName = str(fitListName.replace('.npy', '_details.npy'));

  if verbose:
    print('\nFitList: %s [expDir is %s]' % (fitListName, expDir));

  # Load datalist, then specific cell
  if preLoadDataList is None:
    try:
      dataList = hf.np_smart_load(str(loc_data + dataListName));
    except:
      dataListName = hf.get_datalist(expDir, force_full=force_full, new_v1=True);
      dataList = hf.np_smart_load(str(loc_data + dataListName));
  else: # avoid loading it in mp.pool
    dataList = preLoadDataList;
    
  dataNames = dataList['unitName'];
  if verbose:
    print('loading data structure from %s...' % loc_data);
  try:
    expInd = hf.exp_name_to_ind(dataList['expType'][cellNum-1]);
  except:
    if expDir == 'V1_BB/':
      expInd = -1; # for sfBB
    elif expDir == 'V1_orig/':
      expInd = 1;
  if verbose:
    print('expInd is %d' % expInd);
  # - then cell
  if expInd == -1:
    S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfBB.npy')); # why -1? 0 indexing...
  else:
    S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfm.npy')); # why -1? 0 indexing...
  if expInd == -1:
    expInfo = S['sfBB_core'];
  else:
    expInfo = S['sfm']['exp']['trial'];

  respOverwrite = None; # default to None, but if vecCorrected and expInd != -1, then we will specify
  if verbose:
    print('respMeasure, vecCorrected: %d, %d' % (respMeasure, vecCorrected));
  if respMeasure == 1 and expInd!=1: # we cannot do F1 on V1_orig
    # NOTE: For F1, we keep responses per component, and zero-out the blanks later on
    if vecCorrected: # then do vecF1 correction
      if expInd == -1:
        # Overwrite f1 spikes
        vec_corr_mask, vec_corr_base = hf_sfBB.adjust_f1_byTrial(expInfo);
        expInfo['f1_mask'] = vec_corr_mask;
        expInfo['f1_base'] = vec_corr_base;
      else:
        respOverwrite = hf.adjust_f1_byTrial(expInfo, expInd);
    else: # then do phAmp correction here!
      if expInd == -1:
        # Overwrite f1 --> for BB, must pretend that it's still vecCorrectedF1 (required for the default phAdvCorr to apply)
        _, _, _, maskF1byPhAmp = hf_sfBB.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=1, returnByTr=1);
        vec_corr_mask, vec_corr_base = hf_sfBB.adjust_f1_byTrial(expInfo, maskF1byPhAmp=maskF1byPhAmp);
        expInfo['f1_mask'] = vec_corr_mask;
        expInfo['f1_base'] = vec_corr_base;
      else: # and the same for the "main" experiment
        rvcFits = hf.get_rvc_fits(loc_data, expInd, cellNum, rvcName=rvcName, rvcMod=rvcMod, direc=rvcDir, vecF1=0);
        respOverwrite, whichMeasure_out = hf.get_adjusted_spikerate(expInfo, cellNum, expInd, loc_data, rvcName=rvcFits, rvcMod=-1, baseline_sub=False, return_measure=True, vecF1=0, force_f1=True, returnByComp=True); # if we're here, then we get F1 regardless of f1f0
        # however, if whichMeasure is D.C., then we quit!
        if whichMeasure_out==0: # i.e. dc
          print('Cell %d failed in getting F1 responses' % cellNum);
          return [], []; # this one failed --> could not get DC
  elif respMeasure == 1 and expInd==1:
    if to_save:
      sys.exit('Cannot run F1 model analysis on V1_orig/ experiment - exiting!');
    else:
      return [], []; # return two blank placeholders so that the parallelization can move on

  if k_fold>1:
    nTrs_total = expInfo['num'][-1] if expInd != -1 else expInfo['trial']['ori'].shape[-1];
    k_fold_state = k_fold_state if k_fold_state is not None else cellNum; # why make the random state based on cell#? To make all model iterations for a given cell fit the same sets of trials!
    cv_gen = KFold(n_splits=k_fold, shuffle=k_fold_shuff, random_state=k_fold_state).split(range(nTrs_total));
    # and pre-create lists for relevant info:
    NLL_cv = [];
    testNLL_cv = [];
    params_cv = [];
    init_params_cv = [];
  else:
    cv_gen = range(1);

  ###################
  ###### WORKING/TODO [22.11.18] handle cross-validation within each setModel call
  #####  ----- why? For one, each experiment/cell will have a different number of trials, so we cannot call the kfold from outside of setModel when we parallelize
  #####  ----- ... without doing extra work
  #####  ----- Outline:
  #####  ------- if not doing C-V, then no extra work!
  #####  ------- if we ARE doing cross-val, then 
  #####  --------- we unpack the SKlearn KFold().split() call into train/test
  #####  --------- our code already handles the training easily!
  #####  ----------- at the end, evaluate the loss on the heldout/test data
  #####  ----------- decide what to include in the saving of each cell's optimization
  #####  ----------- package it all in an easy to unpack way --> c'est la vie!
  ###################
  for iii, fold_i in enumerate(cv_gen):
    if k_fold == 1:
      whichTrials = whichTrials
    else:
      whichTrials, testTrials = fold_i # unpack

    try:
      trInf, resp = process_data(expInfo, expInd=expInd, respMeasure=respMeasure, whichTrials=whichTrials, respOverwrite=respOverwrite, singleGratsOnly=singleGratsOnly);
      resps_detached = np.nansum(resp.detach().numpy(), axis=1);
      # however, this ignores the blanks -- so we have to reconstitute the original order/full experiment, filling in these responses in the correct trial locations
      #nTrs_total = len(trInf['num'])
      ######### --- NOTE: trying to replace nTrs_total with the above call? (i.e. before this loop)
      nTrs_total = expInfo['num'][-1] if expInd != -1 else expInfo['trial']['ori'].shape[-1];
      #########
      resps_full = np.nan * np.zeros((nTrs_total, ));
      resps_full[trInf['num']] = resps_detached;
      if expInd != -1: # i.e. anything but B+B
        _, _, expByCond, _ = hf.organize_resp(resps_full, expInfo, expInd);
      else:
        dcResp, f1Resp = hf_sfBB.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecCorrected);
        if respMeasure == 0:
          expByCond = dcResp[:,:,0]
        else:
          expByCond = f1Resp[:,:,0,0] if vecCorrected else f1Resp[:,:,0]; 
      unique_sfs = np.unique(trInf['sf'][:,0])
      # NOTE: Before 23.01.08, this was initialization...incl. for 23.01.04 fits [v successful]
      #pref_sf = unique_sfs[np.argmax(expByCond[0,:,-1])] if expInd != -1 else expInfo['baseSF'][0]; # we can just use baseSf
      # the below is a new attempt...
      if expInd == -1: # i.e. B+B --> don't just assume the baseSF is the preference, since sometimes we were off and other cells are pulled off the array, not tailored to...
        # take C.O.M.??
        mn_perSf = np.nanmean(expByCond,axis=0);
        pref_sf = np.power(2, np.nansum(np.log2(unique_sfs)*mn_perSf)/np.nansum(mn_perSf))
        #pref_sf = unique_sfs[np.argmax(mn_perSf)]; # add up all responses over contrast, which SF has highest...
      else:
        #pref_sf = unique_sfs[np.argmax(expByCond[0,:,-1])];
        # take C.O.M.??
        try:
          mn_perSf = np.nanmean(expByCond[0], axis=1); # avgs. over SF (note opposite arrangement of sfBB/expInd=-1)
          pref_sf = np.power(2, np.nansum(np.log2(unique_sfs)*mn_perSf)/np.nansum(mn_perSf))
        except:
          pref_sf = unique_sfs[np.argmax(expByCond[0,:,-1])];
      if verbose:
        print('prefSf: %.2f [will initialize at this + some jitter]' % pref_sf);
    except:
      if to_save:
        raise Exception("Could not process_data in mrpt.setModel --> cell %d, respMeasure %d" % (cellNum, respMeasure))
      else:
        return [], [];
    # we zero out the blanks later on for all other loss types, but do it here otherwise
    if lossType == 3:
      if respMeasure == 1:
        blanks = np.where(trInf['con']==0); # then we'll need to zero-out the blanks
        resp[blanks] = 1e-6; # TODO: should NOT be magic number (replace with global_min?)
      orgMeans = _cast_as_tensor(organize_mean_perCond(trInf, resp));

    respStr = hf_sfBB.get_resp_str(respMeasure);

    if initFromCurr == -1:
      # Let's try to load the parameters from a simpler fit:
      # - i.e., if no LGN and weighted, then flat model parameters
      # ------- if LGN, then equivalent model without LGN
      # ------- if LGN which can be shifted off of the standard, then the main LGN fit
      try:
        fitList_simpler = hf.np_smart_load(str(loc_data + fitListName_simpler));
        curr_fit_smpl = fitList_simpler[cellNum-1][respStr];
        curr_params = curr_fit_smpl['params'];
        testModel = sfNormMod(curr_params, expInd=expInd, excType=excType, normType=fitType_simpler, lossType=lossType, lgnConType=lgnConType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd_simpler, applyLGNtoNorm=applyLGNtoNorm, normToOne=normToOne, useFullNormResp=useFullNormResp, normFiltersToOne=normFiltersToOne, toFit=False, dgNormFunc=dgNormFunc)
        # NOW, we should re-package the parameters into the shape they would be for the current fitType
        # create interim initial param list
        cp = np.zeros((hf.nParamsByType(fitType=fitType, excType=excType, lgnType=lgnFrontEnd), ));
        # first 8 params will be in common -- always
        cp[0:8] = curr_params[0:8];
        if fitType_simpler == 1: # i.e. the parameters we're initiazing from are flat (untuned) normalization
          if fitType != 1: # then we're initializing a non-flat model fit from a flat model --> will need to guess that weighting parameters
            # --- note: assumes fitType = 2 or 6 
            # --- furthermore, we start with high normStd [to mimic flat] and normMean near prefSf
            init_gs_mean = _cast_as_tensor(torch.log(testModel.minPrefSf + testModel.maxPrefSf*torch.sigmoid(testModel.prefSf)));
            init_gs_std = 4
            cp[8] = init_gs_mean; #
            cp[9] = init_gs_std;
        if lgnFrontEnd_simpler>0: # the model we're initializing from has an LGN front end
          cp[-1] = curr_params[-1];
        # and copy it back
        curr_params = np.copy(cp);
        print('\n------ successful initFromCurr -1!! ------');
      except:
        initFromCurr = 0; # then we will not initFromCurr...
        pass; # we'll just proceed below (and will skip any initFromCurr later on)

    if os.path.isfile(loc_data + fitListName): # do not overwrite our curr_params if -1
      fitList = hf.np_smart_load(str(loc_data + fitListName));
      try:
        curr_fit = fitList[cellNum-1][respStr];
        if initFromCurr != -1 and fitListName_nonCV is None: # then we want to get initial parameters; otherwise, we already set up initial parameters!
          curr_params = curr_fit['params'];
          # Run the model, evaluate the loss to ensure we have a valid parameter set saved -- otherwise, we'll generate new parameters
          testModel = sfNormMod(curr_params, expInd=expInd, excType=excType, normType=fitType, lossType=lossType, lgnConType=lgnConType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd, applyLGNtoNorm=applyLGNtoNorm, normToOne=normToOne, useFullNormResp=useFullNormResp, normFiltersToOne=normFiltersToOne, toFit=False, dgNormFunc=dgNormFunc)
          trInfTemp, respTemp = process_data(expInfo, expInd, respMeasure, respOverwrite=respOverwrite, singleGratsOnly=singleGratsOnly) # warning: added respOverwrite here; also add whichTrials???
          predictions = testModel.forward(trInfTemp, respMeasure=respMeasure);
          if testModel.lossType == 3: # DEPRECATED AS OF 22.12.16 - SHOULD NOT REACH HERE
            loss_test = loss_sfNormMod(_cast_as_tensor(predictions.flatten()), _cast_as_tensor(respTemp.flatten()), testModel.lossType, varGain=testModel.varGain)
          else:
            loss_test = loss_sfNormMod(_cast_as_tensor(predictions.flatten()), _cast_as_tensor(respTemp.flatten()), testModel.lossType)
            if np.isnan(loss_test.item()):
              initFromCurr = 0; # then we've saved bad parameters -- force new ones!
      except:
        initFromCurr = 0; # force the standard initialization routine
    else:
      initFromCurr = 0 if (initFromCurr==1 and fitListName_nonCV is None) else initFromCurr; # if we were trying to initFromCurr (1) AND this isn't a CV fit, we failed; otherwise, still try initFromCurr?
      fitList = dict();
      curr_fit = dict();
    # Not quite done with initialization...
    # ...if this is a CV fit and initFromCurr==1, then we should try to load the non-CV parameters
    if fitListName_nonCV is not None: # try to load those non-CV fits, get the params
      try:
        fitList_nonCV = hf.np_smart_load(str(loc_data + fitListName_nonCV));
        nonCV_fit = fitList_nonCV[cellNum-1][respStr];
        curr_params = nonCV_fit['params'];
        print('initializing from non-CV parameters!')
      except: # if this doesn't work, then we just give up on initializing from previous fit
        initFromCurr = 0;

    ########
    ### set parameters
    ########
    # --- first, estimate prefSf, normConst if possible; inhAsym, normMean/Std
    prefSfEst_goal = np.random.uniform(0.75, 1.5) * pref_sf; # validated as good; late 2022
    sig_inv_input = (pSfFloor+prefSfEst_goal)/pSfBound;
    prefSfEst = -np.log((1-sig_inv_input)/sig_inv_input)
    # --- then, set up each parameter
    pref_sf = float(prefSfEst) if initFromCurr==0 else curr_params[0];
    if excType == 1:
      #dOrd_preSigmoid = np.random.uniform(0.5, 1.15)
      #dOrd_preSigmoid = np.random.uniform(1, 2.5) # validated as good; used as of late 2022 (incl. 23.01.04 fits)

      # new attempt on 23.01.08
      dOrd_preSigmoid = np.random.uniform(1, 2.5) if pref_sf>=2 else np.random.uniform(0.35, 0.75);
      dOrdSp = -np.log((_sigmoidDord-dOrd_preSigmoid)/dOrd_preSigmoid) if initFromCurr==0 else curr_params[1];
    elif excType == 2:
      if _sigmoidSigma is None:
        sigLow = np.random.uniform(0.1, 0.3) if initFromCurr==0 else curr_params[1];
        # - make sigHigh relative to sigLow, but bias it to be lower, i.e. narrower
        sigHigh = sigLow*np.random.uniform(0.5, 1.25) if initFromCurr==0 else curr_params[-1-np.sign(lgnFrontEnd)]; # if lgnFrontEnd == 0, then it's the last param; otherwise it's the 2nd to last param
      else: # this is from modCompare::smarter initialization, assuming _sigmoidSigma = 5
        sigLow = np.random.uniform(-2, -0.5);
        sigHigh = np.random.uniform(-2, -0.5);
    #####
    # norm: first normConst, then normMean/Std if applicable
    #####
    # 22.11.03 --> with not NormFiltersToOne, -1.5 works as a start except when lgnFrontEnd is on --> then make the normConst stronger to start
    if expInd == -1: # we want slightly stronger normalization for B+B to avoid ruining the base F1 response...
      normConst = 0.5 if normToOne==1 and normFiltersToOne else -1.25 + 2*np.sign(lgnFrontEnd); # per Tony, just start with a low value (i.e. closer to linear)
    else:
      normConst = 0.5 if normToOne==1 and normFiltersToOne else -1.5 + 2*np.sign(lgnFrontEnd); # per Tony, just start with a low value (i.e. closer to linear)
    if fitType <= 1:
      inhAsym = 0;
    if fitType == 2 or fitType == 5 or fitType == 6 or fitType == 4 or fitType == 7: # yes, we've put 4 last because it's rarely if ever used
      # see modCompare.ipynb, "Smarter initialization" for details
      if dgNormFunc:
        normMean = np.random.uniform(0.5, 1.5) * prefSfEst if initFromCurr==0 else curr_params[8];
        normStd = np.random.uniform(0.7,1.3) * dOrdSp if initFromCurr==0 else curr_params[9];
      else:
        normMean = np.random.uniform(0.75, 1.25) * np.log10(prefSfEst_goal) if initFromCurr==0 else curr_params[8]; # start as matched to excFilter
        normStd = np.random.uniform(0.3, 2) if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)
      if fitType == 5:
        normGain = np.random.uniform(-3, -1) if initFromCurr == 0 else curr_params[10]; # will be a sigmoid-ed value...
    normConst = normConst if initFromCurr==0 else curr_params[2];
    # --- then respExp and other scalars/"noise" terms
    if fixRespExp is not None:
      respExp = fixRespExp; # then, we set it to this value and make it a tensor (rather than parameter)
    else:
      respExp = np.random.uniform(1.5, 2.5) if initFromCurr==0 else curr_params[3];
    if newMethod == 0:
      # easier to start with a small scalar and work up, rather than work down
      respScalar = np.random.uniform(200, 700) if initFromCurr==0 else curr_params[4];
      noiseEarly = -1 if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
      noiseLate = 1e-1 if initFromCurr==0 else curr_params[6];
    else: # see modCompare.ipynb, "Smarter initialization" for details
      if respMeasure == 0: # slightly different range of successfully-fit respScalars for DC vs. F1 fits 
        if spring2021_adj:
          respScalar = np.random.uniform(-7, -2) if initFromCurr==0 else curr_params[4];
        else:
          respScalar = np.power(10, np.random.uniform(-1, 0)) if initFromCurr==0 else curr_params[4];
        if force_earlyNoise is None:
          noiseLate = np.random.uniform(-0.4, 0.4) if initFromCurr==0 else curr_params[6];
          noiseEarly = np.random.uniform(-0.5, 0.1) if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
        else: # again, from modCompare::smarter initialization, with force_noiseEarly = 0
          noiseLate = np.random.uniform(-2, 0.4) if initFromCurr==0 else curr_params[6];
          noiseEarly = force_earlyNoise if initFromCurr==0 else curr_params[6];
      elif respMeasure == 1:
        if spring2021_adj:
          respScalar = np.random.uniform(-12, -4) if initFromCurr==0 else curr_params[4];
        else:
          respScalar = np.power(10, np.random.uniform(-2.5, -0.5)) if initFromCurr==0 else curr_params[4];
        if force_earlyNoise is None:
          noiseLate = np.random.uniform(0, 1) if initFromCurr==0 else curr_params[6];
          noiseEarly = np.random.uniform(-0.5, 0.5) if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
        else: # again, from modCompare::smarter initialization, with force_noiseEarly = 0
          noiseLate = np.random.uniform(0, 3) if initFromCurr==0 else curr_params[6];
          noiseEarly = force_earlyNoise if initFromCurr==0 else curr_params[6];

      # in these cases, overwrite noiseLate and respScalar, since both are now applied AFTER the FFT
      if normToOne == 1:
        if expInd != -1:
          try:
            minResp, maxResp = np.nanmin(expByCond[0]), np.nanmax(expByCond[0]);
            assert(~np.isnan(minResp) & ~np.isnan(maxResp));
          except: # if either is nan, try all data (used for temp. mixtures-only effort on 23.01.09)
            minResp, maxResp = np.nanmin(expByCond), np.nanmax(expByCond);
        else:
          minResp, maxResp = np.nanmin(expByCond), np.nanmax(expByCond);
        if force_earlyNoise is None:
          noiseEarly = np.random.uniform(-0.003, 0) if initFromCurr==0 else curr_params[5]; # negative noiseEarly gives ODD results! helpful for strong, tuned suppression, but not good as a start
        else:
          noiseEarly = force_earlyNoise
        noiseLate = np.random.uniform(0.7, 1.3) * minResp if initFromCurr==0 else curr_params[6];
        #respScalar = np.random.uniform(0.9, 1.1) * (maxResp - noiseLate) if initFromCurr==0 else curr_params[4];
        # why div/40? Seems that high con, pref. SF only has FFT of ~40 spks/s
        # --- note: it WAS div/40 or div/20 when not full normResp --> not that it is, we use the below
        # -- if we don't do re-scaling below
        respScalar = np.random.uniform(0.6,1.2) * (maxResp-noiseLate)/2; # all heuristics...as of late 2022 (good)
        if normFiltersToOne and lgnFrontEnd>0:
          respScalar /= 500; # completely a heuristic!!!
        elif not normFiltersToOne and lgnFrontEnd>0:
          respScalar /= 250; # completely heuristic :(
        elif not normFiltersToOne and lgnFrontEnd==0:
          respScalar /= 750; # completely heuristic :(
        if (fitType==2 or fitType==6 or fitType==7) and lgnFrontEnd==0: # i.e. tuned gain
          respScalar *= 1.5; # need slighly stronger respScalar in these cases?
        # overwrite respScalar if initFromCurr!=0
        if initFromCurr!=0:
          respScalar = curr_params[4];
    if fitType>1: # i.e. not flat norm
      # increased starting value for width as of 22.10.25
      normStd = np.random.uniform(1.25, 2.25) if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)
      normGain = np.random.uniform(0.5, 1) if (initFromCurr == 0 or fitType!=5) else curr_params[10]; # will be a sigmoid-ed value...
      if fitType == 4:
        normStd_low, normStd_high = -1.5,-2; # start with the high freq. side being slightly narrower than the low freq. side
    lgnCtrSf = pref_sf # initialize to the same value as preferred SF (i.e. LGN centered around V1 prefSF)
    if lgnFrontEnd > 0:
      # Now, the LGN weighting 
      mWt_preSigmoid = np.random.uniform(0.25, 0.75); # this is what we want the real/effective mWeight initialization to be
      mWeight = -np.log((1-mWt_preSigmoid)/mWt_preSigmoid) if initFromCurr==0 else curr_params[-1];
    else:
      mWeight = -99; # just a "dummy" value

    # --- finally, actually create the parameter list
    if fitType <= 1:
      if excType == 1:
        param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, inhAsym, mWeight);
      elif excType == 2:
        param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, inhAsym, sigHigh, mWeight);
    elif fitType == 2 or fitType == 6 or fitType == 7:
      if excType == 1:
        param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, normMean, normStd, mWeight);
      elif excType == 2:
        param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, normMean, normStd, sigHigh, mWeight);
    elif fitType == 5:
      ### TODO: make this less redundant???
      if excType == 1:
        param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, normMean, normStd, normGain, mWeight);
      elif excType == 2:
        param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, normMean, normStd, normGain, sigHigh, mWeight);
    elif fitType == 4:
      ### TODO: make this less redundant???
      if excType == 1:
        param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, normMean, normStd_low, normStd_high, mWeight);
      elif excType == 2:
        param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, lgnCtrSf, normMean, normStd_low, normStd_high, sigHigh, mWeight);
    # After all possible model configs...
    if lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
      param_list = param_list[0:-1];   

    ### define model, grab training parameters
    model = sfNormMod(param_list, expInd, excType=excType, normType=fitType, lossType=lossType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd, lgnConType=lgnConType, applyLGNtoNorm=applyLGNtoNorm, pSfBound=pSfBound, pSfBound_low=pSfFloor, normToOne=normToOne, useFullNormResp=useFullNormResp, fullDataset=batch_size>2000, normFiltersToOne=normFiltersToOne, dgNormFunc=dgNormFunc)

    training_parameters = [p for p in model.parameters() if p.requires_grad]
    if verbose:
      model.print_params(); # optionally, nicely print the initial parameters...

    if returnOnlyInits: # 22.10.06 --> make an option to just use this for returning the initial parameters
      return param_list;

    ###  data wrapping
    #dw = dataWrapper(expInfo, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite, shufflePh=True, shuffleTf=True);
    dw = dataWrapper(expInfo, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite, singleGratsOnly=singleGratsOnly, whichTrials=whichTrials); # respOverwrite defined above (None if DC or if expInd=-1)
    exp_length = expInfo['num'][-1] if expInd!=-1 else expInfo['trial']['con'].shape[-1];
    dl_shuffle = False; # batch_size<2000 # i.e. if batch_size<2000, then shuffle!
    dl_droplast = bool(np.mod(exp_length,batch_size)<10) # if the last iteration will have fewer than 10 trials, drop it!
    dataloader = torchdata.DataLoader(dw, batch_size, shuffle=dl_shuffle, drop_last=dl_droplast)

    ### then set up the optimization
    optimizer = torch.optim.Adam(training_parameters, amsgrad=True, lr=learning_rate, ) # amsgrad is variant of opt
    # - and the LR scheduler, if applicable
    if scheduler:
      # value of 0.5 per Billy (21.02.09); patience is # of epochs before we start to reduce the LR
      LR_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                                factor=0.3, patience=np.maximum(10, int(max_epochs/10))); # factor was 0.5 when l.r. was 0.10; 0.15 when lr was 0.20
                                                                #factor=0.3, patience=np.maximum(8, int(max_epochs/10))); # factor was 0.5 when l.r. was 0.10; 0.15 when lr was 0.20
                                                                #factor=0.3, patience=int(max_epochs/15)); # factor was 0.5 when learning rate was 0.10; 0.15 when lr was 0.20

    # - then data
    # - predefine some arrays for tracking loss
    loss_history = []
    start_time = time.time()
    time_history = []
    model_history = []
    hessian_history = []

    if len(np.unique(trInf['ori']))==1: # then we can pre-compute the ori!
      only_ori = np.unique(trInf['ori']);
      preCompOri = [torch.cos(_cast_as_tensor((np.pi/180)*only_ori)), torch.sin(_cast_as_tensor((np.pi/180)*only_ori))]
    else:
      preCompOri = None;
    first_pred = model.forward(trInf, respMeasure=respMeasure, preCompOri=preCompOri);

    #import cProfile
    #cProfile.runctx('model.simpleResp_matMul(trInf, preCompOri=preCompOri)', {'model':model}, locals())
    #cProfile.runctx('model.forward(trInf, respMeasure=respMeasure, preCompOri=preCompOri)', {'model':model}, locals())

    accum = np.nan; # keep track of accumulator (will be replaced with zero at first step)
    for t in range(max_epochs):
        optimizer.zero_grad() # reset the gradient for each epoch!

        loss_history.append([])
        time_history.append([])

        for bb, (feature, target) in enumerate(dataloader):
            predictions = model.forward(feature, respMeasure=respMeasure, preCompOri=preCompOri)
            if respMeasure == 1: # figure out which stimulus components were blank for the given trials
              if expInd == -1:
                maskInd, baseInd = hf_sfBB.get_mask_base_inds();
                target['resp'][target['maskCon']==0, maskInd] = 1e-6 # force F1 ~ 0 if con of that stim is 0
                target['resp'][target['baseCon']==0, baseInd] = 1e-6 # force F1 ~ 0 if con of that stim is 0
                predictions[target['maskCon']==0, maskInd] = 1e-6 # force F1 ~ 0 if con of that stim is 0
                predictions[target['baseCon']==0, baseInd] = 1e-6 # force F1 ~ 0 if con of that stim is 0
              else:
                blanks = np.where(target['cons']==0);
                target['resp'][blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
                predictions[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
            target = target['resp'].flatten(); # since it's [nTr, 1], just make it [nTr] (if respMeasure == 0)
            predictions = predictions.flatten(); # either [nTr, nComp] to [nComp*nTr] or [nTr,1] to [nTr]
            if model.lossType == 3: # DEPRECATED - SHOULD NEVER REACH HERE as of 22.12.26
              loss_curr = loss_sfNormMod(predictions, target, model.lossType, varGain=model.varGain)
            else:
              loss_curr = loss_sfNormMod(predictions, target, model.lossType)

            if np.mod(t,500)==0: # and bb==0:
                if bb == 0:
                  now = datetime.datetime.now()
                  current_time = now.strftime("%H:%M:%S")
                  if verbose:
                    print('\n****** STEP %d [%s] [t=%s] [prev loss: %.3f] *********' % (t, respStr, current_time, accum))
                  accum = 0;
                prms = model.named_parameters()
                curr_loss = loss_curr.item();
                accum += curr_loss;

            if trackStepsReduced==False: # only then, track time_history
              loss_history[t].append(loss_curr.item())
              time_history[t].append(time.time() - start_time)
            else: # track only the loss, and make it float16 for smaller size!
              loss_history[t].append(np.float16(loss_curr.item()))
            if np.isnan(loss_curr.item()) or np.isinf(loss_curr.item()):
              if to_save: # we raise an exception here and then try again.
                raise Exception("Loss is nan or inf on epoch %s, batch %s!" % (t, 0))
              else: # otherwise, it's assumed that we're running this in parallel, so let's just give up on this cell!
                return [], []; # we'll just save empty lists...

            loss_curr.backward(retain_graph=True)
            optimizer.step()
            if scheduler:
              LR_scheduler.step(loss_curr.item());
            model.update_manual(verbose=False);

        model.eval()
        model.train()

    ##############
    #### OPTIM IS DONE ####
    ##############
    # Most importantly, get the optimal parameters
    opt_params = model.return_params(); # check this...
    curr_resp = model.forward(dw.trInf, respMeasure=respMeasure, preCompOri=preCompOri);
    gt_resp = _cast_as_tensor(dw.resp);
    # fix up responses if respMeasure == 1 (i.e. if mask or base con is 0, match the data & model responses...)
    if respMeasure == 1:
      if expInd == -1:
        maskInd, baseInd = hf_sfBB.get_mask_base_inds();
        curr_resp[dw.trInf['con'][:,maskInd]==0, maskInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
        curr_resp[dw.trInf['con'][:,baseInd]==0, baseInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
        gt_resp[dw.trInf['con'][:,maskInd]==0, maskInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
        gt_resp[dw.trInf['con'][:,baseInd]==0, baseInd] = 1e-6 # force F1 ~= 0 if con of that st
      else:
        blanks = np.where(dw.trInf['con']==0);
        curr_resp[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
        gt_resp[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0

    if model.lossType == 3: # DEPRECATED AS OF 22.12.26
      NLL = loss_sfNormMod(curr_resp.flatten(), gt_resp.flatten(), model.lossType, varGain=model.varGain).detach().numpy();
    else:
      NLL = loss_sfNormMod(curr_resp.flatten(), gt_resp.flatten(), model.lossType).detach().numpy();

    ## we've finished optimization, so reload again to make sure that this NLL is better than the currently saved one
    ## -- why do we have to do it again here? We may be running multiple fits for the same cells at the same and we want to make sure that if one of those has updated, we don't overwrite that opt. if it's better
    currNLL = 1e7;
    if os.path.exists(loc_data + fitListName) and to_save: # otherwise, no need to reload...
      fitList = hf.np_smart_load(str(loc_data + fitListName));
    try: # well, even if fitList loads, we might not have currNLL, so we have to have an exception here
      currNLL = fitList[cellNum-1][respStr]['NLL']; # exists - either from real fit or as placeholder
    except:
      pass; # we've already defined the currNLL...

    try:
      nll_history = fitList[cellNum-1][respStr]['nll_history'];
    except:
      nll_history = np.array([]);

    ##############
    # NOW - handle any cross-validation stuff
    # ----- pass the test trials into the datawrapper
    # ----- run those trials through the model
    # ----- evaluate the loss
    ##############
    if k_fold>1:
      dw_test = dataWrapper(expInfo, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite, singleGratsOnly=singleGratsOnly, whichTrials=testTrials);
      model.clear_saved_calcs();
      curr_resp = model.forward(dw_test.trInf, respMeasure=respMeasure, preCompOri=preCompOri, normOverwrite=True); # need to re-do the normalization!
      gt_resp = _cast_as_tensor(dw_test.resp);
      # fix up responses if respMeasure == 1 (i.e. if mask or base con is 0, match the data & model responses...)
      if respMeasure == 1:
        if expInd == -1:
          maskInd, baseInd = hf_sfBB.get_mask_base_inds();
          curr_resp[dw_test.trInf['con'][:,maskInd]==0, maskInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
          curr_resp[dw_test.trInf['con'][:,baseInd]==0, baseInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
          gt_resp[dw_test.trInf['con'][:,maskInd]==0, maskInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
          gt_resp[dw_test.trInf['con'][:,baseInd]==0, baseInd] = 1e-6 # force F1 ~= 0 if con of that st
        else:
          blanks = np.where(dw_test.trInf['con']==0);
          curr_resp[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
          gt_resp[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0

      if model.lossType == 3: # DEPRECATED AS OF 22.12.26
        testNLL = loss_sfNormMod(curr_resp.flatten(), gt_resp.flatten(), model.lossType, varGain=model.varGain).detach().numpy();
      else:
        testNLL = loss_sfNormMod(curr_resp.flatten(), gt_resp.flatten(), model.lossType).detach().numpy();
      
      # Since this is a C-V fit, save the NLL, testNLL, params as a list!
      NLL_cv.append(NLL);
      testNLL_cv.append(testNLL);
      params_cv.append(opt_params);
      init_params_cv.append(param_list);
      print('finished fold %d of %d' % (iii+1, k_fold))

  ########
  ### END OF k-fold loop!
  ########

  ### SAVE: Now we save the results, including the results of each step, if specified
  if verbose:
    print('...finished. New NLL (%.2f) vs. previous NLL (%.2f)' % (NLL, currNLL)); 
  # reload fitlist in case changes have been made with the file elsewhere!
  if os.path.exists(loc_data + fitListName) and to_save:
    fitList = hf.np_smart_load(str(loc_data + fitListName));
  # else, nothing to reload!!!
  # but...if we reloaded fitList and we don't have this key (cell) saved yet, recreate the key entry...

  # curr_fit will slot into fitList[cellNum-1][respStr]
  if k_fold>1: # i.e. if it's a cross-val thing, then we must update regardless
    curr_fit = dict();
    curr_fit['NLL_train'] = NLL_cv;
    curr_fit['NLL_test'] = testNLL_cv;
    curr_fit['params'] = params_cv;
    # NEW: Also save *when* this most recent fit was made (19.02.04); and nll_history below
    curr_fit['time'] = datetime.datetime.now();
    # NEW: Also also save entire loss/optimizaiotn structure
    optInfo = dict();
    optInfo['call'] = optimizer;
    # now, include some info about the cross-val!
    optInfo['k_fold'] = k_fold;
    optInfo['k_fold_random'] = k_fold_shuff;
    optInfo['test_len'] = dw.trInf['con'].shape[0];
    optInfo['train_len'] = dw_test.trInf['con'].shape[0];
    optInfo['epochs'] = max_epochs;
    optInfo['batch_size'] = batch_size;
    optInfo['learning_rate'] = learning_rate;
    optInfo['shuffle'] = dl_shuffle;
    optInfo['dropLast'] = dl_droplast;
    optInfo['init_params'] = init_params_cv; # should be helpful to know starting values
    curr_fit['opt'] = optInfo;
    curr_fit['nll_history'] = np.append(nll_history, NLL); # NOTE: The nll_history is just for the last fold!
  else:
    # now, if the NLL is now the best, update this
    if NLL < currNLL:
      curr_fit = dict();
      curr_fit['NLL'] = NLL;
      curr_fit['params'] = opt_params;
      # NEW: Also save *when* this most recent fit was made (19.02.04); and nll_history below
      curr_fit['time'] = datetime.datetime.now();
      # NEW: Also also save entire loss/optimizaiotn structure
      optInfo = dict();
      optInfo['call'] = optimizer;
      optInfo['epochs'] = max_epochs;
      optInfo['batch_size'] = batch_size;
      optInfo['learning_rate'] = learning_rate;
      optInfo['shuffle'] = dl_shuffle;
      optInfo['dropLast'] = dl_droplast;
      optInfo['init_params'] = param_list; # should be helpful to know starting values
      curr_fit['opt'] = optInfo;
      curr_fit['nll_history'] = np.append(nll_history, NLL);
    else:
      if verbose:
        print('new NLL not less than currNLL, not saving result, but updating overall fit list (i.e. tracking each fit)');

  if to_save:
    if cellNum-1 not in fitList:
      print('cell did not exist yet');
      fitList[cellNum-1] = dict();
      fitList[cellNum-1][respStr] = dict();
    elif respStr not in fitList[cellNum-1]:
      print('%s did not exist yet' % respStr);
      fitList[cellNum-1][respStr] = dict();
    else:
      print('we will be overwriting %s (if updating)' % respStr);
    # again, this is under to_save; otherwise, we're just going to return curr_fit
    fitList[cellNum-1][respStr] = curr_fit;
    np.save(loc_data + fitListName, fitList);

  if os.path.exists(loc_data + stepListName):
    try:
      stepList = hf.np_smart_load(str(loc_data + stepListName));
    except: # if the file is corrupted in some way...
      stepList = dict();
    try:
      curr_steplist = stepList[cellNum-1][respStr];
    except:
      curr_steplist = dict();
  else:
    stepList = dict();
    curr_steplist = dict();

  # now the step list, if needed
  if trackSteps and NLL < currNLL:
    curr_steplist = dict(); # again, will slot into stepList[cellNum-1][respStr]
    if trackStepsReduced==False: # only then, track time_history
      curr_steplist['time'] = time_history;
    # but always track loss_history
    curr_steplist['loss'] = loss_history;
 
    if to_save:
      if cellNum-1 not in stepList:
        print('[steplist] cell did not exist yet');
        stepList[cellNum-1] = dict();
        stepList[cellNum-1][respStr] = dict();
      elif respStr not in stepList[cellNum-1]:
        print('%s did not exist yet' % respStr);
        stepList[cellNum-1][respStr] = dict();
      else:
        print('we will be overwriting %s (if updating)' % respStr);
      stepList[cellNum-1][respStr] = curr_steplist;
      np.save(loc_data + stepListName, stepList);

  if to_save:
    return NLL, opt_params; #, loss_history; # LAST RETURN ITEM IS TEMPORARY 22.10.04
  else:
    return curr_fit, curr_steplist;

#############


if __name__ == '__main__':

    if len(sys.argv) < 9:
      print('uhoh...you need 9 arguments here'); # and one is the script itself...
      print('See this file (setModel) or batchFitUnix.sh for guidance');
      exit();

    cellNum      = int(sys.argv[1]);
    expDir       = sys.argv[2];
    excType      = int(sys.argv[3]);
    lossType     = int(sys.argv[4]);
    fitType      = int(sys.argv[5]);
    lgnFrontOn   = int(sys.argv[6]);
    initFromCurr = int(sys.argv[7]);
    trackSteps   = int(sys.argv[8]);

    if len(sys.argv) > 9:
      kMult = float(sys.argv[9]);
    else:
      kMult = 0.10; # default (see modCompare.ipynb for details)

    if len(sys.argv) > 10:
      newMethod = float(sys.argv[10]);
    else:
      newMethod = 0; # default

    if len(sys.argv) > 11:
      vecCorrected = int(sys.argv[11]);
    else:
      vecCorrected = 0;

    if len(sys.argv) > 12:
      fixRespExp = float(sys.argv[12]);
      if fixRespExp <= 0: # this is the code to not fix the respExp
        fixRespExp = None;
    else:
      fixRespExp = None; # default (see modCompare.ipynb for details)

    if len(sys.argv) > 13: # what model of contrast do we apply for the LGN (separate for M & P or a joint one?)
      lgnConType = int(sys.argv[13]);
      if lgnConType <= 0:
        lgnConType = 1;
    else:
      lgnConType = 1;

    if len(sys.argv) > 14:
      kfold = int(sys.argv[14]); # i.e. 5-fold
      if kfold <= 1:
        kfold = None;
    else:
      kfold = None; # i.e. not doing cross-val

    if len(sys.argv) > 15:
      dgNormFunc = int(sys.argv[15]);
    else:
      dgNormFunc = 0; # we default to NOT using deriv. Gauss for gain control tuning

    if len(sys.argv) > 16:
      _LGNforNorm = int(sys.argv[16]);
    else:
      _LGNforNorm = 1; # default to applying the LGN filters to the front-end
      
    #######
    # NOW: Note that the below values for optimization should be kept up-to-date with the defaults
    # ------ furthermore, note that these values are only passed in for parallel call (i.e. cellNum<0)
    #######
    if len(sys.argv) > 17:
      max_epochs = int(sys.argv[17]);
      print('\tspecified epochs: %d' % max_epochs);
    else:
      #max_epochs = 2500 if kfold is None else 1250; # fewer epochs when cross-val
      #max_epochs = 500; # use for temp/quick/debugging fits
      max_epochs = 250; # use for temp/quick/debugging fits
      
    if len(sys.argv) > 18:
      learning_rate = float(sys.argv[18]);
      print('\tspecified learning rate: %.2e' % learning_rate);
    else:
      #learning_rate = 0.0175; # the standard
      #learning_rate = 0.0375; # use for temp/quick/debugging fits
      learning_rate = 0.0575; # use for temp/quick/debugging fits

    if len(sys.argv) > 19:
      batch_size = int(sys.argv[19]);
      print('\tspecified batch_size: %d' % batch_size);
    else:
      batch_size = 3000;

    start = time.process_time();
    dcOk = 0; f1Ok = 0 if (expDir == 'V1/' or expDir == 'V1_BB/') else 1; # i.e. we don't bother fitting F1 if fit is from V1_orig/ or altExp/
    nTry = 1; # 30
    if cellNum >= 0:
      while not dcOk and nTry>0:
        try:
          setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule, singleGratsOnly=singleGratsOnly, k_fold=kfold, max_epochs=max_epochs, learning_rate=learning_rate, batch_size=batch_size, dgNormFunc=dgNormFunc); # first do DC

          #import cProfile
          #cProfile.runctx('setModel(cellNum, expDir, excType, lossType, fitType, 1, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule)', {'setModel': setModel}, locals())
          #cProfile.runctx('setModel(cellNum, expDir, excType, lossType, fitType, 0, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule)', {'setModel': setModel}, locals())

          dcOk = 1;
          print('passed with nTry = %d' % nTry);
        except Exception as e:
          print(e)
          pass;
        nTry -= 1;
      # now, do F1 fits

      nTry=1; #30; # reset nTry...
      while not f1Ok and nTry>0:
        try:
          setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule, singleGratsOnly=singleGratsOnly, k_fold=kfold, max_epochs=max_epochs, learning_rate=learning_rate, batch_size=batch_size, dgNormFunc=dgNormFunc); # then F1
          f1Ok = 1;
          print('passed with nTry = %d' % nTry);
        except Exception as e:
          print(e)
          pass;
        nTry -= 1;

    elif cellNum == -1 or cellNum == -2: # what is -2? Just for debugging purposes, restricts the celLNums to #nCpu (makes things faster)
      nCpu = 20; # mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU

      loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
      loc_data = loc_base + expDir + 'structures/';
      dataList = hf.np_smart_load(str(loc_data + dataListName));
      dataNames = dataList['unitName'];
      len_to_use = len(dataNames);
      len_to_use = len_to_use if cellNum==-1 or len_to_use<nCpu else nCpu 
      cellNums = np.arange(1, 1+len_to_use);
      # also, if cellNum == -2, make sure we flag that the names are test names
      testNames = True if cellNum == -2 else False;

      from functools import partial
      import multiprocessing as mp
      print('***cpu count: %02d***' % nCpu);
      loc_str = 'HPC' if 'pl1465' in loc_data else '';
      fL_name = 'fitList%s_pyt_nr230118a%s%s%s' % (loc_str, '_noRE' if fixRespExp is not None else '', '_noSched' if _schedule==False else '', '_sg' if singleGratsOnly else '');
      
      # do f1 here?
      sm_perCell = partial(setModel, expDir=expDir, excType=excType, lossType=lossType, fitType=fitType, lgnFrontEnd=lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule, to_save=False, singleGratsOnly=singleGratsOnly, fL_name=fL_name, preLoadDataList=dataList, k_fold=kfold, max_epochs=max_epochs, learning_rate=learning_rate, batch_size=batch_size, testingNames=testNames, dgNormFunc=dgNormFunc);
      with mp.Pool(processes = nCpu) as pool:
        smFits_f1 = pool.map(sm_perCell, cellNums); # use starmap if you to pass in multiple args
        pool.close();

      # First, DC? (should only do DC or F1?)
      sm_perCell = partial(setModel, expDir=expDir, excType=excType, lossType=lossType, fitType=fitType, lgnFrontEnd=lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule, to_save=False, singleGratsOnly=singleGratsOnly, fL_name=fL_name, preLoadDataList=dataList, k_fold=kfold, max_epochs=max_epochs, learning_rate=learning_rate, batch_size=batch_size, testingNames=testNames, dgNormFunc=dgNormFunc);
      with mp.Pool(processes = nCpu) as pool:
        smFits_dc = pool.map(sm_perCell, cellNums); # use starmap if you to pass in multiple args
        pool.close();

      ### do the saving HERE!
      todoCV = 0 if kfold is None else 1;
      testingInfo = [max_epochs,learning_rate,batch_size]; # wrapping for naming purposes...
      fitListName = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType, lgnType=lgnFrontOn, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=todoCV, excType=excType, lgnForNorm=_LGNforNorm, testingNames=testNames, testingInfo=testingInfo, dgNormFunc=dgNormFunc)
      if os.path.isfile(loc_data + fitListName):
        print('reloading fit list...');
        fitListNPY = hf.np_smart_load(loc_data + fitListName);
      else:
        fitListNPY = dict();
      # and load fit details
      fitDetailsName = fitListName.replace('.npy', '_details.npy');
      if os.path.isfile(loc_data + fitDetailsName):
        print('reloading fit list...');
        fitDetailsNPY = hf.np_smart_load(loc_data + fitDetailsName);
      else:
        fitDetailsNPY = dict();

      # now, iterate through and fit!
      respStr_dc = hf_sfBB.get_resp_str(0);
      respStr_f1 = hf_sfBB.get_resp_str(1);
      for (iii, currFit_dc), currFit_f1 in zip(enumerate(smFits_dc), smFits_f1):
        fitListNPY[iii] = dict();
        # dc
        fitListNPY[iii][respStr_dc] = currFit_dc[0];
        # f1
        fitListNPY[iii][respStr_f1] = currFit_f1[0];
        try: # try the details
          fitDetailsNPY[iii] = dict();
          fitDetailsNPY[iii][respStr_dc] = currFit_dc[1];
          fitDetailsNPY[iii][respStr_f1] = currFit_f1[1];
        except:
          pass;
      # --- finally, save
      np.save(loc_data + fitListName, fitListNPY)
      np.save(loc_data + fitDetailsName, fitDetailsNPY)

      from get_mod_varExpl import save_mod_varExpl
      if kfold is None:
        save_mod_varExpl(fL_name, expDir, fitType, lgnFrontOn, kfold, excType=excType, lossType=lossType, lgnConType=lgnConType, dgNormFunc=dgNormFunc);
      else:
        [save_mod_varExpl(fL_name, expDir, fitType, lgnFrontOn, kfold_curr, excType=excType, lossType=lossType, lgnConType=lgnConType, dgNormFunc=dgNormFunc) for kfold_curr in range(kfold)];

    enddd = time.process_time();
    print('Took %d minutes -- dc %d || f1 %d' % ((enddd-start)/60, dcOk, f1Ok));
