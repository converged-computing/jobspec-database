
%makeStimDataBase.m:
%this script process and save stimulus data(saveGaborBankOut), together with in-silico simulation 
%that will be used for the model fitting (in MASSIVE) by wrapper_encoding.m

if isempty(getenv('COMPUTERNAME'))
    addpath(genpath('~/git'));
    % addDirPrefs; %BAD IDEA TO write matlabprefs.mat in a batch job!!    
    [~,narrays] = getArray('script_makeStimDataBase.sh');
    setenv('LD_LIBRARY_PATH', '/usr/local/matlab/r2021a/sys/opengl/lib/glnxa64:/usr/local/matlab/r2021a/bin/glnxa64:/usr/local/matlab/r2021a/extern/lib/glnxa64:/usr/local/matlab/r2021a/cefclient/sys/os/glnxa64:/usr/local/matlab/r2021a/runtime/glnxa64:/usr/local/matlab/r2021a/sys/java/jre/glnxa64/jre/lib/amd64/native_threads:/usr/local/matlab/r2021a/sys/java/jre/glnxa64/jre/lib/amd64/server:/usr/local/libjpeg-turbo/1.4.2/lib64:/opt/munge-0.5.14/lib:/opt/slurm-22.05.9/lib:/opt/slurm-22.05.9/lib/slurm:/usr/lib64:');
    addpath('/usr/bin/');

else
    narrays = 1;
end

%% draw slurm ID for parallel computation specifying stimulus ID    
pen = getPen; 


expID = 1;


roiSuffix = '';

%% imaging parameters
rescaleFac = 0.1;
procParam.cutoffFreq = 0.02; %0.1
procParam.lpFreq = []; %2
rotateInfo = [];
rebuildImageData = false;
makeMask = false;%true;
uploadResult = true;
dsRate = 1;%[Hz] %sampling rate of hemodynamic coupling function
useGPU = 1;

%% stimulus parameters
aparam = getAnalysisParam(ID);
stimXrange = aparam.stimXrange;
stimYrange = aparam.stimYrange;
stimSuffix = aparam.stimSuffix;

% gabor bank filter 
gaborBankParamIdx.cparamIdx = 1;
gaborBankParamIdx.gparamIdx = 2; %4 for only small RFs
gaborBankParamIdx.nlparamIdx = 1;
gaborBankParamIdx.dsparamIdx = 1;
gaborBankParamIdx.nrmparamIdx = 1;
gaborBankParamIdx.predsRate = 15; %Hz %mod(dsRate, predsRate) must be 0
%< sampling rate of gabor bank filter

expInfo = getExpInfoNatMov(expID);
dataPaths = getDataPaths(expInfo, rescaleFac, roiSuffix, stimSuffix);

%% load cic and stimInfo
load(dataPaths.imageSaveName,'cic','stimInfo');

%% motion-energy model computation from visual stimuli
if ~exist(dataPaths.stimSaveName,'file') 
    
    [stimInfo.stimXdeg, stimInfo.stimYdeg] = stimPix2Deg(stimInfo, stimXrange, stimYrange);
    screenPixNew = [max(stimYrange)-min(stimYrange)+1 max(stimXrange)-min(stimXrange)+1];
    stimInfo.width = stimInfo.width * screenPixNew(2)/stimInfo.screenPix(2);
    stimInfo.height = stimInfo.height * screenPixNew(1)/stimInfo.screenPix(1);
    stimInfo.screenPix = screenPixNew;
    
    %% prepare model output SLOW
    theseTrials = pen:narrays:cic.nrTrials;
    saveGaborBankOut(dataPaths, cic, dsRate, gaborBankParamIdx, ...
        stimYrange, stimXrange, theseTrials, useGPU);
    
    %% check if all data is processed
    [~,prefix] = fileparts(dataPaths.stimSaveName);
    fileExists = zeros(cic.nrTrials,1);
    for itr = 1:cic.nrTrials
        tempData = [prefix '_temp_' num2str(itr) '.mat'];
        
        fileExists(itr)=exist(tempData, 'file');
    end
    
    if all(fileExists)
        %% append across trials
        S_fin = [];TimeVec_stim_cat=[];
        for itr = 1:cic.nrTrials
            %tempData = ['preprocAll_temp_' num2str(itr)];
            [~,prefix] = fileparts(dataPaths.stimSaveName);
            tempData = [prefix '_temp_' num2str(itr) '.mat'];
            load(tempData,'frames_fin','TimeVec_stim');
            
            S_fin = cat(1,S_fin, frames_fin);
            if itr==1
                TimeVec_stim_cat = TimeVec_stim;
            else
                TimeVec_stim_cat = cat(1, TimeVec_stim_cat, TimeVec_stim_cat(end) + 1/dsRate + TimeVec_stim);
            end
        end
        
        %% save gabor filter output as .mat
        save( dataPaths.stimSaveName, 'TimeVec_stim_cat', 'S_fin', ...
            'gaborBankParamIdx', 'dsRate','cic','stimInfo');
        
        delete([prefix '*.mat']); %delete temporary files
    end
else
    save( dataPaths.stimSaveName, 'cic','stimInfo','-append');
end


