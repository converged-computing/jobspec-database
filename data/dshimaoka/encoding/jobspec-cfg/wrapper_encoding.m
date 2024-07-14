%wrapper_encoding.m
%this script loads processed data by makeDataBase.m,
%fit one pixel with ridge regression
%evaluate the fit result with in-silico simulation


if isempty(getenv('COMPUTERNAME'))
    addpath(genpath('~/git'));
    % addDirPrefs; %BAD IDEA TO write matlabprefs.mat in a batch job!!
    [~,narrays] = getArray('script_wrapper.sh');
else
    narrays = 1;
end


ID = 9;
doTrain = 1; %train a gabor bank filter or use it for insilico simulation
doRF = 1;
doORSF = 1;
subtractImageMeans = 0;
roiSuffix = '';
stimSuffix = '_square30_2';%
regressSuffix = '_nxv';

omitSec = 5; %omit initial XX sec for training
rescaleFac = 0.1;

expInfo = getExpInfoNatMov(ID);

%% draw slurm ID for parallel computation specifying ROI position
pen = getPen;


%% path
dataPaths = getDataPaths(expInfo,rescaleFac,roiSuffix, stimSuffix);
dataPaths.encodingSavePrefix = [dataPaths.encodingSavePrefix regressSuffix];

inSilicoRFStimName = [dataPaths.stimSaveName(1:end-4) '_insilicoRFstim.mat'];
inSilicoORSFStimName = [dataPaths.stimSaveName(1:end-4) '_insilicoORSFstim.mat'];


load( dataPaths.stimSaveName, 'TimeVec_stim_cat', 'dsRate','S_fin',...
    'gaborBankParamIdx','stimInfo');
if subtractImageMeans
    load(dataPaths.imageSaveName, 'imageMeans_proc');
else
    imageMeans_proc = [];
end


%% estimate the energy-model parameters w cross validation
nMovies = numel(stimInfo.stimLabels);
movDur = stimInfo.duration;%[s]
trainIdx = [];
for imov = 1:nMovies
    trainIdx = [trainIdx (omitSec*dsRate+1:movDur*dsRate)+(imov-1)*movDur*dsRate];
end

%% estimation of filter-bank coefficients
trainParam.regressType = 'ridge';
trainParam.ridgeParam = 1e6;%logspace(5,7,3); %[1 1e3 1e5 1e7]; %search the best within these values
trainParam.KFolds = 5; %cross validation. Only valid if numel(ridgeParam)>1
trainParam.tavg = 0; %tavg = 0 requires 32GB ram. if 0, use avg within Param.lagFrames to estimate coefficients
trainParam.Fs = dsRate; %hz after downsampling
trainParam.lagFrames = 2:4;%2:9;%round(0/dsRate):round(5/dsRate);%frame delays to train a neuron
trainParam.useGPU = 1; %for ridgeXs local GPU is not sufficient

%% in-silico simulation
analysisTwin = [2 trainParam.lagFrames(end)/dsRate];


%% stimuli
%load(dataPaths.imageSaveName,'stimInfo')

%% load neural data
%TODO: copy timetable data to local
disp('Loading tabular text datastore');
ds = tabularTextDatastore(dataPaths.timeTableSaveName);

nTotPix = numel(ds.VariableNames)-1;

%% retrieve unsuccessful analysis
tgtIdx = detectNGidx(dataPaths.encodingSavePrefix, nTotPix);
maxJID = numel(pen:narrays:numel(tgtIdx));

errorID=[];
for JID = 1:maxJID
    try
        disp([num2str(JID) '/' num2str(maxJID)]);
        
        roiIdx = tgtIdx(pen + (JID-1)*narrays);
        
        %TODO: save data locally
        encodingSaveName = [dataPaths.encodingSavePrefix '_roiIdx' num2str(roiIdx) '.mat'];
        
        trainAndSimulate(trainParam, ds, S_fin, roiIdx, trainIdx, stimInfo, imageMeans_proc, ...
            gaborBankParamIdx, encodingSaveName, inSilicoRFStimName, ...
            inSilicoORSFStimName, analysisTwin,doTrain, doRF, doORSF);
        
    catch err
        disp(err);
        errorID = [roiIdx errorID];
        continue
    end
end

%% re-run pixels with error
while ~isempty(errorID)
    try
        disp(['remaining ' num2str(errorID)]);
      
        roiIdx = errorID(1);
        encodingSaveName = [dataPaths.encodingSavePrefix '_roiIdx' num2str(roiIdx) '.mat'];
        
        trainAndSimulate(trainParam, ds, S_fin, roiIdx, trainIdx, stimInfo, imageMeans_proc, ...
            gaborBankParamIdx, encodingSaveName, inSilicoRFStimName,...
            inSilicoORSFStimName, analysisTwin,doTrain, doRF, doORSF);
        if numel(errorID)>1
            errorID = errorID(2:end);
        elseif numel(errorID)==1
            errorID=[];
        end
    catch err
        errorID = [errorID(2:end) errorID(1)];
    end
end


