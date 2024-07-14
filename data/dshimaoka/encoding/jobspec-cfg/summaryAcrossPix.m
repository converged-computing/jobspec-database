if ~ispc
    addpath(genpath('~/git'));
    addDirPrefs;
end

expID = [1 2 3 8 9];
for ididx = 1:numel(expID)
    ID = expID(ididx);
    useGPU = 1;
    rescaleFac = 0.10;
    dsRate = 1;
    remakeSummary = 0;
    reAnalyze = 0;
    ORSFfitOption = 1; %3:peakSF,fitOR
    RFfitOption = 0; %1:count #significant pixels
    roiSuffix = '';
    aparam = getAnalysisParam(ID, stimSuffix);
    
    %% path
    expInfo = getExpInfoNatMov(ID);
    dataPaths = getDataPaths(expInfo,rescaleFac, roiSuffix, aparam.stimSuffix);
    encodingSavePrefix = [dataPaths.encodingSavePrefix aparam.regressSuffix];
    
    load(dataPaths.roiSaveName, 'X','Y','theseIdx','meanImage');
    thisROI = meanImage; %153x120
    roiIdx = 1:numel(X);
    %clear imageData;
    
    load(dataPaths.stimSaveName,'stimInfo')
    stimSz = [stimInfo.height stimInfo.width];
    clear stimInfo;
    
    load( dataPaths.stimSaveName, 'gaborBankParamIdx');
    
    
    if remakeSummary
        scrSz = size(thisROI);
        RF_Cx = nan(numel(thisROI),1);
        RF_Cy = nan(numel(thisROI),1);
        RF_sigma = nan(numel(thisROI),1);
        RF_mean =cell(numel(thisROI),1);% nan(scrSz(1), scrSz(2),18,32);
        RF_peakAmp = nan(numel(thisROI),1);
        expVal = nan(numel(thisROI),1);
        correlation = nan(numel(thisROI),1);
        bestSF = nan(numel(thisROI),1);
        bestOR = nan(numel(thisROI),1);
        bestSFF = nan(numel(thisROI),1);
        bestTF = nan(numel(thisROI),1);
        ridgeParam = nan(numel(thisROI),1);
        ngIdx = [];
        tic; %18h for CJ224 @officePC
        for ii = 1:numel(roiIdx)
            disp(ii)
            encodingSaveName = [encodingSavePrefix '_roiIdx' num2str(roiIdx(ii)) '.mat'];
            if exist(encodingSaveName,'file')
                try
                    encodingResult = load(encodingSaveName, 'RF_insilico','trained','trainParam');
                    RF_insilico = encodingResult.RF_insilico;
                    trained = encodingResult.trained;
                    trainParam = encodingResult.trainParam;
                catch err
                    disp(['MISSING ' encodingSaveName]);
                    ngIdx = [ngIdx roiIdx(ii)];
                    continue;
                end
            else
                disp(['MISSING ' encodingSaveName]);
                ngIdx = [ngIdx roiIdx(ii)];
                continue;
            end
            
            correlation(ii) = trained.corr;
            expVal(ii) = trained.expval;
            ridgeParam(ii) = trained.ridgeParam_optimal;
            
            %% RF
            %trange = [0 trainParam.lagFrames(end)/dsRate];
            trange = [2 trainParam.lagFrames(end)/dsRate];
            
            if reAnalyze
                if ID == 1 %|| ID == 2
                    RF_insilico.noiseRF.maxRFsize=15;%20;%30;%10;%7;%5;%3.5;
                elseif ID == 8
                    RF_insilico.noiseRF.maxRFsize=10;
                end
                RF_insilico = analyzeInSilicoRF(RF_insilico, -1, trange, ...
                    aparam.xlim, aparam.ylim, RFfitOption);
                %showInSilicoRF(RF_insilico, trange);
            end
            
            RF_Cx(ii) = RF_insilico.noiseRF.RF_Cx;
            RF_Cy(ii) = RF_insilico.noiseRF.RF_Cy;
            RF_sigma(ii) = RF_insilico.noiseRF.sigma;
            RF_peakAmp(ii) = RF_insilico.noiseRF.peakAmp;
            
            tidx = find(RF_insilico.noiseRF.RFdelay>=trange(1) & RF_insilico.noiseRF.RFdelay<=trange(2));
            RF_mean{ii} = mean(RF_insilico.noiseRF.RF(:,:,tidx),3);
            
            
            %% ORSF ... too heavy for officePC
            try
                if reAnalyze
                    RF_insilico = analyzeInSilicoORSF(RF_insilico, -1, trange, ORSFfitOption);
                end
                bestSF(ii) = RF_insilico.ORSF.bestSF;
                %bestOR(ii) = RF_insilico.ORSF.bestOR;
            catch err
                ngIdx = [ngIdx ii];
            end
            
            %% DIRSFTF
            bestSFF(ii) = RF_insilico.DIRSFTF.bestSF;
            bestTF(ii) = RF_insilico.DIRSFTF.bestTF;
            bestOR(ii) = RF_insilico.DIRSFTF.bestDIR;
            
            %bestAmp(ii) = amp;
        end
        t = toc;
        
        
        %% convert to 2D
        RF_Cx2 = nan(size(thisROI));
        RF_Cy2 = nan(size(thisROI));
        RF_sigma2 = nan(size(thisROI));
        RF_peakAmp2 = nan(size(thisROI));
        bestSF2 = nan(size(thisROI));
        bestOR2 = nan(size(thisROI));
        bestSFF2 = nan(size(thisROI));
        bestTF2 = nan(size(thisROI));
        expVal2 = nan(size(thisROI));
        correlation2 = nan(size(thisROI));
        ridgeParam2 = nan(size(thisROI));
        RF_mean2 = nan(size(thisROI,1),size(thisROI,2),RF_insilico.noiseRF.screenPix(1),...
            RF_insilico.noiseRF.screenPix(2));
        for ii = 1:numel(roiIdx)
            try
                RF_Cx2(Y(roiIdx(ii)),X(roiIdx(ii))) = RF_Cx(ii);
                RF_Cy2(Y(roiIdx(ii)),X(roiIdx(ii))) = RF_Cy(ii);
                expVal2(Y(roiIdx(ii)),X(roiIdx(ii))) = expVal(ii);
                correlation2(Y(roiIdx(ii)),X(roiIdx(ii))) = correlation(ii);
                ridgeParam2(Y(roiIdx(ii)),X(roiIdx(ii))) = ridgeParam(ii);
                RF_mean2(Y(roiIdx(ii)),X(roiIdx(ii)),:,:) = RF_mean{ii};
                bestSF2(Y(roiIdx(ii)),X(roiIdx(ii))) = bestSF(ii);
                bestOR2(Y(roiIdx(ii)),X(roiIdx(ii))) = bestOR(ii);
                RF_sigma2(Y(roiIdx(ii)),X(roiIdx(ii))) = RF_sigma(ii);
                RF_peakAmp2(Y(roiIdx(ii)),X(roiIdx(ii))) = RF_peakAmp(ii);
                
                bestSFF2(Y(roiIdx(ii)),X(roiIdx(ii))) = bestSFF(ii);
                bestTF2(Y(roiIdx(ii)),X(roiIdx(ii))) = bestTF(ii);
            catch err
                continue
            end
        end
        
        mask = nan(size(thisROI));
        for ii = 1:numel(roiIdx)
            mask(Y(roiIdx(ii)),X(roiIdx(ii))) = 1;
        end
        if ID==9
            mask(49:end,:)=0;
        end
        
        summary.RF_Cx = RF_Cx2;
        summary.RF_Cy = RF_Cy2;
        summary.RF_sigma = RF_sigma2;
        summary.RF_mean = RF_mean2;
        summary.ridgeParam = ridgeParam2;
        summary.RF_mean = RF_mean2;
        summary.bestSF = bestSF2;
        summary.bestOR = bestOR2;
        summary.bestSFF = bestSFF2;
        summary.bestTF = bestTF2;
        summary.expVar = expVal2;
        summary.correlation = correlation2;
        summary.thisROI = thisROI;
        summary.roiIdx = roiIdx;
        summary.mask = mask;
        
        
        %%vfs
        sfFac = 1;
        prefMaps_xy = [];
        prefMaps_xy(:,:,1)=summary.RF_Cx;
        prefMaps_xy(:,:,2)=summary.RF_Cy;
        if ID==9
            prefMaps_xy(:,:,2)=-prefMaps_xy(:,:,2);
        end
        summary.vfs=getVFS(prefMaps_xy, sfFac);
        
        %% test
        %     corrth=aparam.corrth;
        %     prefMaps_xy_i=[];
        %     for ii = 1:2
        %         prefMaps_xy_c = prefMaps_xy(:,:,ii);
        %         prefMaps_xy_c(summary.correlation<corrth)=NaN;
        %         prefMaps_xy_i(:,:,ii)=prefMaps_xy_c;
        %     end
        %     prefMaps_xy_i = interpNanImages(prefMaps_xy_i);
        %     vfs_c=getVFS(prefMaps_xy_i, sfFac);
        %     subplot(121);imagesc(summary.vfs);
        %     subplot(122); imagesc(vfs_c); hold on; contour(summary.correlation<corrth, [.5 .5],'color','k');
        
        
        save([encodingSavePrefix '_summary'],'summary');
        
    else
        load([encodingSavePrefix '_summary'],'summary','summary_adj','fvX','fvY');
        encodingSaveName = [encodingSavePrefix '_roiIdx' num2str(roiIdx(1)) '.mat'];
        load(encodingSaveName, 'RF_insilico');
    end
    
%     %% adjust Cx and Cy, interpolate NANs
%     [fvY,fvX] = getFoveaPix(ID, rescaleFac);
%     summary_adj = summary;
%     summary_adj.RF_Cx = interpNanImages(summary.RF_Cx - summary.RF_Cx(fvY,fvX));
%     summary_adj.RF_Cy = interpNanImages((summary.RF_Cy - summary.RF_Cy(fvY,fvX)));
%     %summary_adj.RF_Cy = -summary_adj.RF_Cy;
%     %summary_adj.vfs = -summary_adj.vfs;
%     if ID==9
%         summary_adj.RF_Cx = -summary_adj.RF_Cx;
%     end
%     summary_adj.RF_sigma = interpNanImages(summary_adj.RF_sigma);
%     summary_adj.RF_mean = (summary.RF_mean);
%     summary_adj.bestSF = interpNanImages(summary_adj.bestSF);
%     summary_adj.bestOR = interpNanImages(summary_adj.bestOR);
%     summary_adj.correlation = interpNanImages(summary_adj.correlation);
%     summary_adj.expVar = interpNanImages(summary_adj.expVar);
%     
%     stimXaxis_ori = RF_insilico.noiseRF.xaxis;
%     stimYaxis_ori = RF_insilico.noiseRF.yaxis;
%     
%     save([encodingSavePrefix '_summary'],'summary_adj','stimXaxis_ori','stimYaxis_ori',...
%         'fvX','fvY','-append');
    
    
    %% summary figure
%     [sumFig, sumAxes]=showSummaryFig(summary, aparam.flipLR);
%     set(sumFig,'position',[0 0 1900 1400]);
%     set(sumAxes(2),'xlim',[min(X) max(X)]);
%     set(sumAxes(2),'ylim',[min(Y) max(Y)]);
%     % set(sumFig,'position',[0 0 1900 1000]);
%     % set(sumAxes(2),'clim', [-8 8]);
%     % set(sumAxes(3),'clim', [-10 10]);
%     savePaperFigure(sumFig,[encodingSavePrefix '_summary']);
%     
%     %summary_adj.mask = summary.mask .* (summary_adj.correlation>corr_th);
%     [sumFig, sumAxes]=showSummaryFig(summary_adj, aparam.flipLR);
%     set(sumFig,'position',[0 0 1900 1400]);
%     set(sumAxes(2),'xlim',[min(X) max(X)]);
%     set(sumAxes(2),'ylim',[min(Y) max(Y)]);
%     % set(sumFig,'position',[0 0 1900 1000]);
%     set(sumAxes(2),'clim', aparam.showXrange);%[-7 1]);
%     set(sumAxes(3),'clim', aparam.showYrange);%[-7 7]);
%     savePaperFigure(sumFig,[encodingSavePrefix '_summary_adj']);
    
    [f_comp, signMap, signBorder, CyBorder, newmask] = ...
        showCompositeMap(summary_adj, aparam.corrth, aparam.showXrange, ...
        aparam.showYrange, rescaleFac, aparam.flipLR, aparam.brainPix);

    savePaperFigure(f_comp,[num2str(ID) '_compositeMap'], 'w');

    
    %% show mRFs on maps of preferred position
%     stimXaxis = stimXaxis_ori - summary.RF_Cx(fvY,fvX);
%     stimYaxis = (stimYaxis_ori - summary.RF_Cy(fvY,fvX));
%     for ib = 1:numel(aparam.brainPix)
%         [f_panel, f_location] = showRFpanels(summary_adj, aparam.brainPix(ib).brain_x, ...
%             aparam.brainPix(ib).brain_y, ...
%             stimXaxis, stimYaxis, aparam.showXrange, aparam.showYrange, rescaleFac,...
%             aparam.flipLR);
%         %savePaperFigure(f_panel,[encodingSavePrefix '_mRFs']);
%         savePaperFigure(f_location,[num2str(ID) '_mRFlocs_pwg' num2str(ib)], 'w');
%     end
    
    
    close all
end

%% pixel position on Brain
% load('\\ad.monash.edu\home\User006\dshi0006\Documents\MATLAB\2023ImagingPaper\ephys2Image_CJ231_pen1.mat');
% subplot(121);imagesc(baseImage)
% hold on
% plot(10*brain_x(2),10*brain_y(2),'rs');
% axis equal tight off
% colormap(gray);
% subplot(122);imagesc(inputImage_registered)
% axis equal tight off
% colormap(gray);
% savePaperFigure(gcf, ['brainImage_ephys2Image_CJ231_pen1']);

