addpath('~/MITgcm/utils/matlab/')
doyz=1;
yzvars={'T','U','V','Psim','S','W','T','T','U','V','Psim','S','W'};
yzwdirs=[0,1,1,1,0,1,0,0,1,1,1,0,1]; % with direction
yzpltrng=[-100,-100,-100,-100,-100,-100,0,2072160,2072160,2072160,2072160,2072160,2072160];
nyzvar=length(yzvars);
skipyzvar={};

doyx=1;
yxvars={'T','U','S','V','T','U','S','V','T','U','S','V'};
yxwdirs=[0,1,0,1,0,1,0,1,0,1,0,1];
yxpltrng=[-100,-100,-100,-100,2072160,2072160,2072160,2072160,2072160,2072160,2072160,2072160];
yxshowlev=[0,Inf,0,Inf,0,Inf,0,Inf,4,4,4,4]; % 0: water-ice interface % -1: 2d % Inf: max value
nyxvar=length(yxvars);
skipyxvar={};

doyt=1;
ytvars={'T','U','Psim','S','SHICE_fwFlux','T','U','Psim','S','SHICE_fwFlux'};
ytwdirs=[0,1,1,0,1,0,1,1,0,1];
ytpltrng=[0,0,0,0,0,Inf,Inf,Inf,Inf,Inf];
ytshowlev=[0,Inf,Inf,0,-1,0,Inf,Inf,0,-1]; % 0: water-ice interface % -1: 2d % Inf: max value
nytvar=length(ytvars);
skipytvar={'SHICE_fwFlux'};

appendix='_CASENAME_';
eachiter=_EACHITER_;
currentiter=_CURRENTITER_;
dt=_DT_;
heredir=pwd;
disp(heredir)

if currentiter<=0
    nytvar=floor(nytvar/2);
end

%% loop over all variables
if doyz
disp('doing yz contour now')
init=0;
for ivar=1:nyzvar
if ~any(strcmp(skipyzvar,char(yzvars(ivar))))
    if currentiter>=0
    if yzpltrng(ivar)>=0 && yzpltrng(ivar)<=currentiter+eachiter
        iter=mod(yzpltrng(ivar),eachiter);
        iter=yzpltrng(ivar)-iter-(iter==0 & yzpltrng(ivar)~=0)*eachiter;
        disp([heredir,'/run',sprintf('%d',iter)])

        cd([heredir,'/run',sprintf('%d',iter)]);
    else
        cd([heredir,'/run',sprintf('%d',currentiter)]);
        if yzpltrng(ivar)>=0
            yzpltrng(ivar)=Inf;
        end
    end
    else
        if yzpltrng(ivar)>=0
            yzpltrng(ivar)=Inf;
        end
    end

    % load grid
    if ~init
        init=1;
        XC=squeeze(mean(rdmds('XC',NaN),1));
        YC=squeeze(mean(rdmds('YC',NaN),1));
        RC=squeeze(mean(rdmds('RC',NaN),1));
        DRF=squeeze(mean(rdmds('DRF',NaN),1))';
        hFacC=rdmds('hFacC',NaN);
        mask=hFacC.*0+1;
        mask(hFacC<0.999)=NaN;
        hFacC=squeeze(mean(hFacC,1));
        maskmean=squeeze(mean(mask,1));
        wgt=cosd(YC')/mean(cosd(YC));
    end

    % read data
    var=char(yzvars(ivar));
    disp(['doing ',var])
    if strcmp(var,'Psim')
        if yzpltrng(ivar)<0
            Vardata=rdmds('V',NaN);
        else
            [Vardata,yzpltrng(ivar)]=rdmds('V',yzpltrng(ivar));
        end
        nd=ndims(Vardata);
        if nd==4
            yzpltrng(ivar)=max(-size(Vardata,4)+1,yzpltrng(ivar));
            Varmean=squeeze(mean(mean(Vardata(:,:,:,end+yzpltrng(ivar):end),1),nd));
        else
            Varmean=squeeze(mean(Vardata,1));
        end
        Varmean=cumsum(Varmean.*hFacC.*DRF,2,'reverse').*maskmean;
        
    else
        if yzpltrng(ivar)<0
            Vardata=rdmds(var,NaN);
        else
            [Vardata,yzpltrng(ivar)]=rdmds(var,yzpltrng(ivar));
        end

        nd=ndims(Vardata);
        if nd==4 % have time dimension
            yzpltrng(ivar)=max(-size(Vardata,4)+1,yzpltrng(ivar));
            Varmean=squeeze(mean(mean(Vardata(:,:,:,end+yzpltrng(ivar):end),1),nd)).*maskmean;
        else
            Varmean=squeeze(mean(Vardata,1)).*maskmean;
        end

    end
    vmax=nanmax(Varmean(:));
    vmin=nanmin(Varmean(:));
    vmean=round(nanmean(Varmean(:)),3,'significant');
    fprintf('max:%.3g, min:%.3g\n\n',vmax,vmin) 

    % plot
    f=figure('visible','off');
    if ~yzwdirs(ivar)
        reldev=abs((vmax-vmin)/(vmax+vmin));
        sigdig=int8(ceil(log10(1/reldev)))+1;
        vmax_rnd=round(vmax,sigdig,'significant');
        vmin_rnd=round(vmin,sigdig,'significant');
        conts=[vmin_rnd:(vmax_rnd-vmin_rnd)/20:vmax_rnd];
        conts=conts-vmean;
        if isempty(conts)
            conts=0;
        end
    else
        vmax_rnd=round(nanmax(abs(Varmean(:))),2,'significant');
        conts=[-vmax_rnd:vmax_rnd/10:vmax_rnd];
        if isempty(conts)
            conts=0;
        end
    end

    contourf(YC,RC/1e3, Varmean'-vmean*(~yzwdirs(ivar)),[-1e30,conts,1e30],'LineColor','none')
    if length(conts)~=1
        caxis([conts(1) conts(end)])
    end
    colormap(jet)
    colorbar
    set(gca,'FontSize',16)
    xlabel('lat')
    xticks([-90:30:90])
    ylabel('H (km)')
    if ~yzwdirs(ivar)
        title([var,sprintf(' (mean: %.3g)', vmean)])
    else
        title(var)
    end

    saveas(f,[var,sprintf('%d',yzpltrng(ivar)),'_yz_',appendix,'.png'])

    if ~yzwdirs(ivar)
        f=figure('visible','off');
        Varanom=Varmean-nanmean(Varmean.*wgt,1);
        amp=nanmax(abs(Varanom(:)));
        fprintf('amp=%.3g\n\n',amp)
        amp=max(amp,1e-14);
        conts=[-amp:amp/10:amp];
        contourf(YC,RC/1e3, Varanom',[-1e30,conts,1e30],'LineColor','none')
        if length(conts)~=1
            caxis([conts(1) conts(end)])
        end
        colormap(jet)
        colorbar
        set(gca,'FontSize',16)
        xlabel('lat')
        xticks([-90:30:90])
        ylabel('H (km)')
        title(var)
        saveas(f,[var,sprintf('%d',yzpltrng(ivar)),'_yzanom_',appendix,'.png'])
    end
end
end
end

%% y-x plot
if doyx
disp('doing yx contour now')
init=0;
for ivar=1:nyxvar
if ~any(strcmp(skipyxvar,char(yxvars(ivar))))
    var=char(yxvars(ivar));
    disp(['doing ',var])
    if currentiter>=0
    if yxpltrng(ivar)>=0 && yxpltrng(ivar)<=currentiter+eachiter
        iter=mod(yxpltrng(ivar),eachiter);
        iter=yxpltrng(ivar)-iter-(iter==0 & yxpltrng(ivar)~=0)*eachiter;
        disp([heredir,'/run',sprintf('%d',iter)])

        cd([heredir,'/run',sprintf('%d',iter)]);
    else
        cd([heredir,'/run',sprintf('%d',currentiter)]);
        if yxpltrng(ivar)>=0
            yxpltrng(ivar)=Inf;
        end
    end
    else
        if yxpltrng(ivar)>=0
            yxpltrng(ivar)=Inf;
        end
    end

    % load grid
    if ~init
        init=1;
        XC=squeeze(rdmds('XC',NaN));
        YC=squeeze(rdmds('YC',NaN));
        RC=squeeze(rdmds('RC',NaN));
        DRF=rdmds('DRF',NaN);
        hFacC=rdmds('hFacC',NaN);
        mask=hFacC.*0+1;
        mask(hFacC<0.999)=NaN;
    end

    % read data
    if yxpltrng(ivar)<0
        Vardata=rdmds(var,NaN);
        nd=ndims(Vardata);
        Vardata=mean(Vardata,nd);
    else
        [Vardata,yxpltrng(ivar)]=rdmds(var,yxpltrng(ivar));
    end
    Vardata=Vardata.*mask;

    % select slice
    if yxshowlev(ivar)>=0 % 3d
        if yxshowlev(ivar)~=0  
            if isinf(yxshowlev(ivar))
                maxinxy=squeeze(nanmax(nanmax(squeeze(Vardata),[],1),[],2));
                [~,ilmax]=nanmax(maxinxy);
                yxshowlev(ivar)=ilmax;
            end
            Varslice=squeeze(Vardata(:,:,yxshowlev(ivar)));
        else % follow interface
            Varslice=squeeze(Vardata(:,:,1)).*0;
            for ix=1:size(Vardata,1)
            for iy=2:size(Vardata,2)-1
                tmp=find(~isnan(squeeze(mask(ix,iy,:))),1);
                Varslice(ix,iy)=Vardata(ix,iy,tmp);
            end
            end
        end

    else % 2d
        Varslice=squeeze(Vardata);
    end

    % min & max
    vmax=nanmax(Varslice(:));
    vmin=nanmin(Varslice(:));
    vmean=round(nanmean(Varslice(:)),3,'significant');
    fprintf('max:%.3g, min:%.3g\n\n',vmax,vmin) 

    % plot
    f=figure('visible','off');
    if ~yxwdirs(ivar)
        reldev=abs((vmax-vmin)/(vmax+vmin));
        sigdig=int8(ceil(log10(1/reldev)))+1;
        vmax_rnd=round(vmax,sigdig,'significant');
        vmin_rnd=round(vmin,sigdig,'significant');
        conts=[vmin_rnd:(vmax_rnd-vmin_rnd)/20:vmax_rnd];
        conts=conts-vmean;
        if isempty(conts)
            conts=0;
        end
    else
        vmax_rnd=round(nanmax(abs(Varslice(:))),2,'significant');
        conts=[-vmax_rnd:vmax_rnd/10:vmax_rnd];
        if isempty(conts)
            conts=0;
        end
    end
    contourf(XC, YC, Varslice-vmean*(~yxwdirs(ivar)),[-1e30,conts,1e30],'LineColor','none')
    disp('done contour')
    if length(conts)~=1
    caxis([conts(1) conts(end)])
    end
    colormap(jet)
    colorbar
    disp('done colorbar')
    set(gca,'FontSize',16)
    xlabel('lon')
    yticks([-90:30:90])
    ylabel('lat')
    disp('done label')
    if ~yxwdirs(ivar)
        title([var,sprintf(' (mean: %.3g)', vmean)])
    else
        title(var)
    end
    disp('done title')

    saveas(f,[var,sprintf('%d',yxpltrng(ivar)),'_yx_z',sprintf('%d',yxshowlev(ivar)),'_',appendix,'.png'])
    disp('done save yx plot')

    if ~yxwdirs(ivar)
        f=figure('visible','off');
        Varanom=Varslice-nanmean(Varslice,1);
        amp=nanmax(abs(Varanom(:)));
        fprintf('amp=%.3g\n\n',amp)
        amp=max(amp,1e-14);
        conts=[-amp:amp/10:amp];
        contourf(XC, YC, Varanom,[-1e30,conts,1e30],'LineColor','none')
        if length(conts)~=1
        caxis([conts(1) conts(end)])
        end
        colormap(jet)
        colorbar
        set(gca,'FontSize',16)
        xlabel('lon')
        yticks([-90:30:90])
        ylabel('lat')
        title(var)
        saveas(f,[var,sprintf('%d',yxpltrng(ivar)),'_yxanom_z',sprintf('%d',yxshowlev(ivar)),'_',appendix,'.png'])
    end

end
end
end


%% y-t plot
if doyt
disp('doing yt contour now')
init=0;
for ivar=1:nytvar
if ~any(strcmp(skipytvar,char(ytvars(ivar))))
    var=char(ytvars(ivar));
    disp(['doing ',var])
    if currentiter>=0
        ytpltrng(ivar)=min(ytpltrng(ivar),currentiter/eachiter);
        cd([heredir,'/run',sprintf('%d',eachiter*ytpltrng(ivar))]);
    else
        ytpltrng(ivar)=currentiter;
    end
    % load grid
    if ~init
        init=1;
        XC=squeeze(mean(rdmds('XC',NaN),1));
        YC=squeeze(mean(rdmds('YC',NaN),1));
        RC=squeeze(mean(rdmds('RC',NaN),1));
        DRF=mean(rdmds('DRF',NaN),1);
        hFacC=mean(rdmds('hFacC',NaN),1);
        mask=hFacC.*0+1;
        mask(hFacC<0.999)=NaN;
        maskmean=squeeze(mean(mask,1));
        wgt=cosd(YC')/mean(cosd(YC));
    end

    % read data
    if strcmp(var,'Psim')
        [Vardata,its]=rdmds('V',NaN);
        Vardata=mean(Vardata.*mask,1);
        Vardata=cumsum(Vardata.*hFacC.*DRF,3,'reverse');
    else
        [Vardata,its]=rdmds(var,NaN);
        if ytshowlev(ivar)>=0
            Vardata=mean(Vardata.*mask,1);
        else
            Vardata=mean(Vardata,1);
        end
    end

    % select slice
    if ytshowlev(ivar)>=0 % 3d
        if ytshowlev(ivar)~=0  
            if isinf(ytshowlev(ivar))
                maxiny=nanmax(squeeze(mean(Vardata,4)),[],1);
                [~,ilmax]=nanmax(maxiny);
                ytshowlev(ivar)=ilmax;
            end
            Varslice=squeeze(mean(Vardata(:,:,ytshowlev(ivar),:),3));
        else % follow interface
            Varslice=squeeze(Vardata(1,:,1,:)).*0;
            for iy=2:length(YC)-1
                tmp=find(~isnan(squeeze(maskmean(iy,:))),1);
                Varslice(iy,:)=Vardata(1,iy,tmp,:);
            end
        end

    else % 2d
        Varslice=squeeze(Vardata);
    end

    % min & max
    vmax=nanmax(Varslice(:));
    vmin=nanmin(Varslice(:));
    Varslice2=Varslice(:,end-floor(length(its)/2));
    vmean=round(nanmean(Varslice2(:)),3,'significant');
    fprintf('max:%.3g, min:%.3g\n\n',vmax,vmin) 
    
    % plot
    f=figure('visible','off');
    if ~ytwdirs(ivar)
        reldev=abs((vmax-vmin)/(vmax+vmin));
        sigdig=int8(ceil(log10(1/reldev)))+1;
        vmax_rnd=round(vmax,sigdig,'significant');
        vmin_rnd=round(vmin,sigdig,'significant');
        conts=[vmin_rnd:(vmax_rnd-vmin_rnd)/20:vmax_rnd];
        conts=conts-vmean;
        if isempty(conts)
            conts=0;
        end
    else
        vmax_rnd=round(nanmax(abs(Varslice(:))),2,'significant');
        conts=[-vmax_rnd:vmax_rnd/10:vmax_rnd];
        if isempty(conts)
            conts=0;
        end
    end

    contourf(YC, its*dt/365/86400, Varslice'-vmean*(~ytwdirs(ivar)),[-1e30,conts,1e30],'LineColor','none')
    if length(conts)~=1
        caxis([conts(1) conts(end)])
    end
    colormap(jet)
    colorbar
    set(gca,'FontSize',16)
    xlabel('lat')
    xticks([-90:30:90])
    ylabel('time (yr)')
    if ~ytwdirs(ivar)
        title([var,sprintf(' (mean: %.3g)', vmean)])
    else
        title(var)
    end

    saveas(f,[var,sprintf('%d',ytpltrng(ivar)*eachiter),'_yt_z',sprintf('%d',ytshowlev(ivar)),'_',appendix,'.png'])

    if ~ytwdirs(ivar)
        f=figure('visible','off');
        Varanom=Varslice-nanmean(Varslice.*wgt,1);
        Varanom2=Varanom(:,floor(length(its)/2):end);
        amp=nanmax(abs(Varanom2(:)));
        fprintf('amp=%.3g\n\n',amp)
        amp=max(amp,1e-14);
        conts=[-amp:amp/10:amp];
        contourf(YC, its*dt/365/86400, Varanom',[-1e30,conts,1e30],'LineColor','none')
        if length(conts)~=1
            caxis([conts(1) conts(end)])
        end
        colormap(jet)
        colorbar
        set(gca,'FontSize',16)
        xlabel('lat')
        xticks([-90:30:90])
        ylabel('time (yr)')
        title(var)
        saveas(f,[var,sprintf('%d',ytpltrng(ivar)*eachiter),'_ytanom_z',sprintf('%d',ytshowlev(ivar)),'_',appendix,'.png'])
    end

end
end
end
