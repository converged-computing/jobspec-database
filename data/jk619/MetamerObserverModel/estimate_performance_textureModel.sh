#!/bin/sh
#SBATCH --job-name=performance_estimation_texture
#SBATCH -a 0-19  # run this script as 2 jobs with SLURM_ARRAY_TASK_ID = 0 and 1. Add more numbers for more jobs!
#SBATCH --nodes=1 # nodes per job
#SBATCH --cpus-per-task=16 #~2 days to run PRFs
#SBATCH --mem=64gb # More memory you request the less priority you get
#SBATCH --time=50:00:00 # Max request to be safe...
#SBATCH --output=/scratch/jk7127/logs/performance_est_out_ses_%x-%a.txt # Define output log location
#SBATCH --error=/scratch/jk7127/logs/performance_est_err_ses_%x-%a.txt # and the error logs for when it inevitably crashes
#SBATCH --mail-user=jk7127@nyu.edu #email
#SBATCH --mail-type=END #email me when it crashes or better, ends

all_subjects=(azulejos bike boats gnarled graffiti grooming highway ivy leaves lettuce llama nyc palm portrait quad rocks terraces tiles treetop troop)

module load matlab/2020b

# this variable tells us the job number:
jobnum=$SLURM_ARRAY_TASK_ID
if [ $jobnum -ge ${#all_subjects[@]} ]
then echo "Invalid subject id: $jobnum"
     exit 1
fi
sub=${all_subjects[$jobnum]}
wscale=$1;
echo $sub
echo $wscale
# startup matlab...
matlab -nodesktop -nodisplay -nosplash <<EOF


clc
clear
[~,whoami]      = system('whoami'); whoami = whoami(1:end-1)

scenario        = 'bar'
mainPath        = sprintf('/scratch/%s/MetamerObserverModel/',whoami)
myimage         = '$sub';
windowPath      = fullfile(mainPath,'windows');

switch scenario
    case 'bar'

        targetPath      = fullfile(mainPath,'target_images_bar');
        metPath         = fullfile(mainPath,'metamers_bar');
        resPath         = fullfile(mainPath,'results_bar');

    case 'crop'

        targetPath      = fullfile(mainPath,'target_images_crop');
        metPath         = fullfile(mainPath,'metamers_crop');
        resPath         = fullfile(mainPath,'results_crop');

    case 'allimg'

        targetPath      = fullfile(mainPath,'target_images');
        metPath         = fullfile(mainPath,'metamers');
        resPath         = fullfile(mainPath,'results');
end


mkdir(resPath);


maskPath        = fullfile(mainPath,'mask');
codePath        = fullfile(mainPath,'functions');

addpath(genpath(codePath));

met_types       = {'metamers_energy_met';'metamers_energy_ref'};
w.scaling	    = str2num('$wscale');
w.aspect        = 2;
w.size          = 2048;

myimage
analysis_type   = {'tar_vs_met';'met_vs_met'};

%% load windows & maskInds
load(sprintf('%s/window_2048x2048_s=%.2f_a=%i.mat',windowPath,w.scaling,w.aspect),'opts')
load(sprintf('%s/maskInds.mat',maskPath))


%% load params for target image

target = load(sprintf('%s/params_%s_s=%.2f_a=%i.mat',targetPath,myimage,w.scaling,w.aspect));
%% find params for metamers with different scalings (they are both in metamers_energy_met and metamers_energy_ref directories)
allscalings = {};

for mer = 1 : length(met_types)
    scalings = dir(sprintf('%s/%s/energy_model/%s/*scaling*',metPath,met_types{mer},myimage));

    for s = 1 : length(scalings)

        allscalings  = cat(1,allscalings,[scalings(s).folder filesep scalings(s).name]);

    end

end

disp('done')
% remove 0.27 duplicate from both
toremove = find(contains(allscalings,'0.27'));
allscalings(toremove(2)) = [];
allscalings
%%
clear results
results = table;
for a = 1 : length(analysis_type)

    % pick which comparison to do (met vs. met or target vs. met)
    myanalysis = analysis_type{a};

    for s = 1 : length(allscalings)

        myparams = dir([allscalings{s} filesep sprintf('params*%.2f*.mat',w.scaling)]);
        mystr = strfind(allscalings{s},'-');
        myscaling = str2double(allscalings{s}(mystr+1:end));

        disp(myscaling)
        % prep Corey's structure for analysis (3 dim should be stats to
        % compare)

        if strcmp('tar_vs_met',myanalysis)
            disp('tar_vs_met')
            clear allResp
            allResp{1}(1,1,1,:,:) = target.Out_masked;
            metamer = load([myparams(1).folder filesep myparams(1).name]);
            allResp{1}(1,1,2,:,:) = metamer.Out_masked;

		
        elseif  strcmp('met_vs_met',myanalysis)
            disp('met_vs_met')
            clear allResp
            metamer1 = load([myparams(1).folder filesep myparams(1).name]);
            metamer2 = load([myparams(2).folder filesep myparams(2).name]);

	    allResp{1}(1,1,1,:,:) = metamer1.Out_masked;
        allResp{1}(1,1,2,:,:) = metamer2.Out_masked;

        end




        %%
        inParams.inds           = maskInds; % orig inds or maskInds
        inParams.scale          = opts.windows.scale;
        inParams.normVanHat     = 1;
        inParams.normBroderick  = 0;
        inParams.normRFParams   = 0;
        inParams.scaleNoise     = 0;
        inParams.scaleWeights   = 0;    % 0 or maskScaleFile
        inParams.singlePool     = 0;
        inParams.paramGroups    = 'notPixelsAndMags'; %'notPixels', %'notPixelsAndMags',
        inParams.thresh         = 0;    %0.01
        inParams.paramNorm      = 3;    % 1 = within params, across RFs; 2 = within RFs, across params; 3 = across params/RFs; 4 = none
        inParams.intN           = 0;    % amount of internal noise
        inParams.lateN          = 0;    % amount of late noise
        inParams.nTrials        = 2500;  % number of trials per condition; 2500 final version
        inParams.windowInd      = opts.indices;
        inParams.broderickTask  = 1;
        inParams.scenario       = scenario;

	if w.scaling == 0.21
            midN    = [0.0056    0.0028    0.0014]
        elseif w.scaling == 0.46
            midN    = [0.0109    0.0055    0.0027]
        elseif w.scaling == 0.84
            midN    = [0.0192    0.0096    0.0048]
        end

	inParams.midN = midN

        if inParams.broderickTask
            makefolder = sprintf('%s/%s/s=%.2f_a=%i_vanHat=%i_brod=%i_paramNom=%i_LR_2AFC/',resPath,myimage,w.scaling,w.aspect,inParams.normVanHat,inParams.normBroderick,inParams.paramNorm);
        else
            makefolder = sprintf('%s/%s/s=%.2f_a=%i_vanHat=%i_paramNom=%i_ABX/',resPath,myimage,w.scaling,w.aspect,inParams.normVanHat,inParams.paramNorm);
        end

        %if ~exist(sprintf('%s/resultsTextureModel.csv',makefolder))

	    for iP = 1 : length(inParams.midN)
                    
                    
                    [sampPerf_L1,sampPerf_L2,sampPerf_L4,dists] = getPerf2afc(allResp,inParams,iP);
                    
                    tmp_table(iP).met_scaling = myscaling;
                    tmp_table(iP).mean_L1 = nanmean(sampPerf_L1);
                    tmp_table(iP).mean_L2 = nanmean(sampPerf_L2);
                    tmp_table(iP).mean_L4 = nanmean(sampPerf_L4);
                    
%                   tmp_table(iP).std = std(sampPerf)./sqrt(size(sampPerf,1));
                    tmp_table(iP).noise = inParams.midN(iP);
                    tmp_table(iP).image = {myimage};
                    tmp_table(iP).type = {myanalysis};
                    tmp_table(iP).window_scaling = w.scaling;
              end

            for iP = 1 : length(inParams.midN)

                tmp_table(iP).window_scaling = w.scaling;
                results = cat(1,results,struct2table(tmp_table(iP)));

            end
        %else
        %    disp('done')
        %end
    end
end

disp(makefolder)
if ~exist(makefolder,'dir')
    mkdir(makefolder);
end

if ~isempty(results)
    writetable(results,sprintf('%s/resultsTextureModel.csv',makefolder))
else
    error('empty table')
end



EOF

exit 0


