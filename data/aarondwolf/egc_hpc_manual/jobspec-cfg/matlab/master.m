
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filename: 0_master.m
%
%
% Purpose: This is the master .m file for <PROJECT NAME>.
%
%
% Created by: adw54
% Created on: 7 Oct 2020
% Last updated on: 7 Oct 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%---------------------------------------------------------------------------
%
%% Section 1: Directories
%
%--------------------------------------------------------------------------

% Set-Up Working Directories
if getenv('username') == "adw54" & ispc
     root = "C:/Users/adw54/Pande Research Dropbox/Aaron Wolf/Grace/Documents/egc_hpc_manual"
else if getenv('USER') == "adw54" & isunix
        root = "/home/adw54/Documents/egc_hpc_manual"
    end
end

cd(root)

disp('Setup Done!')

%---------------------------------------------------------------------------
%
%% Section 2: Run sub-files
%
%--------------------------------------------------------------------------

run 'matlab/read.m'
