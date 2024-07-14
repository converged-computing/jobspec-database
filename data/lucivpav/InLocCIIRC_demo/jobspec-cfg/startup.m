addpath('functions/ht_pnp_function');
addpath('functions/at_netvlad_function');
addpath('functions/utils');
addpath('functions/wustl_function');
addpath('functions/relja_matlab');
addpath('functions/relja_matlab/matconvnet/');
addpath('functions/netvlad/');
addpath('functions/inpaint_nans');
addpath('functions/permn');
addpath('functions/InLocCIIRC_utils/projectMesh');
addpath('functions/InLocCIIRC_utils/rotationMatrix');
addpath('functions/InLocCIIRC_utils/buildCutoutName');
addpath('functions/InLocCIIRC_utils/mkdirIfNonExistent');
addpath('functions/InLocCIIRC_utils/rotationDistance');
addpath('functions/InLocCIIRC_utils/P_to_str');
addpath('functions/InLocCIIRC_utils/at_netvlad_function');
addpath('functions/InLocCIIRC_utils/environment');
addpath('functions/InLocCIIRC_utils/load_CIIRC_transformation');
addpath('functions/InLocCIIRC_utils/params');
addpath('functions/InLocCIIRC_utils/loadPoseFromInLocCIIRC_demo');
addpath('functions/InLocCIIRC_utils/multiCameraPose');
addpath('functions/InLocCIIRC_utils/getPosesFromHoloLens');
addpath('functions/InLocCIIRC_utils/buildK');
addpath('functions/InLocCIIRC_utils/queryNameToQueryId');

env = environment();
if strcmp(env, 'ciirc')
    addpath('/home/lucivpav/NetVLAD');
elseif strcmp(env, 'cmp')
    addpath('/datagrid/personal/lucivpav/NetVLAD');
elseif strcmp(env, 'laptop')
    addpath('/Volumes/GoogleDrive/Můj disk/ARTwin/InLocCIIRC_dataset/NetVLAD');
end

addpath('functions/yael_matlab_linux64_v438');
run('functions/vlfeat/toolbox/vl_setup.m');
run('functions/matconvnet/matlab/vl_setupnn.m');
