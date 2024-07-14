#!/bin/bash
#SBATCH --job-name="PyNets"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=56G
#SBATCH --time=24:00:00

set -x
set -e

bl2bids

# usage: pynets [-h] -id A subject id or other unique identifier
#               [A subject id or other unique identifier ...]
#               [-func Path to input functional file required for functional connectomes) [Path to input functional file (required for functional connectomes) ...]]
#               [-dwi Path to diffusion-weighted imaging data file (required for dmri connectomes) [Path to diffusion-weighted imaging data file (required for dmri connectomes) ...]]
#               [-bval Path to b-values file (required for dmri connectomes) [Path to b-values file (required for dmri connectomes) ...]]
#               [-bvec Path to b-vectors file (required for dmri connectomes) [Path to b-vectors file (required for dmri connectomes) ...]]
#               [-anat Path to a skull-stripped anatomical Nifti1Image [Path to a skull-stripped anatomical Nifti1Image ...]]
#               [-m Path to a T1w brain mask image (if available) in native anatomical space [Path to a T1w brain mask image (if available) in native anatomical space ...]]
#               [-conf Confound regressor file (.tsv/.csv format) [Confound regressor file (.tsv/.csv format) ...]]
#               [-g Path to graph file input. [Path to graph file input. ...]]
#               [-roi Path to binarized Region-of-Interest (ROI) Nifti1Image in template MNI space. [Path to binarized Region-of-Interest (ROI) Nifti1Image in template MNI space. ...]]
#               [-ref Atlas reference file path]
#               [-way Path to binarized Nifti1Image to constrain tractography [Path to binarized Nifti1Image to constrain tractography ...]]
#               [-mod Connectivity estimation/reconstruction method [Connectivity estimation/reconstruction method ...]]
#               [-a Atlas [Atlas ...]]
#               [-ns Spherical centroid node size [Spherical centroid node size ...]]
#               [-thr Graph threshold]
#               [-min_thr Multi-thresholding minimum threshold]
#               [-max_thr Multi-thresholding maximum threshold]
#               [-step_thr Multi-thresholding step size]
#               [-sm Smoothing value (mm fwhm) [Smoothing value (mm fwhm) ...]]
#               [-hp High-pass filter (Hz) [High-pass filter (Hz) ...]]
#               [-es Node extraction strategy [Node extraction strategy ...]]
#               [-k Number of k clusters [Number of k clusters ...]]
#               [-ct Clustering type [Clustering type ...]]
#               [-cm Cluster mask [Cluster mask ...]]
#               [-ml Minimum fiber length for tracking [Minimum fiber length for tracking ...]]
#               [-em Error margin [Error margin ...]]
#               [-dg Direction getter [Direction getter ...]]
#               [-norm Normalization strategy for resulting graph(s)] [-bin]
#               [-dt] [-mst] [-p Pruning Strategy] [-df]
#               [-mplx Perform various levels of multiplex graph analysis (only if both structural and diffusion connectometry is run simultaneously.]
#               [-embed] [-spheres]
#               [-n Resting-state network [Resting-state network ...]]
#               [-vox {1mm,2mm}] [-plt] [-pm Cores,memory]
#               [-plug Scheduler type] [-v] [-noclean] [-work Working directory]
#               [--version]
#               output_dir
#
# PyNets: A Reproducible Workflow for Structural and Functional Connectome
# Ensemble Learning
#
# positional arguments:
#   output_dir            The directory to store pynets derivatives.
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -id A subject id or other unique identifier [A subject id or other unique identifier ...]
#                         An subject identifier OR list of subject identifiers,
#                         separated by space and of equivalent length to the
#                         list of input files indicated with the -func flag.
#                         This parameter must be an alphanumeric string and can
#                         be arbitrarily chosen. If functional and dmri
#                         connectomes are being generated simultaneously, then
#                         space-separated id's need to be repeated to match the
#                         total input file count.
#   -func Path to input functional file (required for functional connectomes) [Path to input functional file (required for functional connectomes) ...]
#                         Specify either a path to a preprocessed functional
#                         Nifti1Image in MNI152 space OR multiple space-
#                         separated paths to multiple preprocessed functional
#                         Nifti1Image files in MNI152 space and in .nii or
#                         .nii.gz format, OR the path to a text file containing
#                         a list of paths to subject files.
#   -dwi Path to diffusion-weighted imaging data file (required for dmri connectomes) [Path to diffusion-weighted imaging data file (required for dmri connectomes) ...]
#                         Specify either a path to a preprocessed dmri diffusion
#                         Nifti1Image in native diffusion space and in .nii or
#                         .nii.gz format OR multiple space-separated paths to
#                         multiple preprocessed dmri diffusion Nifti1Image files
#                         in native diffusion space and in .nii or .nii.gz
#                         format.
#   -bval Path to b-values file (required for dmri connectomes) [Path to b-values file (required for dmri connectomes) ...]
#                         Specify either a path to a b-values text file
#                         containing gradient shell values per diffusion
#                         direction OR multiple space-separated paths to
#                         multiple b-values text files in the order of
#                         accompanying b-vectors and dwi files.
#   -bvec Path to b-vectors file (required for dmri connectomes) [Path to b-vectors file (required for dmri connectomes) ...]
#                         Specify either a path to a b-vectors text file
#                         containing gradient directions (x,y,z) per diffusion
#                         direction OR multiple space-separated paths to
#                         multiple b-vectors text files in the order of
#                         accompanying b-values and dwi files.
#   -anat Path to a skull-stripped anatomical Nifti1Image [Path to a skull-stripped anatomical Nifti1Image ...]
#                         Required for dmri and/or functional connectomes.
#                         Multiple paths to multiple anatomical files should be
#                         specified by space in the order of accompanying
#                         functional and/or dmri files. If functional and dmri
#                         connectomes are both being generated simultaneously,
#                         then anatomical Nifti1Image file paths need to be
#                         repeated, but separated by comma.
#   -m Path to a T1w brain mask image (if available) in native anatomical space [Path to a T1w brain mask image (if available) in native anatomical space ...]
#                         File path to a T1w brain mask Nifti image (if
#                         available) in native anatomical space OR multiple file
#                         paths to multiple T1w brain mask Nifti images in the
#                         case of running multiple participants, in which case
#                         paths should be separated by a space. If no brain mask
#                         is supplied, the template mask will be used (see
#                         runconfig.yaml).
#   -conf Confound regressor file (.tsv/.csv format) [Confound regressor file (.tsv/.csv format) ...]
#                         Optionally specify a path to a confound regressor file
#                         to reduce noise in the time-series estimation for the
#                         graph. This can also be a list of paths in the case of
#                         running multiplesubjects, which requires separation by
#                         space and of equivalent length to the list of input
#                         files indicated with the -func flag.
#   -g Path to graph file input. [Path to graph file input. ...]
#                         In either .txt, .npy, .graphml, .csv, .ssv, .tsv, or
#                         .gpickle format. This skips fMRI and dMRI graph
#                         estimation workflows and begins at the thresholding
#                         and graph analysis stage. Multiple graph files
#                         corresponding to multiple subject ID's should be
#                         separated by space, and multiple graph files
#                         corresponding to the same subject ID should be
#                         separated by comma. If the `-g` flag is used, then the
#                         `-id` flag must also be used. Consider also including
#                         `-thr` flag to activate thresholding only or the `-p`
#                         and `-norm` flags if graph defragementation or
#                         normalization is desired. The `-mod` flag can be used
#                         for additional provenance/file-naming.
#   -roi Path to binarized Region-of-Interest (ROI) Nifti1Image in template MNI space. [Path to binarized Region-of-Interest (ROI) Nifti1Image in template MNI space. ...]
#                         Optionally specify a binarized ROI mask and retain
#                         only those nodes of a parcellation contained within
#                         that mask for connectome estimation.
#   -ref Atlas reference file path
#                         Specify the path to the atlas reference .txt file that
#                         maps labels to intensities corresponding to the atlas
#                         parcellation file specified with the -a flag.
#   -way Path to binarized Nifti1Image to constrain tractography [Path to binarized Nifti1Image to constrain tractography ...]
#                         Optionally specify a binarized ROI mask in MNI-space
#                         toconstrain tractography in the case of dmri
#                         connectome estimation.
#   -mod Connectivity estimation/reconstruction method [Connectivity estimation/reconstruction method ...]
#                         (metaparameter): Specify connectivity estimation
#                         model. For fMRI, possible models include: corr for
#                         correlation, cov for covariance, sps for precision
#                         covariance, partcorr for partial correlation. If skgmm
#                         is installed (https://github.com/skggm/skggm), then
#                         QuicGraphicalLasso, QuicGraphicalLassoCV,
#                         QuicGraphicalLassoEBIC, and
#                         AdaptiveQuicGraphicalLasso. For dMRI, current models
#                         include csa, csd, and sfm.
#   -a Atlas [Atlas ...]  (metaparameter): Specify an atlas name from nilearn or
#                         local (pynets) library, and/or specify a path to a
#                         custom parcellation/atlas Nifti1Image file in MNI
#                         space. Labels shouldbe spatially distinct across
#                         hemispheres and ordered with consecutive integers with
#                         a value of 0 as the background label.If specifying a
#                         list of paths to multiple parcellations, separate them
#                         by space. If you wish to iterate your pynets run over
#                         multiple atlases, separate them by space. Available
#                         nilearn atlases are: atlas_aal atlas_talairach_gyrus
#                         atlas_talairach_ba atlas_talairach_lobe
#                         atlas_harvard_oxford atlas_destrieux_2009 atlas_msdl
#                         coords_dosenbach_2010 coords_power_2011
#                         atlas_pauli_2017. Available local atlases are:
#                         destrieux2009_rois BrainnetomeAtlasFan2016
#                         VoxelwiseParcellationt0515kLeadDBS
#                         Juelichgmthr252mmEickhoff2005 CorticalAreaParcellation
#                         fromRestingStateCorrelationsGordon2014
#                         whole_brain_cluster_labels_PCA100
#                         AICHAreorderedJoliot2015
#                         HarvardOxfordThr252mmWholeBrainMakris2006
#                         VoxelwiseParcellationt058kLeadDBS MICCAI2012MultiAtlas
#                         LabelingWorkshopandChallengeNeuromorphometrics
#                         Hammers_mithAtlasn30r83Hammers2003Gousias2008
#                         AALTzourioMazoyer2002 DesikanKlein2012
#                         AAL2zourioMazoyer2002
#                         VoxelwiseParcellationt0435kLeadDBS AICHAJoliot2015
#                         whole_brain_cluster_labels_PCA200
#                         RandomParcellationsc05meanalll43Craddock2011
#   -ns Spherical centroid node size [Spherical centroid node size ...]
#                         (metaparameter): Optionally specify coordinate-based
#                         node radius size(s). Default is 4 mm for fMRI and 8mm
#                         for dMRI. If you wish to iterate the pipeline across
#                         multiple node sizes, separate the list by space (e.g.
#                         2 4 6).
#   -thr Graph threshold  Optionally specify a threshold indicating a proportion
#                         of weights to preserve in the graph. Default is no
#                         thresholding. If `-mst`, `-dt`, or `-df` flags are not
#                         included, than proportional thresholding will be
#                         performed
#   -min_thr Multi-thresholding minimum threshold
#                         (metaparameter): Minimum threshold for multi-
#                         thresholding.
#   -max_thr Multi-thresholding maximum threshold
#                         (metaparameter): Maximum threshold for multi-
#                         thresholding.
#   -step_thr Multi-thresholding step size
#                         (metaparameter): Threshold step value for multi-
#                         thresholding. Default is 0.01.
#   -sm Smoothing value (mm fwhm) [Smoothing value (mm fwhm) ...]
#                         (metaparameter): Optionally specify smoothing
#                         width(s). Default is 0 / no smoothing. If you wish to
#                         iterate the pipeline across multiple smoothing
#                         separate the list by space (e.g. 2 4 6).
#   -hp High-pass filter (Hz) [High-pass filter (Hz) ...]
#                         (metaparameter): Optionally specify high-pass filter
#                         values to apply to node-extracted time-series for
#                         fMRI. Default is None. If you wish to iterate the
#                         pipeline across multiple high-pass filter thresholds,
#                         values, separate the list by space (e.g. 0.008 0.01).
#   -es Node extraction strategy [Node extraction strategy ...]
#                         Include this flag if you are running functional
#                         connectometry using parcel labels and wish to specify
#                         the name of a specific function (i.e. other than the
#                         mean) to reduce the region's time-series. Options are:
#                         `sum`, `mean`, `median`, `mininum`, `maximum`,
#                         `variance`, `standard_deviation`.
#   -k Number of k clusters [Number of k clusters ...]
#                         (metaparameter): Specify a number of clusters to
#                         produce. If you wish to iterate the pipeline across
#                         multiple values of k, separate the list by space (e.g.
#                         100 150 200).
#   -ct Clustering type [Clustering type ...]
#                         (metaparameter): Specify the types of clustering to
#                         use. Recommended options are: ward, rena, kmeans, or
#                         ncut. Note that imposing spatial constraints with a
#                         mask consisting of disconnected components will
#                         leading to clustering instability in the case of
#                         complete, average, or single clustering. If specifying
#                         list of clustering types, separate them by space.
#   -cm Cluster mask [Cluster mask ...]
#                         (metaparameter): Specify the path to a Nifti1Image
#                         mask file to constrained functional clustering. If
#                         specifying a list of paths to multiple cluster masks,
#                         separate them by space.
#   -ml Minimum fiber length for tracking [Minimum fiber length for tracking ...]
#                         (metaparameter): Include this flag to manually specify
#                         a minimum tract length (mm) for dmri connectome
#                         tracking. Default is 10. If you wish to iterate the
#                         pipeline across multiple minimums, separate the list
#                         by space (e.g. 10 30 50).
#   -em Error margin [Error margin ...]
#                         (metaparameter): Distance (in the units of the
#                         streamlines, usually mm). If any coordinate in the
#                         streamline is within this distance from the center of
#                         any voxel in the ROI, the filtering criterion is set
#                         to True for this streamline, otherwise False. Defaults
#                         to the distance between the center of each voxel and
#                         the corner of the voxel. Default is 5.
#   -dg Direction getter [Direction getter ...]
#                         (metaparameter): Include this flag to manually specify
#                         the statistical approach to tracking for dmri
#                         connectome estimation. Options are: det
#                         (deterministic), closest (clos), and prob
#                         (probabilistic). Default is det. If you wish to
#                         iterate the pipeline across multiple direction-getting
#                         methods, separate the list by space (e.g. 'det',
#                         'prob', 'clos').
#   -norm Normalization strategy for resulting graph(s)
#                         Include this flag to normalize the resulting graph by
#                         (1) maximum edge weight; (2) using log10; (3) using
#                         pass-to-ranks for all non-zero edges; (4) using pass-
#                         to-ranks for all non-zero edges relative to the number
#                         of nodes; (5) using pass-to-ranks with zero-edge
#                         boost; and (6) which standardizes the matrix to values
#                         [0, 1]. Default is (6).
#   -bin                  Include this flag to binarize the resulting graph such
#                         that edges are boolean and not weighted.
#   -dt                   Optionally use this flag if you wish to threshold to
#                         achieve a given density or densities indicated by the
#                         -thr and -min_thr, -max_thr, -step_thr flags,
#                         respectively.
#   -mst                  Optionally use this flag if you wish to apply local
#                         thresholding via the Minimum Spanning Tree approach.
#                         -thr values in this case correspond to a target
#                         density (if the -dt flag is also included), otherwise
#                         a target proportional threshold.
#   -p Pruning Strategy   Include this flag to (1) prune the graph of any
#                         isolated + fully disconnected nodes (i.e. anti-
#                         fragmentation), (2) prune the graph of all but hubs as
#                         defined by any of a variety of definitions (see
#                         ruconfig.yaml), or (3) retain only the largest
#                         connected component subgraph. Default is no pruning.
#                         Include `-p 1` to enable fragmentation-protection.
#   -df                   Optionally use this flag if you wish to apply local
#                         thresholding via the disparity filter approach. -thr
#                         values in this case correspond to α.
#   -mplx Perform various levels of multiplex graph analysis (only) if both structural and diffusion connectometry is run simultaneously.
#                         Include this flag to perform multiplex graph analysis
#                         across structural-functional connectome modalities.
#                         Options include level (1) Create multiplex graphs
#                         using motif-matched adaptive thresholding; (2)
#                         Additionally perform multiplex graph embedding and
#                         analysis. Default is (0) which is no multiplex
#                         analysis.
#   -embed                Optionally use this flag if you wish to embed the
#                         ensemble(s) produced into feature vector(s).
#   -spheres              Include this flag to use spheres instead of parcels as
#                         nodes.
#   -n Resting-state network [Resting-state network ...]
#                         Optionally specify the name of any of the 2017 Yeo-
#                         Schaefer RSNs (7-network or 17-network): Vis, SomMot,
#                         DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent,
#                         VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB,
#                         SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole,
#                         ContA, ContB, ContC, DefaultA, DefaultB, DefaultC,
#                         TempPar. If listing multiple RSNs, separate them by
#                         space. (e.g. -n 'Default' 'Cont' 'SalVentAttn')'.
#   -vox {1mm,2mm}        Optionally use this flag if you wish to change the
#                         resolution of the images in the workflow. Default is
#                         2mm.
#   -plt                  Optionally use this flag if you wish to activate
#                         plotting of adjacency matrices, connectomes, and time-
#                         series.
#   -pm Cores,memory      Number of cores to use, number of GB of memory to use
#                         for single subject run, entered as two integers
#                         seperated by comma. Otherwise, default is `auto`,
#                         which uses all resources detected on the current
#                         compute node.
#   -plug Scheduler type  Include this flag to specify a workflow plugin other
#                         than the default MultiProc.
#   -v                    Verbose print for debugging.
#   -noclean              Disable post-workflow clean-up of temporary runtime
#                         metadata.
#   -work Working directory
#                         Specify the path to a working directory for pynets to
#                         run. Default is /tmp/work.
#   --version             show program's version number and exit

function abspath { echo $(cd $(dirname $1); pwd)/$(basename $1); }

#construct arguments for inputs
t1w=$(jq -r .t1 config.json)

id=$(jq -r .id config.json)
if [[ $id == "null" ]] || [[ $id == '' ]] || [ -z $id ]; then
    sub=$(jq -r '._inputs[] | select(.id == "anat") | .meta.subject' config.json)
fi

optional=""

rm -rf /tmp/work && mkdir -p /tmp/work && chmod a+rwx /tmp/work
rm -rf output && mkdir -p output && chmod a+rw output

dwi=$(jq -r .dwi config.json)
if [[ ($dwi != "null" || $dwi == '') ]]; then
    bval=$(jq -r .bvals config.json)
    bvec=$(jq -r .bvecs config.json)
    optional="$optional -dwi `abspath $dwi` -bval `abspath $bval` -bvec `abspath $bvec`"
    ses=$(jq -r '._inputs[] | select(.id == "dwi") | .meta.session' config.json)
fi

bold=$(jq -r .bold config.json)
if [[ ($bold != "null" || $bold == '') ]]; then
    conf=$(jq -r .regressors config.json)
    if [[ ($conf != "null" || $conf == '') ]]; then
        optional="$optional -func `abspath $bold` -conf `abspath $conf`"
    else
        optional="$optional -func `abspath $bold`"
    fi
    ses=$(jq -r '._inputs[] | select(.id == "task") | .meta.session' config.json)
fi

if [[ ($bold == "null" || $bold == '') ]] && [[ ($dwi == "null" || $dwi == '') ]];then
    echo "\n\nAt least one of BOLD and DWI are required!"
    exit 1
fi

mask=$(jq -r .mask config.json)
if [[ $mask != "null" ]];then
    optional="$optional -m `abspath $mask`"
fi

atlas="$(jq -r .atlas config.json)"
uatlas="$(jq -r .uatlas config.json)"
if [[ ($atlas != "null" || $atlas != '') && ($uatlas == "null" || $uatlas == '') ]]; then
    if [ $(echo -e "$atlas" | wc -l) -gt 1 ]; then
        optional="$optional -a"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$atlas"
    else
        optional="$optional -a $atlas"
    fi
elif [[ ($atlas == "null" || $atlas == '') && ($uatlas != "null" || $uatlas != '') ]]; then
    if [ $(echo -e "$uatlas" | wc -l) -gt 1 ]; then
        optional="$optional -a"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< `abspath "$uatlas"`
    else
        optional="$optional -a `abspath $uatlas`"
    fi
elif [[ ($atlas != "null" || $atlas != '') && ($uatlas != "null" || $uatlas != '') ]]; then
    if [ $(echo -e "$uatlas" | wc -l) -gt 1 ]; then
        optional="$optional -a"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< `abspath "$uatlas"`
    else
        optional="$optional -a `abspath $uatlas`"
    fi
    if [ $(echo -e "$atlas" | wc -l) -gt 1 ]; then
        optional="$optional "
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$atlas"
    else
        optional="$optional $atlas"
    fi
fi

min_thr=$(jq -r .min_thr config.json)
max_thr=$(jq -r .max_thr config.json)
step_thr=$(jq -r .step_thr config.json)
thr=$(jq -r .thr config.json)
if [[ $min_thr != "null" ]]; then
    optional="$optional -min_thr $min_thr"
    optional="$optional -max_thr $max_thr"
    optional="$optional -step_thr $step_thr"
else
    optional="$optional -thr $thr"
fi

# Boolean options
[ "$(jq -r .mst config.json)" == "true" ] && optional="$optional -mst"
[ "$(jq -r .dt config.json)" == "true" ] && optional="$optional -dt"
[ "$(jq -r .embed config.json)" == "true" ] && optional="$optional -embed"
[ "$(jq -r .df config.json)" == "true" ] && optional="$optional -df"
[ "$(jq -r .plt config.json)" == "true" ] && optional="$optional -plt"
[ "$(jq -r .bin config.json)" == "true" ] && optional="$optional -bin"
[ "$(jq -r .spheres config.json)" == "true" ] && optional="$optional -spheres"

prune=$(jq -r .p config.json)
if [[ "$prune" != "null" ]]; then
    optional="$optional -p $prune"
fi

norm=$(jq -r .norm config.json)
if [[ "$norm" != "null" ]]; then
    optional="$optional -norm $norm"
fi

mplx=$(jq -r .mplx config.json)
if [[ "$mplx" != "null" ]]; then
    optional="$optional -mplx $mplx"
fi

rsn=$(jq -r .n config.json)
if [[ "$rsn" != "null" ]]; then
    if [ $(echo -e "$rsn" | wc -l) -gt 1 ]; then
        optional="$optional -n"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$rsn"
    else
        optional="$optional -n $rsn"
    fi
fi

es=$(jq -r .es config.json)
if [[ "$es" != "null" ]]; then
    if [ $(echo -e "$es" | wc -l) -gt 1 ]; then
        optional="$optional -es"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$es"
    else
        optional="$optional -es $es"
    fi
fi

sm=$(jq -r .sm config.json)
if [[ "$sm" != "null" ]]; then
    if [ $(echo -e "$sm" | wc -l) -gt 1 ]; then
        optional="$optional -sm"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$sm"
    else
        optional="$optional -sm $sm"
    fi
fi

hp=$(jq -r .hp config.json)
if [[ "$hp" != "null" ]]; then
    if [ $(echo -e "$hp" | wc -l) -gt 1 ]; then
        optional="$optional -hp"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$hp"
    else
        optional="$optional -hp $hp"
    fi
fi

dg=$(jq -r .dg config.json)
if [[ "$dg" != "null" ]]; then
    if [ $(echo -e "$dg" | wc -l) -gt 1 ]; then
        optional="$optional -dg"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$dg"
    else
        optional="$optional -dg $dg"
    fi
fi

ml=$(jq -r .ml config.json)
if [[ "$ml" != "null" ]]; then
    if [ $(echo -e "$ml" | wc -l) -gt 1 ]; then
        optional="$optional -ml"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$ml"
    else
        optional="$optional -ml $ml"
    fi
fi

em=$(jq -r .em config.json)
if [[ "$em" != "null" ]]; then
    if [ $(echo -e "$em" | wc -l) -gt 1 ]; then
        optional="$optional -em"
        while IFS= read -r line ; do
            optional="$optional $line"
        done <<< "$em"
    else
        optional="$optional -em $em"
    fi
fi


if [ -z "$ses" ]; then
    id="$sub"
else
    id="$sub"_"$ses"
fi

echo "\n\n\n\n"$id"\n\n\n\n"

#sync; echo 3 | tee /proc/sys/vm/drop_caches && swapoff -a && swapon -a

if [ ! -z $SCRATCH ]; then
    WORKINGDIR=$SCRATCH
elif [ ! -z $SINGULARITY_CACHEDIR ]; then
    mkdir -p $SINGULARITY_CACHEDIR/tmp_sifs
    WORKINGDIR=$SINGULARITY_CACHEDIR/tmp_sifs
elif [ ! -z $WORK ]; then
    WORKINGDIR=$WORK
else
    mkdir -p /var/lib/singularity
    WORKINGDIR=/var/lib/singularity
fi

if [ ! -f $WORKINGDIR/sing_pynets.sif ]; then
    singularity build $WORKINGDIR/sing_pynets.sif docker://dpys/pynets:49ec611cce24736cc7ca14db0b5fe828c464dde5
fi

cp $WORKINGDIR/sing_pynets.sif $WORKINGDIR/sing_pynets_"$id".sif

mkdir -p "$PWD"/tmp
export SINGULARITY_TMPDIR=$WORKINGDIR
#singularity exec --cleanenv -B "$PWD"/tmp:/tmp docker://dpys/pynets:49ec611cce24736cc7ca14db0b5fe828c464dde5 pynets \
singularity exec --cleanenv $WORKINGDIR/sing_pynets_"$id".sif pynets \
    "$PWD"/output \
    -id $id \
    -anat `abspath $t1w` \
    -work /tmp/work \
    -mod $(jq -r .mod config.json) \
    -plug 'MultiProc' \
    -pm '24,56' \
    $optional

rm -rf /tmp/work/*
