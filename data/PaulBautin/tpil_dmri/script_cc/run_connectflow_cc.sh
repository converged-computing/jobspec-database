#!/bin/bash

# This would run Tractoflow with the following parameters:
#   - bids: clinical data from TPIL lab (27 CLBP and 25 control subjects), if not in BIDS format use flag --input
#   - with-singularity: container image scilus v1.3.0 (runs: dmriqc_flow, tractoflow, recobundleX, tractometry)
#   - with-report: outputs a processing report when pipeline is finished running
#   - Dti_shells 0 and 1000 (usually <1200), Fodf_shells 0 1000 and 2000 (usually >700, multishell CSD-ms).
#   - profile: bundling, bundling profile will set the seeding strategy to WM as opposed to interface seeding that is usually used for connectomics


#SBATCH --nodes=1              # --> Generally depends on your nb of subjects.
                               # See the comment for the cpus-per-task. One general rule could be
                               # that if you have more subjects than cores/cpus (ex, if you process 38
                               # subjects on 32 cpus-per-task), you could ask for one more node.
#SBATCH --cpus-per-task=32     # --> You can see here the choices. For beluga, you can choose 32, 40 or 64.
                               # https://docs.computecanada.ca/wiki/B%C3%A9luga/en#Node_Characteristics
#SBATCH --mem=0                # --> 0 means you take all the memory of the node. If you think you will need
                               # all the node, you can keep 0.
#SBATCH --time=24:00:00

#SBATCH --mail-user=paul.bautin@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load StdEnv/2020 java/14.0.2 nextflow/21.10.3 apptainer/1.1.8


my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/scilus_1.5.0.sif' # or .img
my_main_nf='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/connectoflow/main.nf'
my_input='/home/pabaua/scratch/tpil_dev/results/clbp/23-08-15_connectivity_prep/results'
my_template='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/atlas/mni_masked.nii.gz'
my_labels_list='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/atlas_brainnetome_first_label_list.txt'


NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input --labels_list $my_labels_list --template $my_template \
    --apply_t1_labels_transfo false -with-singularity $my_singularity_img -resume



