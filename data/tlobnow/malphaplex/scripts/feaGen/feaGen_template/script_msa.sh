#!/bin/bash -l
#SBATCH -J 1_MSA
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

# We will use 2 GPUs:
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=250000

# When using >1 GPUs, please adapt the variable XLA_PYTHON_CLIENT_MEM_FRACTION as well (see below)!
#SBATCH --mail-type=NONE
#SBATCH --mail-user=$USER@mpiib-berlin.mpg.de
#SBATCH --time=06:00:00

# AlphaFold2 template submit script (single sequence case) for RAVEN @ MPCDF,
# please create a local copy and customize to your use case.
#
# Important: Access the AF2 data ONLY via ${ALPHAFOLD_DATA} provided by MPCDF,
# please don't create per-user copies of the database in '/ptmp' or '/u' for performance reasons.

set -e

module purge
module load alphafold/2.3.1
module load cuda/11.4

# include parameters common to the CPU and the GPU steps
source 01_user_parameters.inc

# check if the directories set by the alphafold module do exist
if [ ! -d ${ALPHAFOLD_DATA} ]; then
  echo "Could not find ${ALPHAFOLD_DATA}. STOP."
  exit 1
fi
mkdir -p ${OUTPUT_DIR}


# make CUDA and AI libs accessible
export LD_LIBRARY_PATH=${ALPHAFOLD_HOME}/lib:${LD_LIBRARY_PATH}
# put temporary files into a ramdisk
export TMPDIR=${JOB_SHMTMPDIR}

### ENABLE CUDA UNIFIED MEMORY ###############################################
export TF_FORCE_UNIFIED_MEMORY=1
        # Enable jax allocation tweak to allow for larger models, note that
        # with unified memory the fraction can be larger than 1.0 (=100% of single GPU memory):
        # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
        # When using 3 GPUs:
export XLA_PYTHON_CLIENT_MEM_FRACTION=4


# run threaded tools with the correct number of threads (MPCDF customization)
export NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


## Variable settings collected and adapted from
## https://github.com/deepmind/alphafold/blob/main/docker/run_docker.py

# Path to directory of supporting data, contains 'params' dir.
DOWNLOAD_DIR=${ALPHAFOLD_DATA}

# Path to the Uniref90 database for use by JackHMMER.
#uniref90_database_path = os.path.join(FLAGS.data_dir, 'uniref90', 'uniref90.fasta')
uniref90_database_path=${DOWNLOAD_DIR}/uniref90/uniref90.fasta

# Path to the MGnify database for use by JackHMMER.
#mgnify_database_path = os.path.join( FLAGS.data_dir, 'mgnify', 'mgy_clusters_2018_12.fa')
#mgnify_database_path=${DOWNLOAD_DIR}/mgnify/mgy_clusters_2018_12.fa
mgnify_database_path=/raven/ri/public_sequence_data/alphafold2/git-v2.3.1/data/mgnify/mgy_clusters_2022_05.fa

# Path to the Uniprot database for use by JackHMMER.
#uniprot_database_path = os.path.join(FLAGS.data_dir, 'uniprot', 'uniprot.fasta')
uniprot_database_path=${DOWNLOAD_DIR}/uniprot/uniprot.fasta

# Path to the BFD database for use by HHblits.
#bfd_database_path = os.path.join(FLAGS.data_dir, 'bfd', 'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')
bfd_database_path=${DOWNLOAD_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt

# Path to the Small BFD database for use by JackHMMER.
# small_bfd_database_path = os.path.join(FLAGS.data_dir, 'small_bfd', 'bfd-first_non_consensus_sequences.fasta')
small_bfd_database_path=${DOWNLOAD_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta

# Path to the PDB seqres database for use by hmmsearch.
#pdb_seqres_database_path = os.path.join(FLAGS.data_dir, 'pdb_seqres', 'pdb_seqres.txt')
pdb_seqres_database_path=${DOWNLOAD_DIR}/pdb_seqres/pdb_seqres.txt

# Path to the Uniclust30 database for use by HHblits.
#uniclust30_database_path = os.path.join(FLAGS.data_dir, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')
uniclust30_database_path=${DOWNLOAD_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08

# Path to the PDB70 database for use by HHsearch.
# pdb70_database_path = os.path.join(FLAGS.data_dir, 'pdb70', 'pdb70')
pdb70_database_path=${DOWNLOAD_DIR}/pdb70/pdb70

# Path to a directory with template mmCIF structures, each named <pdb_id>.cif.
#template_mmcif_dir = os.path.join(FLAGS.data_dir, 'pdb_mmcif', 'mmcif_files')
template_mmcif_dir=${DOWNLOAD_DIR}/pdb_mmcif/mmcif_files

# Path to a file mapping obsolete PDB IDs to their replacements.
#obsolete_pdbs_path = os.path.join(FLAGS.data_dir, 'pdb_mmcif', 'obsolete.dat')
obsolete_pdbs_path=${DOWNLOAD_DIR}/pdb_mmcif/obsolete.dat


# run the application
#srun ${ALPHAFOLD_HOME}/bin/python3 ${ALPHAFOLD_HOME}/app/alphafold/run_alphafold.py \
srun ${ALPHAFOLD_HOME}/bin/python3 -u $AF_DIR/run_af2c_fea.py \
        --output_dir="${OUTPUT_DIR}" \
        --fasta_paths="${FASTA_PATHS}" \
        --db_preset="${PRESET}" \
        --data_dir=${DOWNLOAD_DIR} \
	--small_bfd_database_path=${small_bfd_database_path} \
        --uniref90_database_path=${uniref90_database_path} \
        --mgnify_database_path=${mgnify_database_path} \
        --pdb70_database_path=${pdb70_database_path} \
        --template_mmcif_dir=${template_mmcif_dir} \
        --obsolete_pdbs_path=${obsolete_pdbs_path} \
        --max_template_date="2023-01-01"
#       ^^^ last line: limit to msa and templates on the CPU, then STOP


# Available command line flags (output from '--helpfull'):
#
# run_alphafold.py:
#   --[no]benchmark: Run multiple JAX model evaluations to obtain a timing that
#     excludes the compilation time, which should be more indicative of the time
#     required for inferencing many proteins.
#     (default: 'false')
#   --bfd_database_path: Path to the BFD database for use by HHblits.
#   --data_dir: Path to directory of supporting data.
#   --db_preset: <full_dbs|reduced_dbs>: Choose preset MSA database configuration
#     - smaller genetic database config (reduced_dbs) or full genetic database
#     config  (full_dbs)
#     (default: 'full_dbs')
#   --fasta_paths: Paths to FASTA files, each containing a prediction target that
#     will be folded one after another. If a FASTA file contains multiple
#     sequences, then it will be folded as a multimer. Paths should be separated
#     by commas. All FASTA paths must have a unique basename as the basename is
#     used to name the output directories for each prediction.
#     (a comma separated list)
#   --hhblits_binary_path: Path to the HHblits executable.
#     (default: '/raven/u/khr/af2-install/bin/hhblits')
#   --hhsearch_binary_path: Path to the HHsearch executable.
#     (default: '/raven/u/khr/af2-install/bin/hhsearch')
#   --hmmbuild_binary_path: Path to the hmmbuild executable.
#     (default: '/raven/u/khr/af2-install/bin/hmmbuild')
#   --hmmsearch_binary_path: Path to the hmmsearch executable.
#     (default: '/raven/u/khr/af2-install/bin/hmmsearch')
#   --jackhmmer_binary_path: Path to the JackHMMER executable.
#     (default: '/raven/u/khr/af2-install/bin/jackhmmer')
#   --kalign_binary_path: Path to the Kalign executable.
#     (default: '/raven/u/khr/af2-install/bin/kalign')
#   --max_template_date: Maximum template release date to consider. Important if
#     folding historical test sets.
#   --mgnify_database_path: Path to the MGnify database for use by JackHMMER.
#   --model_preset: <monomer|monomer_casp14|monomer_ptm|multimer>: Choose preset
#     model configuration - the monomer model, the monomer model with extra
#     ensembling, monomer model with pTM head, or multimer model
#     (default: 'monomer')
#   --num_multimer_predictions_per_model: How many predictions (each with a
#     different random seed) will be generated per model. E.g. if this is 2 and
#     there are 5 models then there will be 10 predictions per input. Note: this
#     FLAG only applies if model_preset=multimer
#     (default: '5')
#     (an integer)
#   --obsolete_pdbs_path: Path to file containing a mapping from obsolete PDB IDs
#     to the PDB IDs of their replacements.
#   --output_dir: Path to a directory that will store the results.
#   --pdb70_database_path: Path to the PDB70 database for use by HHsearch.
#   --pdb_seqres_database_path: Path to the PDB seqres database for use by
#     hmmsearch.
#   --random_seed: The random seed for the data pipeline. By default, this is
#     randomly generated. Note that even if this is set, Alphafold may still not
#     be deterministic, because processes like GPU inference are nondeterministic.
#     (an integer)
#   --[no]run_relax: Whether to run the final relaxation step on the predicted
#     models. Turning relax off might result in predictions with distracting
#     stereochemical violations but might help in case you are having issues with
#     the relaxation stage.
#     (default: 'true')
#   --small_bfd_database_path: Path to the small version of BFD used with the
#     "reduced_dbs" preset.
#   --template_mmcif_dir: Path to a directory with template mmCIF structures, each
#     named <pdb_id>.cif
#   --uniclust30_database_path: Path to the Uniclust30 database for use by
#     HHblits.
#   --uniprot_database_path: Path to the Uniprot database for use by JackHMMer.
#   --uniref90_database_path: Path to the Uniref90 database for use by JackHMMER.
#   --[no]use_gpu_relax: Whether to relax on GPU. Relax on GPU can be much faster
#     than CPU, so it is recommended to enable if possible. GPUs must be available
#     if this setting is enabled.
#   --[no]use_precomputed_msas: Whether to read MSAs that have been written to
#     disk instead of running the MSA tools. The MSA files are looked up in the
#     output directory, so it must stay the same between multiple runs that are to
#     reuse the MSAs. WARNING: This will not check if the sequence, database or
#     configuration have changed.
#     (default: 'false')

# absl.app:
#   -?,--[no]help: show this help
#     (default: 'false')
#   --[no]helpfull: show full help
#     (default: 'false')
#   --[no]helpshort: show this help
#     (default: 'false')
#   --[no]helpxml: like --helpfull, but generates XML output
#     (default: 'false')
#   --[no]only_check_args: Set to true to validate args and exit.
#     (default: 'false')
#   --[no]pdb: Alias for --pdb_post_mortem.
#     (default: 'false')
#   --[no]pdb_post_mortem: Set to true to handle uncaught exceptions with PDB post
#     mortem.
#     (default: 'false')
#   --profile_file: Dump profile information to a file (for python -m pstats).
#     Implies --run_with_profiling.
#   --[no]run_with_pdb: Set to true for PDB debug mode
#     (default: 'false')
#   --[no]run_with_profiling: Set to true for profiling the script. Execution will
#     be slower, and the output format might change over time.
#     (default: 'false')
#   --[no]use_cprofile_for_profiling: Use cProfile instead of the profile module
#     for profiling. This has no effect unless --run_with_profiling is set.
#     (default: 'true')

# absl.logging:
#   --[no]alsologtostderr: also log to stderr?
#     (default: 'false')
#   --log_dir: directory to write logfiles into
#     (default: '')
#   --logger_levels: Specify log level of loggers. The format is a CSV list of
#     `name:level`. Where `name` is the logger name used with
#     `logging.getLogger()`, and `level` is a level name  (INFO, DEBUG, etc). e.g.
#     `myapp.foo:INFO,other.logger:DEBUG`
#     (default: '')
#   --[no]logtostderr: Should only log to stderr?
#     (default: 'false')
#   --[no]showprefixforinfo: If False, do not prepend prefix to info messages when
#     it's logged to stderr, --verbosity is set to INFO level, and python logging
#     is used.
#     (default: 'true')
#   --stderrthreshold: log messages at this level, or more severe, to stderr in
#     addition to the logfile.  Possible values are 'debug', 'info', 'warning',
#     'error', and 'fatal'.  Obsoletes --alsologtostderr. Using --alsologtostderr
#     cancels the effect of this flag. Please also note that this flag is subject
#     to --verbosity and requires logfile not be stderr.
#     (default: 'fatal')
#   -v,--verbosity: Logging verbosity level. Messages logged at this level or
#     lower will be included. Set to 1 for debug logging. If the flag was not set
#     or supplied, the value will be changed from the default of -1 (warning) to 0
#     (info) after flags are parsed.
#     (default: '-1')
#     (an integer)

# absl.testing.absltest:
#   --test_random_seed: Random seed for testing. Some test frameworks may change
#     the default value of this flag between runs, so it is not appropriate for
#     seeding probabilistic tests.
#     (default: '301')
#     (an integer)
#   --test_randomize_ordering_seed: If positive, use this as a seed to randomize
#     the execution order for test cases. If "random", pick a random seed to use.
#     If 0 or not set, do not randomize test case execution order. This flag also
#     overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable.
#     (default: '')
#   --test_srcdir: Root of directory tree where source files live
#     (default: '')
#   --test_tmpdir: Directory for temporary testing files
#     (default: '/tmp/absl_testing')
#   --xml_output_file: File to store XML test results
#     (default: '')

# tensorflow.python.ops.parallel_for.pfor:
#   --[no]op_conversion_fallback_to_while_loop: DEPRECATED: Flag is ignored.
#     (default: 'true')

# tensorflow.python.tpu.client.client:
#   --[no]hbm_oom_exit: Exit the script when the TPU HBM is OOM.
#     (default: 'true')
#   --[no]runtime_oom_exit: Exit the script when the TPU runtime is OOM.
#     (default: 'true')

# absl.flags:
#   --flagfile: Insert flag definitions from the given file into the command line.
#     (default: '')
#   --undefok: comma-separated list of flag names that it is okay to specify on
#     the command line even if the program does not define a flag with that name.
#     IMPORTANT: flags in this list that have arguments MUST use the --flag=value
#     format.
#     (default: '')
