#!/usr/bin/env bash

# common command-line option parser for run scripts

source scripts/utils.sh

set -eo pipefail

usage () {
    local ret="${1:-1}"
    echo -e "Usage: $0 [-${HLP_FLG}] [-${OLY_FLG}] [-${SRN_FLG}]" \
        "[-$CLN_FLG] [-$DPC_FLG] [-$DAT_FLG DIR]" \
        "[-$EXP_FLG DIR] [-$VER_FLG I] [-$PRC_FLG N]"\
        "[-$WRK_FLG N] [-$LIB_FLG DIR] [-$XTR_FLG ARGS]"\
        "[-$PCA_FLG $(print_arg_choices "${!PCAS[@]}")]"\
        "[-$MDL_FLG $(print_arg_choices "${!MDLS[@]}")]"\
        "[-$TSK_FLG $(print_arg_choices "${!STASK2DARG[@]}")]"\
        "[-$ORD_FLG NN] [-$VCB_FLG N] [-$WID_FLG NN] [-$BET_FLG 0-1]"
    if ((ret == 0)); then
        cat << EOF 1>&2

$help_message

Options
 -$HLP_FLG      Display this message and exit
 -$OLY_FLG      Perform only one step and return
 -$SRN_FLG      Prefix work-heavy commands with "srun" (i.e. when running
         nested in an sbatch command)
 -$CLN_FLG      If set, intermediary files will be deleted once they are no
         longer necessary
 -$DAT_FLG      Root data directory (default: $data)
 -$EXP_FLG      Root experiment directory (defalt: $exp)
 -$MDL_FLG*     Model to train/evaluate (default: $model)
 -$VER_FLG      Non-negative integer version number (default: $ver)
 -$PRC_FLG      Number of processes to spawn in multi-threaded task
         (default: $nproc)
 -$WRK_FLG      Number of workers per process
 -$LIB_FLG      Directory of where librispeech has been downloaded
         (default: downloads into $data/librispeech/local/data)
 -$XTR_FLG      Extra args to pass to trainer (./run.sh only)
 -$PCA_FLG      Number of dimensions to reduce output to.
         (default: no dim reduction; ./scripts/{zrc,superb,baseline}_run.sh only)
 -$TSK_FLG      SUPERB task to run
         (default: $stask; ./scripts/superb_run.sh only)
 -$ORD_FLG      If >0, additionally decode with an n-gram subword lm of this
         max order (default: $lm_ord; ./scripts/baseline_run.sh only)
 -$VCB_FLG      The subword vocabulary size (default: $vocab_size;
         ./scripts/baseline_run.sh only)
 -$WID_FLG      Beam width for decoding (default: $width;
         ./scripts/baseline_run.sh only)
 -$DPC_FLG      Same as -$CLN_FLG except, if the final step of the script is
         completed, also deletes the logits generated during decoding. Decoding
         with other beam widths, language models, etc. will no longer be
         possible (./scripts/baseline_run.sh only)

*-$MDL_FLG accepts models not in the list (i.e. not in conf/model.*.yaml or
not valid) if the model/version pair already exists in the experiment directory
EOF
    fi
    exit "${1:-1}"
}

# constants
CLN_FLG=z
DPC_FLG=Z
DAT_FLG=d
EXP_FLG=e
HLP_FLG=h
LIB_FLG=l
MDL_FLG=m
OLY_FLG=o
ORD_FLG=n
PCA_FLG=k
PRC_FLG=p
SRN_FLG=s
TSK_FLG=t
VCB_FLG=i
VER_FLG=v
WRK_FLG=w
XTR_FLG=x
WID_FLG=q
BNV_FLG=b

DEFT_FT=raw
declare -A FTS=( [raw]=x [fbank]=x [fbank-80]=x [superb.fbank]=x )
declare -A FT2TD_ARGS=(
    [raw]="$(cat conf/feats.raw.args.txt)"
    [fbank]="$(cat conf/feats.fbank.args.txt)"
    [fbank-80]="$(cat conf/feats.fbank-80.args.txt)"
)
declare -A FT2PAD=(
    [raw]="399"
    [fbank]="0"
    [fbank-80]="0"
)

DEFT_SYS="Unknown"

DEFT_TR=100
BASE_FILES="test-clean.tar.gz test-other.tar.gz dev-clean.tar.gz dev-other.tar.gz train-clean-100.tar.gz librispeech-vocab.txt"
declare -A TR2DESC=( [100]="librispeech train-clean-100" [460]="librispeech train-clean-*" [960]="librispeech train-*" )
declare -A TR2DL_ARGS=(
    [100]="--files $BASE_FILES"
    [460]="--files $BASE_FILES train-clean-360.tar.gz"
    [960]="--files $BASE_FILES train-clean-360.tar.gz train-other-500.tar.gz"
)
declare -A TR2TD_ARGS=(
    [100]="--compute-up-to 100"
    [460]="--compute-up-to 360 --aggregate-by-symlink"
    [960]="--aggregate-by-symlink"
)
declare -A TR2TDIR=(
    [100]="train_clean_100"
    [460]="train_clean_460"
    [960]="train_all_960"
)

declare -A MDLS
for f in conf/model.*.yaml; do
    exp_name="${f:11:-5}"
    act_name="$(cat "$f" | tr -d '\r' | awk '/^name:/ {print $2}')"
    if [ "$exp_name" != "$act_name" ]; then
        echo -e "expected name: in '$f' to be '$exp_name'; got '$act_name';" \
            " ignoring as possible model"
        continue
    fi
    MDLS["$exp_name"]=x
done
if [ -z "${MDLS["cpc.deft"]}" ]; then
    echo -e "Missing cpc.deft!"
    exit 1
fi
declare -A PCAS=( [8]=x [16]=x [32]=x [64]=x [128]=x )

declare -A STASK2DARG=( [pr]=ctc [asr]=asr )
declare -A SASRPART2IDX=(
    [train-clean-100]=0
    [train-clean-360]=1
    [train-other-500]=2
    [dev-clean]=3
    [dev-other]=4
    [test-clean]=5
    [test-other]=6
)

if [[ "$0" =~ "superb_run.sh" ]]; then
    MDLS["superb.fbank"]=x
    TR2DESC["superb"]="Consult SUPERB recipe for more information"
fi

declare BASELINE_CONDS=( lm nolm )

# variables
data="data"
exp="exp"
model="cpc.deft"
libri=
ver=0
nproc=1
nwork=4
only=0
xtra_args=
pca=
cmd=
cmd_p=
stask=pr
clean=false
deepclean=false
lm_ord=0
vocab_size=2000
width=32

while getopts "${HLP_FLG}${OLY_FLG}${SRN_FLG}${CLN_FLG}${DPC_FLG}${DAT_FLG}:${EXP_FLG}:${MDL_FLG}:${VER_FLG}:${PRC_FLG}:${WRK_FLG}:${LIB_FLG}:${XTR_FLG}:${PCA_FLG}:${TSK_FLG}:${ORD_FLG}:${VCB_FLG}:${WID_FLG}:${BNV_FLG}:" opt; do
    case $opt in
        ${HLP_FLG})
            usage 0
            ;;
        ${OLY_FLG})
            only=1
            ;;
        ${SRN_FLG})
            cmd="srun -- "
            cmd_p="srun --ntasks=1 --nodes=1 -- "
            ;;
        ${CLN_FLG})
            clean=true
            ;;
        ${DPC_FLG})
            clean=true
            deepclean=true
            ;;
        ${DAT_FLG})
            argcheck_is_writable $opt "$OPTARG"
            data="$OPTARG"
            ;;
        ${EXP_FLG})
            argcheck_is_writable $opt "$OPTARG"
            exp="$OPTARG"
            ;;
        ${MDL_FLG})
            argcheck_is_basename $opt "$OPTARG"
            model="$OPTARG"
            ;;
        ${VER_FLG})
            argcheck_is_nnint $opt "$OPTARG"
            ver="$OPTARG"
            ;;
        ${PRC_FLG})
            argcheck_is_nat $opt "$OPTARG"
            nproc="$OPTARG"
            ;;
        ${WRK_FLG})
            argcheck_is_nat $opt "$OPTARG"
            nwork="$OPTARG"
            ;;
        ${LIB_FLG})
            argcheck_is_readable $opt "$OPTARG"
            libri="$OPTARG"
            ;;
        ${XTR_FLG})
            xtra_args="$OPTARG"
            ;;
        ${PCA_FLG})
            argcheck_is_a_choice $opt "${!PCAS[@]}" "$OPTARG"
            pca="$OPTARG"
            ;;
        ${TSK_FLG})
            argcheck_is_a_choice $opt "${!STASK2DARG[@]}" "$OPTARG"
            stask="$OPTARG"
            ;;
        ${ORD_FLG})
            argcheck_is_nnint $opt "$OPTARG"
            lm_ord="$OPTARG"
            ;;
        ${VCB_FLG})
            argcheck_is_nat $opt "$OPTARG"
            vocab_size="$OPTARG"
            ;;
        ${WID_FLG})
            argcheck_is_nat $opt "$OPTARG"
            width="$OPTARG"
            ;;
        ?)
            echo "-$OPTARG is not an option"
            usage
            ;;
    esac
done

# the first step is always to write the config file to the experiment
# directory. This way, if the master config file changes, it doesn't muck
# up what we've got
em="$exp/$model/version_$ver"
cfg="$em/model.yaml"
if [ ! -f "$cfg" ]; then
    if [ -z "${MDLS[$model]}" ]; then
        echo "Cannot create new model '$model': conf/model.$model.yaml does not exist or is not valid"
        exit 1
    fi
    mkdir -p "$em"
    if [[ "$model" =~ ^superb ]]; then
        cat <<EOF > "$cfg"
name: $model
system_description: "SUPERB default model $model"
feat_type: raw
train_part: "superb"
EOF
        ((only)) && exit 0
    else
        # this first step ensures conf/model.$model.yaml can be read as-is by
        # the scpc command. The second joins all the default configuration
        # values with the modified ones. The latter ensures the model always
        # trains with a specific configuration, even if the defaults are
        # changed later.
        echo "Checking configuration conf/model.$model.yaml parses"
        $cmd_p scpc-train --read-model-yaml "conf/model.$model.yaml" -h > /dev/null
        echo "Writing $model configuration to '$cfg'"
        $cmd_p scpc-train --print-model-yaml | \
            combine-yaml-files --quiet --nested \
                - "conf/model.$model.yaml" "$cfg"
        ((only)) && exit 0
    fi
fi

ft="$(cat "$cfg" | tr -d '\r' | awk -v ft=$DEFT_FT '$1 == "feat_type:" {ft=$2} END {print ft}')"
if [ -z "${FTS[$ft]}" ]; then
    echo "expected feat_type: in '$cfg' to be one of ${!FTS[*]}; got $ft"
    exit 1
fi

tr="$(cat "$cfg" | tr -d '\r' | awk -v tr=$DEFT_TR '$1 == "train_part:" {gsub(/["'"'"']/, "", $2); tr=$2} END {print tr}')"
if [ -z "${TR2DESC[$tr]}" ]; then
    echo "expected train_part: in '$cfg' to be one of ${!TR2DESC[*]}; got $tr"
    exit 1
fi
tdir="${TR2TDIR[$tr]}"
train_description="${TR2DESC[$tr]}"

system_description="$(cat "$cfg" | tr -d '\r' | awk -v s="$DEFT_SYS" '$1 == "system_description:" {$1=""; print}')"

a="$(mktemp)"
echo "${FT2TD_ARGS[$ft]}" | tr ' ' '\n' > "$a"
if [ ! -f "$em/expert.args.full.txt" ]; then
    cp "$a" "$em/expert.args.full.txt"
else
    d="$(diff "$a" "$em/expert.args.full.txt" || true)"
    if [ ! -z "$d" ]; then
        echo "Expected and actual expert arguments differ!"
        echo "$d"
        exit 1
    fi
fi
for x in "${!PCAS[@]}"; do
    cat "$em/expert.args.full.txt" <(echo "--pca-file
$em/pca_$x.pt") > "$a"
    if [ ! -f "$em/expert.args.pca_$x.txt" ]; then
        cp "$a" "$em/expert.args.pca_$x.txt"
    else
        d="$(diff "$a" "$em/expert.args.pca_$x.txt" || true)"
        if [ ! -z "$d" ]; then
            echo "Expected and actual expert arguments differ!"
            echo "$d"
            exit 1
        fi
    fi
done
rm -f "$a"

if [ -z "$pca" ]; then
    expert_config="$em/expert.args.full.txt"
else
    expert_config="$em/expert.args.pca_$pca.txt"
fi
expert_args="$(cat "$expert_config")"

darg="${STASK2DARG[$stask]}"

echo "cmd: $0 $*"
echo "system description: $system_description ($model)"
echo "training set: $train_description ($tr)"
echo "version: $ver"
