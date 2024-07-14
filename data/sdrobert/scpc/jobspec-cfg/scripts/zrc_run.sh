#!/usr/bin/env bash

# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo '$CONDA_DEFAULT_ENV set to base. This is probably undesired. If you'
    echo "really want this, modify $0 to exclude the check"
    exit 1
fi

if ! pip freeze | grep 'scpc' --quiet; then
    pip install -e '.[zrc]'
fi

set -e

# command-line option parsing
help_message="Score pre-trained speech reps with ZeroSpeech's libriABX task"
source scripts/preamble.sh

dz="$data/zerospeech"
if [ -z "$pca" ]; then
  zs="$em/zrc/librispeech/full"
else
  zs="$em/zrc/librispeech/pca_$pca"
fi


ckpt="$em/best.ckpt"
if [ ! -f "$ckpt" ]; then
    echo "'$ckpt' is not a file (did you finish ./run.sh?)"
    exit 1
fi

# for zerospeech-benchmarks pkg
export APP_DIR="$dz/local/data"
export TEMP_DIR="$TMPDIR"

set -eo pipefail

if [ ! -f "$dz/.complete" ]; then
    echo "Downloading zerospeech abxLS"
    $cmd_p zrc datasets:pull abxLS-dataset
    touch "$dz/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$zs/meta.yaml" ]; then
    echo "Constructing abxLS zerospeech submission using $model model"
    rm -rf "$zs"
    zrc submission:init abxLS "$zs"
    rm "$zs/meta.yaml"
    (
        export fsize="$(scpc-info $expert_args "$ckpt" | awk '$1 == "downsampling_factor" {print $2 / 16000}')"
        export cuda_avail="$(python -c 'import torch; print("true" if torch.cuda.is_available() else "false")')"
        cat conf/zrc.params.template.yaml | envsubst > "$zs/params.yaml"
    )
    (
        export train_description system_description
        cat conf/zrc.meta.template.yaml | envsubst > "$zs/meta.yaml"
    )
    ((only)) && exit 0
fi

if [ ! -f "$zs/.a2r_complete" ]; then
    echo "Computing abxLS speech representations using $model model"
    $cmd_p scpc-a2r $expert_args \
        --in-dir "$APP_DIR/datasets/abxLS-dataset" "$ckpt" "$zs" --numpy
    touch "$zs/.a2r_complete"
    ((only)) && exit 0
fi

if [ ! -f "$zs/scores/.complete" ]; then
    echo "Scoring abxLS zerospeech submissing using $model model"
    $cmd_p zrc benchmarks:run abxLS "$zs"
    touch "$zs/scores/.complete"
    $clean && find "$zs/" -name '*.npy' -delete
    ((only)) && exit 0
fi

if $clean; then
    find "$zs/" -name '*.npy' -delete
fi

cat "$zs/scores/score_all_phonetic.csv"
