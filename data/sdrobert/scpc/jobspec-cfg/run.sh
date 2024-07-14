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

set -eo pipefail

if [ ! -d "prep" ]; then
    echo "'prep' is not a directory. Did you call

    git submodule update --init"

    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo '$CONDA_DEFAULT_ENV set to base. This is probably undesired. If you'
    echo "really want this, modify $0 to exclude the check"
    exit 1
fi

# We likely don't need everything, but just in case
if ! pip freeze | grep 'scpc' --quiet; then
    pip install -e '.[all]'
fi

# command-line option parsing
help_message="Pre-train speech rep model"
source scripts/preamble.sh

dl="$data/librispeech"
dlf="$data/librispeech/$ft"
em="$exp/$model/version_$ver"
ckpt="$em/best.ckpt"

if [ -z "$libri" ] &&  [ ! -f "$dl/.${tr}_complete" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.${tr}_complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download ${TR2DL_ARGS[$tr]}
        touch "$libri/.${tr}_complete"
        ((only)) && exit 0
    fi
fi

if [ ! -f "$dl/.${tr}_complete" ]; then
    echo "Performing common prep of librispeech"
    $cmd_p python prep/librispeech.py "$dl" preamble \
        --speakers-are-readers --exclude-subsets "$libri"
    $cmd_p python prep/librispeech.py "$dl" init_char "$libri"
    touch "$dl/.${tr}_complete"
    ((only)) && exit 0
fi

if [ ! -f "$dlf/.${tr}_complete" ]; then
    echo "Computing $ft features of librispeech"
    $cmd_p python prep/librispeech.py \
        "$dl" torch_dir \
            char $ft ${FT2TD_ARGS[$ft]} ${TR2TD_ARGS[$tr]} --skip-verify
    touch "$dlf/.${tr}_complete"
    ((only)) && exit 0
fi

if [ $tr = 100 ]; then
    # special commands only for train-clean-100 training
    # if [ ! -f "$dlf/train_clean_100/ali/.complete" ]; then
    #     echo "Moving per-frame alignments to data dir for $ft"
    #     rm -f "$dlf/train_clean_100/ali/.complete-"*
    #     if [ "$ft" = "raw" ]; then
    #         # convert 10ms frame alignments to sample alignments and store the
    #         # results in train_clean_100/ali
    #         #
    #         # N.B. --snip-edges=true for kaldi, which means frames = (samps -
    #         #   400) // 160 + 1, or samps <= 160 * (frames - 1) + 400
    #         #
    #         # since the first frame starts at sample 0, the extra 240 to 399
    #         # samples are at the end of the recording. We thus use the
    #         # alignment of the final frame for an additional 399 frames, then
    #         # crop using get-torch-spect-data-dir-info.
    #         align2frames() {
    #             awk -v spf=160 -v pad=${FT2PAD[$ft]} -v i=$i -v I=$nproc '
    # (NR + i - 2) % I == 0 {
    # printf "lbi-%s", $1;
    # for (n=2; n <= NF; ++n) for (m=0; m < spf; ++m) printf " %s", $n;
    # for (m=0; m < pad; ++m) printf " %s", $NF;
    # printf "\n";
    # }'
    #         }
    #     else
    #         align2frames() {
    #             awk -v i=$i -v I=$nproc '(NR + i - 2) % I == 0 {printf "lbi-"; print;}'
    #         }
    #     fi
    #     for i in $(seq 1 $nproc); do
    #         $cmd_p unzip -cq resources/converted_aligned_phones.zip | \
    #             align2frames | \
    #             write-table-to-torch-dir \
    #                 -i iv -o long \
    #                 'ark,t,s,o,cs:-' \
    #                 "$dlf/train_clean_100/ali" && \
    #             touch "$dlf/train_clean_100/ali/.complete-$i" &
    #     done
    #     wait
    #     for i in $(seq 1 $nproc); do
    #         if [ ! -f "$dlf/train_clean_100/ali/.complete-$i" ]; then
    #             echo -e "Process $i/$I failed!"
    #             rm -f "$dlf/train_clean_100/ali/.complete-"*
    #             exit 1
    #         fi
    #     done
    #     rm -f "$dlf/train_clean_100/ali/.complete-"*
    #     touch "$dlf/train_clean_100/ali/.complete"
    #     ((only)) && exit 0
    # fi

    if [ ! -f "$dlf/ext/train_clean_100.info.ark" ]; then
        echo "Fixing alignments and getting info file"
        failed=0
        $cmd_p get-torch-spect-data-dir-info --fix ${FT2PAD[$ft]} \
            "$dlf/"{train_clean_100,ext/train_clean_100.info.ark} || failed=1
        if ((failed)); then
            rm -f "$dlf/ext/train_clean_100.info.ark"
            exit 1
        fi
        ((only)) && exit 0
    fi

    # There's a slight mismatch between the utterances of 
    #   train_clean_100 - train_clean_100_test_subset
    # and
    #   train_clean_100_train_subset
    # which is why we handle training subset of 'train_clean_100' specially
    if [ ! -f "$dlf/train_clean_100_train_subset/.complete" ]; then
        echo "Making train subset of train_clean_100"
        rm -rf "$dlf/train_clean_100_train_subset"
        $cmd_p subset-torch-spect-data-dir --num-workers=$nwork \
            "$dlf/train_clean_100"{,_train_subset} \
            --symlink \
            --utt-list-file resources/train_clean_100_train_subset.txt
        touch "$dlf/train_clean_100_train_subset/.complete"
        ((only)) && exit 0
    fi
fi

if [ ! -f "$dlf/${tdir}_train_subset/.complete" ]; then
    echo "Making train subset of $tdir"
    rm -rf "$dlf/${tdir}_train_subset"
    mkdir -p "$dlf/${tdir}_train_subset"
    find "$dlf/$tdir/feat" -name '*.pt' -exec basename {} \; | \
        cut -d '.' -f 1 | sort | \
        join -v1 - <(sort resources/train_clean_100_test_subset.txt) \
            > "$dlf/${tdir}_train_subset/uttids"
    $cmd_p subset-torch-spect-data-dir --num-workers=$nwork \
        "$dlf/$tdir"{,_train_subset} \
        --symlink \
        --utt-list-file "$dlf/${tdir}_train_subset/uttids"
    $clean && rm "$dlf/${tdir}_train_subset/uttids"
    touch "$dlf/${tdir}_train_subset/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$dlf/train_clean_100_test_subset/.complete" ]; then
    echo "Making test subset of train_clean_100"
    rm -rf "$dlf/train_clean_100_test_subset"
    $cmd_p subset-torch-spect-data-dir --num-workers=$nwork \
        "$dlf/train_clean_100"{,_test_subset} \
        --symlink \
        --utt-list-file resources/train_clean_100_test_subset.txt
    touch "$dlf/train_clean_100_test_subset/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$ckpt" ]; then
    echo "Training $model model"
    $cmd scpc-train \
            --read-model-yaml "$em/model.yaml" \
            "$dlf/${tdir}_train_subset" \
            "$dlf/train_clean_100_test_subset" \
            --root-dir "$exp" \
            "--version=$ver" "--num-workers=$nwork" $xtra_args
    [ -f "$ckpt" ] || exit 1
    $clean && find "$em/" -name '*.ckpt' -not -name 'best.ckpt' -delete
    ((only)) && exit 0
fi

for pca in "${!PCAS[@]}"; do
    pcaf="$em/pca_$pca.pt"
    if [ ! -f "$pcaf" ]; then
        echo "PCA of dim $pca being performed for $model"
        $cmd_p scpc-pca \
            --read-yaml conf/pca.yaml --num-workers=$nwork \
            "$dlf/${tdir}_test_subset" "$ckpt" $pca "$pcaf"
        ((only)) && exit 0
    fi
done

if $clean; then
    find "$em/" -name '*.ckpt' -not -name 'best.ckpt' -delete
fi

# # check the average number of 10ms frames in an utterance
# unzip -cq resources/converted_aligned_phones.zip | \
#     awk 'BEGIN {n = 0; c = 0} {n+=NF - 1; c+=1} END {print n / c}'
# # returns 1267.12
# # 160 frames/sec: 7.9195 secs/utt
# # 16,000 samps/sec: 126,712 samps/utt
# # 20,840 samps/win: 6.187109375 win/utt
# # to get 8 win per batch, you need 2 utt to be safe
# # to get 16 win per batch, you need 3 utt to be safe
# # to get 32 win per batch, you need 6 utt to be safe


# # check the mean and stddev of phoneme lengths, excluding silence
# print-torch-ali-data-dir-length-moments \
#     data/librispeech/raw/train_clean_100/ali --std --exclude-ids 0
# # returns mean=1362.625 std=806.739
# #  - nearest multiple of 160: 1280 (9 frames)
# # sum of three phoneme Gaussians:
# #   mean=3*1362.625=4087.875 std=sqrt(3)*806.739=1397.313
# # 1 phone 95% confidence interval = 2,976.103
# #  - nearest multiple of 160: 3,040 (19 frames)
# # 3 phone 95% confidence interval = 6,882.501
# #  - nearest multiple of 160: 6,880 (43 frames)
# # 64 win/batch * (20,480 samps/win / 3,040 samps/win) ~ 432 win (108 / gpu)
# # 64 win/batch * (20,480 samps/win / 6,880 samps/win) ~ 192 win (48 / gpu)


# # the receptive field of a coefficient w/ K conv layers = r_1
# # r_K = kernel_k
# # r_k = kernel_k + stride_k (r_{k+1} - 1)
# # kernel_5=4,  stride_5=2, r_5 = 4
# # kernel_4=4,  stride_4=2, r_4 = 4 + 2 * 3 = 10
# # kernel_3=4,  stride_3=2, r_3 = 4 + 2 * 9 = 22
# # kernel_2=8,  stride_2=4, r_2 = 8 + 4 * 21 = 92
# # kernel_1=10, stride_1=5, r_1 = 10 + 5 * 91 = 465 ~ 3 frames
