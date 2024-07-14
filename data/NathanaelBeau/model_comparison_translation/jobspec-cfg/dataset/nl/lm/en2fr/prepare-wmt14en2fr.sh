#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-fren.tar"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.fr-en"
    "commoncrawl.fr-en"
    "un/undoc.2000.fr-en"
    "training/news-commentary-v9.fr-en"
    "giga-fren.release2.fixed"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=fr
lang=en-fr
prep=wmt14_en_fr
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done

gunzip giga-fren.release2.fixed.*.gz
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

pwd

echo "get dev set"
for l in $src $tgt; do
    cp $orig/dev/newstest2013.$l $tmp/valid.$l
done

for l in $src $tgt; do
    awk '{ print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done


TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250

for L in $src $tgt; do
    cp $tmp/bpe.valid.$L $prep/valid.$L
    cp $tmp/bpe.test.$L $prep/test.$L
done

pwd

for L in 'train' 'valid' 'test'; do
    echo "merge source and target files for training LM to ${L}"
    python merge_src_tgt.py $prep/$L.en $prep/$L.fr $prep/$L.merge_en_fr
done

MERGE_FILE_TRAIN=train.merge_en_fr
echo "build dictionnary for ${L}"
python build_dictionnary.py $prep/$MERGE_FILE_TRAIN

cp $prep/$MERGE_FILE_TRAIN.json $prep/vocabulary.json

for f in train.merge_en_fr valid.merge_en_fr test.merge_en_fr; do
    echo "transform file into ids to ${f}..."
    python tokenized_to_ids.py $prep/vocabulary.json $prep/$f $prep/$f.ids.gz
done