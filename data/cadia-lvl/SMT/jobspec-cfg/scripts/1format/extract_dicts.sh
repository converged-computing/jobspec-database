#!/bin/bash
# e=fail on pipeline, u=fail on unset var, x=trace commands
set -ex

# Extracts all dictionaries ending with .tsv in data/raw/dictionary and writes their contents to language separted files in data/formatted/dictionary
# Example
# extract_dicts.sh

SRC_DIR=$1
TRG_DIR=$2

function extract_dict() {
  # Extracts the columns of a .tsv file and writes them to separate files with the headers as extension.
  # Example
  # head -n 2 data/raw/dictionary/wiki.tsv 
  # en      is
  # why     af hverju 
  # extract_dict.sh data/raw/dictionary/wiki.tsv target/dir
  # Writes target/dir/wiki.en and target/dir/wiki.en
  SRC=$1
  TRG_DIR=$2

  # Read the extensions from the headers
  EXTENSIONS=($(head -n 1 "$SRC"))
  FILENAME=$(basename "$SRC")
  NAME="${FILENAME%.*}"

  for i in "${!EXTENSIONS[@]}"; do 
    TRG_FILE="$TRG_DIR"/"$NAME"."${EXTENSIONS[$i]}"
    cut -f $((i + 1)) "$SRC" > "$TRG_FILE" 
    # Remove the header
    sed -i '1d' "$TRG_FILE"
  done
}

# Get all the .tsv files
for f in "$SRC_DIR"/*.tsv; do
  extract_dict "$f" "$TRG_DIR"
done