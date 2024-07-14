#!/bin/bash

mkdir -p build
cd build
# build compilation commands
# cmake \
#   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#   -DCMAKE_BUILD_TYPE=Debug \
#   -DKSEQPP_READ_BUILD_TESTS=ON \
#   -DKSEQPP_READ_BUILD_STATIC=OFF \
#   -DKSEQPP_READ_BUILD_BENCHMARKS=ON \
#   ..
# build tests in debug mode
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKSEQPP_READ_BUILD_TESTS=ON \
  -DKSEQPP_READ_BUILD_STATIC=OFF \
  -DKSEQPP_READ_BUILD_BENCHMARKS=OFF \
  ..
cmake --build . -j8
# build benchmarks in release mode
# cmake \
#   -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DKSEQPP_READ_BUILD_TESTS=OFF \
#   -DKSEQPP_READ_BUILD_STATIC=ON \
#   -DKSEQPP_READ_BUILD_BENCHMARKS=ON \
#   ..
# cmake --build . -j8
cd ..


# download benchmark_objects
mkdir -p benchmark_objects
cd benchmark_objects

# Download Fasta
FASTA="FASTA.fna"
## Source: https://www.ncbi.nlm.nih.gov/projects/genome/guide/human/index.shtml
wget -nc "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_genomic.fna.gz" -O "${FASTA}.gz"
if test -f "${FASTA}"; then
  echo "${FASTA} already unzipped, skipping "
else
  echo "Unzipping ${FASTA}"
  gzip -dkf "${FASTA}.gz" > "${FASTA}"
fi

# Download Fastq
FASTQ="FASTQ.fnq"
## Source: https://www.ebi.ac.uk/ena/browser/view/PRJEB32631
wget -nc "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR340/004/ERR3404624/ERR3404624_1.fastq.gz" -O "${FASTQ}.gz"
if test -f "${FASTQ}"; then
  echo "${FASTQ} already unzipped, skipping "
else
  echo "Unzipping ${FASTQ}"
  gzip -dkf "${FASTQ}.gz" > "${FASTQ}"
fi

cd ..
