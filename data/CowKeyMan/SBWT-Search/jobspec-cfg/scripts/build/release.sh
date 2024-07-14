#!/bin/bash

# Build the release version of the main executable for the target platform. It
# takes a single argument which is one of NVIDIA, AMD or CPU. If any other
# argument (any sequence of characters is accepted) is given besides these 3,
# it will skip the cmake step and run the build step only.

if [ $# -ne 1 ]; then
  echo "Usage: ./scripts/build/release.sh <NVIDIA|AMD|CPU|[other]>"
  exit 1
fi

mkdir -p build
cd build
if [ "${1,,}" = nvidia ] || [ "${1,,}" = amd ] || [ "${1,,}" = cpu ];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_MAIN=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=OFF \
    -DENABLE_MARCH_NATIVE=OFF \
    -DHIP_TARGET_DEVICE="$1" \
    -DROCM_BRANCH="rocm-5.4.x" \
    ..
  if [ $? -ne 0 ]; then >&2echo "Cmake generation failed" && cd .. && exit 1; fi
fi
cmake --build . -j8
if [ $? -ne 0 ]; then >&2 echo "Build" && cd .. && exit 1; fi
cd ..
