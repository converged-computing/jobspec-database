#!/bin/bash
set -o nounset

function get_rettype() {
    local fun=$1
    local ARGSTR=$(cat $LLVMDIR/llvm-project/libc/src/math/${fun}.h | grep "(*)") #| cut -d " " -f1
    ARGSTR=${ARGSTR//long long/longlong}
    ARGSTR=${ARGSTR//long int/longint}
    ARGSTR=${ARGSTR//long double/longdouble}
    local RETTYPE=$(echo $ARGSTR | cut -d " " -f1)
    RETTYPE=${RETTYPE//longlong/long long}
    RETTYPE=${RETTYPE//longint/long int}
    RETTYPE=${RETTYPE//longdouble/long double}
    echo $RETTYPE
}