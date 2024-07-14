#!/bin/bash -e

CFGFILE="default.conf"

. "$CFGFILE"

for i in "${!PATHS[@]}"
do
    printf "export %s=\"%s\"\n" "${i}" "${PATHS[$i]}"
    export "${i}=${PATHS[$i]}"
done