#!/bin/bash

# Loads config

CFGFILE="default.conf" # change to "default.conf"

. "$CFGFILE"

# export the required path variables
for i in "${!PATHS[@]}"
do
    printf "%s \u2190 %s\n" "${i}" "${PATHS[$i]}"
    export "${i}=${PATHS[$i]}"
done