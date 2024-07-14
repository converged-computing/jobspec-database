#!/bin/bash
# Application-level GPU Power Measurement using NVIDIA-SMI
# Usage ./power_measure.sh "nvidia-smi ..." "application ..."

echo "Starting NVIDIA-SMI..."
($1) &
nvpid=$!
echo "Sleeping for a while..."
sleep 10
echo "Launching application..."
($2) &
appid=$!
wait "$appid"
echo "Application exited, sleeping for a while..."
sleep 10
echo "Ending NVIDIA-SMI..."
kill "$nvpid"
echo "Making sure CSV dump is completed, sleeping again..."
sleep 60