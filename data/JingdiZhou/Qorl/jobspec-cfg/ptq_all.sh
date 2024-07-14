#!/bin/bash

set -e

if [ -z "$6" ] && [ -z "$7" ]; then
    for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 6 5 4 3 2 1; do
    echo "quantifying model $3"
    python new_ptq.py --algo $1 --env $2 --quantized $i  -f $3 --optimize-choice $4
    done
    python collate_model.py --algo $1 --env $2 --device cuda  -f quantized/ --no-render --exp-id $5 -P
else
    for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 6 5 4 3 2 1; do
    echo "quantifying model $3"
    python new_ptq.py --algo $1 --env $2 --quantized $i  -f $3 --optimize-choice $4
    done
    python collate_model.py --algo $1 --env $2 --device cuda  -f quantized/ --no-render --exp-id $5 --track --rho $6 --learning-rate $7 --optimize-choice $4 -P
fi
