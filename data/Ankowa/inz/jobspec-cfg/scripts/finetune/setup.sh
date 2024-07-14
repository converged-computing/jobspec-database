#!/bin/bash

if test -d ".venv"
then
    echo ".venv already exist"
    source .venv/bin/activate
    pip freeze
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip freeze
    cd diffusers
    pip install .
    pip freeze
    cd examples/text_to_image
    pip install -r requirements.txt 
    pip install xformers
    pip freeze
    cd ../../..
fi