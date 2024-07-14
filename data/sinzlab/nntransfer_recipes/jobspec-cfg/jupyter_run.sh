#!/bin/bash
python3 -m venv ./env  --system-site-packages
. ./env/bin/activate
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=env
jupyter lab
