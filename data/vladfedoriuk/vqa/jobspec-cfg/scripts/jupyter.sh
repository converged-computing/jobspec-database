#!/bin/bash

python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install --editable .

jupyter-notebook --no-browser --port=8989
