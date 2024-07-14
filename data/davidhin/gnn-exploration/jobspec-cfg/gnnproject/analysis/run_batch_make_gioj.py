"""HPC run gnnproject.helpers.make_graph_input_oj.cpg_to_dgl_from_filepath.

This script turns CPG outputs (nodes.csv, edges.csv) into DGLGraph objects, complete
with the edge connections, initial vertice features (W2V), and edge types.
"""
# %%
import os
import pickle as pkl
import sys
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path

import gnnproject as gp
import gnnproject.helpers.make_graph_input_oj as ggi
import numpy as np
from gensim.models import Word2Vec
from gnnproject.helpers.constants import EDGE_TYPES, EDGE_TYPES_CD
from tqdm import tqdm

# %% SETUP
DATASET = "devign_ffmpeg_qemu"  # Change to other datasets
NUM_JOBS = 2000
JOB_ARRAY_NUMBER = int(sys.argv[1]) - 1
if JOB_ARRAY_NUMBER == 0:
    vari = "cfgdfg"
    etm = EDGE_TYPES
    cfgonly = False
if JOB_ARRAY_NUMBER == 1:
    vari = "cfg"
    etm = EDGE_TYPES_CD
    cfgonly = True
if JOB_ARRAY_NUMBER == 2:
    vari = "cpg"
    etm = EDGE_TYPES
    cfgonly = False

# %% MAKE SPLITS
files = sorted(glob(str(gp.external_dir() / f"{DATASET}/functions/*")))
splits = np.array_split(files, NUM_JOBS)
w2v = Word2Vec.load(str(gp.external_dir() / "w2v_models/devign"))


# %% PROCESS SPLIT
def process_split(split: list):
    """Process list of files sequentially."""
    for f in split:
        savedir = gp.get_dir(gp.processed_dir() / f"{DATASET}_dgl_{vari}")
        filename = Path(f).stem
        if os.path.exists(savedir / str(f"{filename}.pkl")):
            continue
        path = gp.processed_dir() / DATASET / filename
        g = ggi.cpg_to_dgl_from_filepath(path, w2v=w2v, cfgonly=cfgonly, etypemap=etm)
        if not g:
            continue

        while True:
            try:
                savedir /= str(f"{filename}.pkl")
                with open(savedir, "wb") as f:
                    pkl.dump(g, f)
                with open(savedir, "rb") as f:
                    pkl.load(f)
            except Exception as E:
                print(E)
            else:
                break


# %% RUN IN PARALLEL LOCALLY
with Pool(4) as p:
    with tqdm(total=NUM_JOBS) as pbar:
        for _ in p.imap_unordered(process_split, splits):
            pbar.update()


# %% Example Load for Debugging
for s in splits[30]:
    filename = Path(s).stem
    savedir = gp.get_dir(gp.processed_dir() / f"{DATASET}_dgl")
    savedir /= str(f"{filename}.pkl")
    try:
        with open(savedir, "rb") as f:
            g = pkl.load(f)
            print(g)
    except Exception as E:
        gp.debug(E)
