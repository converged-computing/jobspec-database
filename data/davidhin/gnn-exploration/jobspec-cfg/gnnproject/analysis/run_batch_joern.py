# %% LOAD IMPORTS
import sys
from glob import glob
from pathlib import Path

import gnnproject as gp
import gnnproject.helpers.joern as gpj
import numpy as np

# %% SETUP
NUM_JOBS = 200
JOB_ARRAY_NUMBER = int(sys.argv[1]) - 1

# %% MAKE SPLITS
files = glob(str(gp.external_dir() / "devign_ffmpeg_qemu/functions/*"))
splits = np.array_split(files, NUM_JOBS)
savedir = gp.processed_dir() / "devign_ffmpeg_qemu"


# %% PROCESS SPLIT
def process_split(split: list):
    """Run joern on list of files sequentially."""
    for f in split:
        done = [Path(i).stem for i in glob(str(savedir / "*"))]
        if Path(f).stem in done:
            gp.debug("Finished, skipping {}".format(Path(f).stem))
            continue
        gp.debug(
            gpj.run_joern(
                filepath=f,
                dataset_name="devign_ffmpeg_qemu",
                save=True,
                joern_parse="joern-parse",
                joern_export="joern-export",
            )
        )


process_split(splits[JOB_ARRAY_NUMBER])
