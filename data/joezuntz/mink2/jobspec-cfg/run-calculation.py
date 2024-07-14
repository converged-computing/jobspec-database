import pyccl
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

from coefficient import * 
coefficient(thr_ct=10, smoothing=20, nside=1024, itr=100, b=10)
