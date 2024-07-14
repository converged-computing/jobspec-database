from model import VideoMAEModel
from clip_sampler import get_frames_from_video_path
from importlib import reload
import av
import json
from tqdm import trange, tqdm
import numpy as np
from experiments import run_experiment
from experiments_runall import run_experiment_all, run_theoretical_best
from utils import get_missclassified
import pickle
import json

np.random.seed(0)
with open('kinetics-dataset/video_paths.pickle', 'rb') as f:
    video_paths = pickle.load(f)
    
with open('data/kinetics400/validate/validate.json') as f:
    annotations_dict = json.load(f)

model = VideoMAEModel()  
# indices = np.array(get_missclassified("outputs/random-num-20-250-1.csv"))

# run_theoretical_best(model, "theoretical-best2", video_paths, indices, seed = 10, num_frames=16, num_examples = 250, outer_batch_size = 20, batch_size = 25)

run_experiment_all(model, "position-all", video_paths, seed = 10, num_frames=4, num_examples = 250, outer_batch_size = 250, batch_size = 25)
run_experiment_all(model, "position-all", video_paths, seed = 20, num_frames=4, num_examples = 250, outer_batch_size = 250, batch_size = 25)