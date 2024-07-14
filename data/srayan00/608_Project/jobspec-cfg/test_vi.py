import numpy as np
import ToyEnv
import psrl
from PosteriorSamplerGMM import GibbsSamplerGMM, HMCpymcGMM, VIpymcGMM
import time
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the PSRL algorithm on the ToyEnv environment')
parser.add_argument('--seed', type=int, default=43, help='The seed for the random number generator')
parser.add_argument("--horizon", type=int, default=100, help="The horizon for the PSRL algorithm")
parser.add_argument("--episode", type=int, default=10, help="The time for the PSRL algorithm")

# Seed 1 = 42, Seed 2 = 43, Seed 3 = 44

if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    h= args.horizon
    t = args.episode
    print("Running for H: ", h)
    new_env = ToyEnv.ToyEnv(4, 3, 2, seed)
    with open(f"results/horizons/env_{seed}_horizon{h}.pkl", "wb") as f:
        pickle.dump(new_env.to_dict(), f)

    init_state = new_env.reset()
    st_time_h = time.time()
    print(f"Running HMC Sampler for {h}")
    alg_h = psrl.PSRL(env=ToyEnv.ToyEnv(4, 3, 2, seed), sampler=VIpymcGMM, H = h, T = t)
    _, _, policy_h = alg_h.run()
    end_time_h = time.time()
    with open(f"results/horizons/{seed}_horizon{h}_V.pkl", "wb") as h:
        pickle.dump(alg_h.to_dict(), h)