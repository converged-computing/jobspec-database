from __future__ import print_function

import os

import torch
import torch.multiprocessing as mp

from quoridor_ai.optimizers import my_optim
from quoridor_ai.envs import get_env
from quoridor_ai.model import ActorCritic
from quoridor_ai.test import test
from quoridor_ai.train import train
from quoridor_ai.settings import settings

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    torch.manual_seed(settings.seed)
    env = get_env()
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    prev_saves = os.listdir('quoridor_ai/saves')
    if len(prev_saves) > 0:
        float_saves = [float(x) for x in prev_saves]
        final_save = str(max(float_saves))
        shared_model.load_state_dict(torch.load('quoridor_ai/saves/' + final_save))
        settings.init_time = - float(final_save)

        print('INIT TIME IS {}'.format(settings.init_time))
        print('LOADED MODEL {}'.format(final_save))

    if settings.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=settings.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(settings.num_processes, settings, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, settings.num_processes):
        p = mp.Process(target=train, args=(rank, settings, shared_model, counter, lock, optimizer,
                                           settings.steps_per_worker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
