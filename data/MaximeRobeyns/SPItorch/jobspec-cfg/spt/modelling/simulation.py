# SPItorch: Inference of Stellar Population Properties in PyTorch
#
# Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import logging
import numpy as np

from enum import Enum
from copy import deepcopy
from rich.progress import Progress
from typing import Type

from spt.load_photometry import load_observation, save_sim, \
                                join_partial_results
from multiprocessing import Process, Queue

from spt.config import ForwardModelParams, SamplingParams


class Simulator:

    def __init__(self, fmp: Type[ForwardModelParams] = ForwardModelParams,
                 use_observation: bool = True):

        mp = fmp()
        o = None
        if use_observation:
            # Uses catalogue in InferenceParams
            o = load_observation(filters=mp.filters)

        self.obs = mp.build_obs_fn(mp.filters, o)  # type: ignore
        self.model = mp.build_model_fn(mp.all_params, mp.ordered_params)  # type: ignore
        self.sps = mp.build_sps_fn(**mp.sps_kwargs)  # type: ignore

        self.dim = len(mp.free_params)
        self.phot_dim = len(fmp.filters)

        self.priors = []
        for p in self.model.free_params:
            self.priors.append(self.model.config_dict[p]['prior'])

    def draw_random_theta(self) -> np.ndarray:
        """Draws a theta parameter vector from the priors

        Inexplicably, the priors don't allow you to draw more than one sample
        at a time...

        TODO: remove this loop from the hot path
        """
        # TODO: now that we have torch priors, replace with single sample
        samples = []
        for p in self.priors:
            samples.append(p.sample())
        return np.hstack(samples)

    def simulate_sample(self) -> tuple[np.ndarray, np.ndarray]:
        theta = self.draw_random_theta()
        _, phot, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return theta, phot


class Status(Enum):
    STARTING = 1
    LOADED = 2
    SAMPLED = 3
    SAVING = 4
    DONE = 5


def work_func(idx: int, n: int, q: Queue, sim: Simulator,
              save_dir: str, logging_freq: int = 10) -> None:
    q.put((idx, Status.STARTING, 0))

    # TODO: this seems to take too long to run... GIL?
    # sim = Simulator(fmp, False, sps)

    theta = np.zeros((n, sim.dim))
    phot = np.zeros((n, sim.phot_dim))

    # Force loading SPS libraries
    for i in range(10):
        theta[i], phot[i] = sim.simulate_sample()

    q.put((idx, Status.LOADED, 0))

    for i in range(n):
        theta[i], phot[i] = sim.simulate_sample()
        if i % logging_freq == 1:
            q.put((idx, Status.SAMPLED, i))

    q.put((idx, Status.SAVING, n))

    save_path = os.path.join(save_dir, f'photometry_sim_{n}_{idx}.h5')
    assert isinstance(sim.obs['phot_wave'], np.ndarray)
    save_sim(save_path, theta, sim.model.free_params, phot, sim.obs['phot_wave'])

    q.put((idx, Status.DONE, n))


def main(sp: SamplingParams = SamplingParams()):

    C = sp.concurrency
    N = sp.n_samples // C
    status_q: Queue = Queue()

    logging.info(f'Creating a dataset size {sp.n_samples} across {C} workers')

    if not os.path.exists(sp.save_dir):
        os.makedirs(sp.save_dir)
        logging.info(f'Created results directory {sp.save_dir}')

    logging.info('[bold]Setting up forward model')
    sim = Simulator(ForwardModelParams, sp.observation)

    # t = progress.add_task('[bold]Loading SPS libraries', start=False, total=10)

    # silence all non-error logs:
    log = logging.getLogger()
    l = log.getEffectiveLevel()
    log.setLevel(logging.ERROR)

    with Progress() as progress:

        tasks = []

        for p in range(C):
            Process(target=work_func,
                    args=(p, N, status_q, deepcopy(sim), sp.save_dir)).start()
            tasks.append(progress.add_task(f'[red]Loading  {p:02}', total=N, start=False))

        done = 0
        while done < C:
            (idx, status, n) = status_q.get()
            if status == Status.LOADED:
                progress.reset(tasks[idx], completed=n, start=True,
                                description=f'[green]Running  {idx:02}')
            elif status == Status.SAMPLED:
                progress.update(tasks[idx], completed=n, start=True,
                                description=f'[green]Running  {idx:02}')
            elif status == Status.SAVING:
                progress.update(tasks[idx], completed=n,
                                description=f'[green]Saving   {idx:02}')
            elif status == Status.DONE:
                progress.update(tasks[idx], completed=n,
                                description=f'[white]Done     {idx:02}')
                done += 1
    log.setLevel(l)

    join_partial_results(sp.save_dir, sp.n_samples, sp.concurrency)


if __name__ == '__main__':

    main()

