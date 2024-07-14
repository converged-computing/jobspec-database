import os
import subprocess
import sys

import numpy as np
import ray
import torch
import torch.nn as nn
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation
from ray.util import ActorPool
from pickle import dump
from sklearn.cluster import AgglomerativeClustering

from wrappers import DoneOnSuccessWrapper


class Model(nn.Module):
    def __init__(self, input_dim, hidden_sizes, out_dim, activation, act_limit):
        super().__init__()
        self.act_limit = act_limit

        layers = [nn.Linear(input_dim, hidden_sizes[0], True), activation()]

        for i in range(len(hidden_sizes) - 1):
            layers += [
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], True),
                activation(),
            ]

        layers += [nn.Linear(hidden_sizes[-1], out_dim, bias=True)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.act_limit

    def genotype(self, g=None):
        if g is not None:
            params = g

            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                w_shape = layer.weight.data.shape
                w_shape_flat = w_shape[0] * w_shape[1]
                b_shape = layer.bias.data.shape
                b_shape_flat = b_shape[0]

                w = params[:w_shape_flat]
                params = params[w_shape_flat:]
                layer.weight.data = w.reshape(w_shape).float()
                b = params[:b_shape_flat]
                params = params[b_shape_flat:]
                layer.bias.data = b.float()
        else:
            params = torch.tensor([])
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                w = layer.weight.data.flatten()
                b = layer.bias.data.flatten()
                f = torch.cat([w, b])
                params = torch.cat([params, f])

            return params


@ray.remote
class Evaluator:
    def __init__(self, render=False) -> None:
        import gym
        import panda_gym
        from gym.envs.registration import make

        self.render = render

        def make_env():
            return FlattenObservation(
                FilterObservation(
                    DoneOnSuccessWrapper(
                        gym.make("PandaReachDense-v2", render=self.render),
                        reward_offset=0,
                    ),
                    filter_keys=["observation", "desired_goal"],
                )
            )

        self.env = make_env()
        # Evaluate only a few first steps
        self.env.env.env.env._max_episode_steps = evaluation_steps
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_limit = self.env.action_space.high[0]
        self.model = Model(
            input_dim=obs_dim,
            hidden_sizes=[32, 32],
            out_dim=act_dim,
            activation=nn.ReLU,
            act_limit=act_limit,
        )

    def evaluate(self, genotype):
        with torch.no_grad():
            self.model.genotype(genotype)
            rets = []
            for _ in range(evaluation_episodes):
                obs = self.env.reset()
                done = False
                ret = 0
                while not done:
                    action = self.model.forward(torch.from_numpy(obs)).numpy()
                    obs, rew, done, _ = self.env.step(action)
                    if self.render:
                        self.env.render()
                    ret += rew
                rets.append(ret)

            return sum(rets) / len(rets)

    def genome_shape(self):
        return self.model.genotype().shape

    def state_dict(self):
        return self.model.state_dict()


def mutate(genome):
    copy = genome.clone()
    mutation_prob = 0.02  # from 0.01 to 0.03
    max_weight = max(copy)
    mutation_mag = max_weight / 256 # dynamic, every population, per individual
    indxs = torch.rand_like(copy) < mutation_prob
    # waga = waga +/- [0.5 , 1.0] * mutation_mag
    sign = 1 if torch.rand(1)[0] < 0.5 else -1
    copy[indxs] += sign * (torch.rand_like(copy[indxs]) / 2 + 0.5) * mutation_mag
    # saving the sign

    return copy


evaluation_steps = 10
evaluation_episodes = 10
species_count = 20
population_size = 100
hidden_sizes = [32, 32]

# The more individuals in population, the better
# No crossover


def distance(genome1, genome2):
    return torch.norm(genome2 - genome1)


stagnation_time = 0

def extract_clusters(X, y):
    s = np.argsort(y)
    return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])

def cluster_species(population):
    clustering = AgglomerativeClustering(n_clusters=species_count)
    species_indxs = clustering.fit_predict(population)
    species = extract_clusters(population, species_indxs)

    for i in range(len(species)):
        n = len(species[i])
        if n < population_size:
            species[i] = torch.cat([species[i], species[i][-1].unsqueeze(dim=0).repeat(population_size - n,1)])
        else:
            species[i] = species[i][:population_size]

        assert(len(species[i]) == population_size)

    return species

if __name__ == "__main__":
    if "darwin" in sys.platform:
        print("Running 'caffeinate' on MacOSX to prevent the system from sleeping")
        subprocess.Popen("caffeinate")

    ray.init(address=os.environ.get("ip_head"))

    print("Nodes in the Ray cluster: {}".format(len(ray.nodes())))
    print("CPUs in the Ray cluster: {}".format(ray.cluster_resources()["CPU"]))
    print("----------------")

    # def make_env():
    #     import gym
    #     import panda_gym
    #     from gym.envs.registration import make
    #     return FlattenObservation(
    #         FilterObservation(
    #             DoneOnSuccessWrapper(
    #                 gym.make("PandaReachDense-v2", render=False),
    #                 reward_offset=0,
    #             ),
    #             filter_keys=["observation", "desired_goal"],
    #         )
    #     )

    # env = make_env()

    with torch.no_grad():
        dummy_eval = Evaluator.remote()
        genome_shape = ray.get(dummy_eval.genome_shape.remote())
        genome = torch.zeros(genome_shape)
        del dummy_eval

        num_cpu = int(ray.cluster_resources()["CPU"])
        pool = ActorPool([Evaluator.remote() for _ in range(num_cpu)])

        # Cluster into species
        population = torch.rand(population_size * species_count, genome.shape[0])
        species = cluster_species(population)

        for generation in range(5000):
            all_fit = torch.tensor([])
            for i in range(len(species)):
                s_pop = species[i]
                fitness_remotes = pool.map(lambda a, v: a.evaluate.remote(v), s_pop)
                fitness = torch.tensor(list(fitness_remotes))

                # No crossover

                # Leave 10% elite unchanged
                _, topind = torch.topk(fitness, int(population_size * 0.1)) 
                elite = s_pop[topind]

                # mutate to fill in the population
                _, topind = torch.topk(fitness, population_size - len(elite)) 
                mutants = torch.stack([mutate(g) for g in s_pop[topind]])

                s_pop = torch.cat((elite, mutants), 0)
                assert(len(s_pop) == population_size)

                species[i] = s_pop
                all_fit = torch.concat([all_fit, fitness])

            # Show the best
            topfit, topind = torch.topk(all_fit, 1)
            print("Best fitness: {}".format(topfit.squeeze()))
            # evaluate(population[topind.squeeze()], viz_env, render=True)

            # Every now and then reculster species
            if generation % 100 == 0:
                population = torch.cat(species, dim=0)
                species = cluster_species(population)
