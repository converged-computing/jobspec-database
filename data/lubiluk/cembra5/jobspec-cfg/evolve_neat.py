from __future__ import print_function

import multiprocessing
import os
import pickle

import neat
import numpy as np
import ray
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation
from ray.util import ActorPool

import visualize
from wrappers import DoneOnSuccessWrapper

runs_per_net = 10

ray.init(address=os.environ.get("ip_head"))

print("Nodes in the Ray cluster: {}".format(len(ray.nodes())))
print("CPUs in the Ray cluster: {}".format(ray.cluster_resources()["CPU"]))
print("----------------")


@ray.remote
class Evaluator:
    def __init__(self) -> None:
        import gym
        import panda_gym

        def make_env():
            return FlattenObservation(
                FilterObservation(
                    DoneOnSuccessWrapper(
                        gym.make("PandaReachDense-v2"), reward_offset=0
                    ),
                    filter_keys=["observation", "desired_goal"],
                )
            )

        self.env = make_env()
        self.act_limit = self.env.action_space.high[0]

    def evaluate(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitnesses = []

        for runs in range(runs_per_net):
            fitness = 0.0
            done = False
            obs = self.env.reset()

            while not done:
                act = net.activate(obs)
                act = np.tanh(act) * self.act_limit
                obs, rew, done, info = self.env.step(act)
                fitness += rew

            fitnesses.append(fitness)

        return min(fitnesses)


num_cpu = int(ray.cluster_resources()["CPU"])
pool = ActorPool([Evaluator.remote() for _ in range(num_cpu)])


def eval_genomes(genomes, config):
    genlist = [g for _, g in genomes]
    fits = list(pool.map(lambda a, v: a.evaluate.remote(v, config), genlist))

    for i in range(len(genlist)):
        genlist[i].fitness = fits[i]


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(eval_genomes)

    # Save the winner.
    with open("data/winner-feedforward", "wb") as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(
        stats, ylog=True, view=True, filename="data/feedforward-fitness.svg"
    )
    visualize.plot_species(stats, view=True, filename="data/feedforward-speciation.svg")

    node_names = None  # {-1: "x", -2: "dx", -3: "theta", -4: "dtheta", 0: "control"}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="data/winner-feedforward.gv",
    )
    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="data/winner-feedforward-enabled.gv",
        show_disabled=False,
    )
    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="data/winner-feedforward-enabled-pruned.gv",
        show_disabled=False,
        prune_unused=True,
    )


if __name__ == "__main__":
    run()
