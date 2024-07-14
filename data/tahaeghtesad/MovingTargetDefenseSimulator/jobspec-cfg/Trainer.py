from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from AttackLearner import AttackLearner
from AttackerQExperience import AttackerQExperience
from AttackerNNExperience import AttackerNNExperience
from Defenders import *
from Attackers import *
from Game import Game
import copy
import itertools
import sys

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

rootLogger.setLevel(logging.INFO)

number_of_servers = 3
episodes = 10000
rounds = 100


whole_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=rounds)


def train(i):
    attack_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=rounds)
    game = Game(utenv=1, setting=1, m=number_of_servers, ca=.2, time_limit=rounds)
    attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=1 / math.sqrt(i + 1), train=False)
    defender = UniformDefender(m=number_of_servers, p=10)

    # attacker = MaxProbeAttacker(m=number_of_servers)
    # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

    game.play(attacker, defender)

    # attacker.finalize(i != 0 and i % 100 == 0)
    # defender.finalize(i != 0 and i % 100 == 0)
    rootLogger.info(f'Game {i + 1}/{episodes}: Attacker/Defender:{int(attacker.utility)}/{int(defender.utility)}')
    return attack_exp.retrieve_exp()


def main():
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    pool = Pool(processes)
    for i in tqdm(range(episodes)):
        # if i % 100 == 0:
        #     whole_exp.plot()
        exps = pool.map(train, [i] * processes)
        total = itertools.chain(exps)
        whole_exp.store_exp(total)
        whole_exp.store()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(-1)
