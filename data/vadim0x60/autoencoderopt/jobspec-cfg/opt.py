from calendar import c
import os
import requests
from pathlib import Path
import json as j
import itertools
import wandb
import nevergrad as ng
from random import sample
import subprocess
from heapq import heappush, heappushpop
from hashlib import blake2b

COMPILE_SERVER_API = os.environ.get('COMPILE_SERVER_API') or 'https://tree2tree.app/api'
MIN_FITNESS = float('-inf')
CHECKPOINT_INTERVAL = os.environ.get('CHECKPOINT_INTERVAL') or 100
TOP_K = int(os.environ.get('TOP_K') or 10)
LATENT_DIM = int(os.environ.get('LATENT_DIM') or 150)

h = blake2b(digest_size=8)
config = {}
config_str = ''

for config_var, default, constructor in (
    ('MAX_TESTS', '32', int),
    ('RANGE', '6', int),
    ('BUDGET', '10', int),
    ('OPTIMIZER', 'NGOpt', lambda x: ng.optimizers.registry[x]),
    ('TASK', 'fuel-cost', str)
):
    config_var_value = os.environ.get(config_var) or default

    config[config_var] = config_var_value
    globals()[config_var] = constructor(config[config_var])
    if len(config_str) + len(config_var_value) > 31:
        h.update(config_str.encode())
        config_str = config_var_value + '-' + h.hexdigest()
    else:
        config_str = config_var_value + '-' + config_str

datasets_path = Path(__file__).parent / 'datasets'

imports = requests.get(COMPILE_SERVER_API + '/imports').text
class Program():
    def __init__(self, uid, source) -> None:
        self.source_file = solutions_path / f'{uid}.cpp'
        self.binary_file = solutions_path / f'{uid}.bin'
        self.persist = False
        self.uid = uid

        with open(self.source_file, 'w') as f:
            f.write(imports)
            f.write('\n\n')
            f.write(source)

    def __lt__(self, other):
        return self.uid < other.uid

    def __enter__(self):
        completed_process = subprocess.run(['g++', str(self.source_file), '-o', str(self.binary_file)], capture_output=True)
        assert not completed_process.stderr
        return self

    def run(self, input_lines):
        completed_process = subprocess.run([str(self.binary_file)],
                                                input='\n'.join(input_lines).encode(),
                                                capture_output=True)

        assert not completed_process.stderr
        self.stdout, self.stderr = completed_process.stdout.decode(), completed_process.stderr.decode()
        return self.stdout.split('\n')

    def __exit__(self, exc_type, exc_value, trace):
        self.binary_file.unlink()

    def __del__(self):
        if not self.persist:
            self.source_file.unlink()

def correctness(expected_outputs, outputs):
    score = 0
    for expected_line, actual_line in zip(expected_outputs, outputs):
        score += (expected_line == actual_line)
    score /= len(expected_outputs)
    return score

def test_program(program, test_cases):
    test_results = [correctness(output_lines, program.run(input_lines)) for input_lines, output_lines in test_cases]
    return sum(test_results) / len(test_results)

def disnumerate_prefix(data, prefix):
    lines = []
    for line_number in itertools.count(start=1):
        try:
            lines.append(data[f'{prefix}{line_number}'])
        except KeyError:
            break
    return lines

def objects2lines(objects):
    lines = []
    for object in objects:
        if isinstance(object, list):
            # In competitive programming tasks arrays are represented as:
            # $ARRAY_LENGTH
            # $ARRAY_ELEMENT_1, $ARRAY_ELEMENT_2, ..., $ARRAY_ELEMENT_ARRAY_LENGTH
            lines.append(str(len(object)))
            lines.append(' '.join(map(str, object)))
        else:
            lines.append(str(object))
    return lines

def parse_test_case(test_case_str):
    test_case = j.loads(test_case_str)
    input_lines = []
    output_lines = []

    input_lines, output_lines = (objects2lines(disnumerate_prefix(test_case, prefix)) 
                                 for prefix in ('input', 'output'))
    return input_lines, output_lines

def sample_tests(task_name):
    random_cases, edge_cases = ((datasets_path / task_name / f'{task_name}-{test_set}.json').read_text().strip().split('\n')
                                for test_set in ['random', 'edge'])

    selected_cases = (edge_cases + sample(random_cases, MAX_TESTS))[:MAX_TESTS]
    return map(parse_test_case, selected_cases)
    
def decode_vector(vector):
    decoding_request = {
        'latent_vector': vector.tolist(),
        'temperature': 0,
        'top_k': 0,
        'top_p': 0
    }

    decoding_response = requests.post(COMPILE_SERVER_API + '/decode', json=decoding_request)
    assert decoding_response.ok
    return decoding_response.json()['program']

def make_report(optimizer, candidate, fitness):
    global best_fitness
    best_fitness = max(fitness, best_fitness)

    report = {
        "#num-ask": optimizer.num_ask,
        "#num-tell": optimizer.num_tell,
        "#num-tell-not-asked": optimizer.num_tell_not_asked,
        "#lineage": candidate.heritage["lineage"],
        "#generation": candidate.generation,
        "#parents_uids": [],
        "#uid": candidate.uid,
        "#fitness": fitness,
        "#best_fitness": best_fitness
    }  

    if isinstance(candidate._meta.get("sigma"), float):
        report["#meta-sigma"] = candidate._meta["sigma"] 
    if candidate.generation > 1:
        report["#parents_uids"] = candidate.parents_uids
    return report

if __name__ == '__main__':
    wandb.init(project='autoencoderopt', config=config)

    solutions_path = Path(wandb.run.dir)
    optimizer_path = solutions_path / f'{config["OPTIMIZER"]}.pickle'
    optimizer = OPTIMIZER(parametrization=ng.p.Array(shape=(LATENT_DIM,), lower=-RANGE, upper=RANGE), budget=BUDGET)

    best_fitness = MIN_FITNESS
    best_programs = []

    for i in range(TOP_K + 1):
        heappush(best_programs, (MIN_FITNESS, Program(i, str(i))))

    def evaluate_candidate(candidate):
        global best_fitness
        try:
            source = decode_vector(candidate.value)
            with Program(candidate.uid, source) as program:
                test_cases = sample_tests(TASK)
                fitness = test_program(program, test_cases)
                best_fitness = max(fitness, best_fitness)
                heappushpop(best_programs, (fitness, program))
        except AssertionError:
            fitness = MIN_FITNESS

        wandb.log(make_report(optimizer, candidate, fitness))
        return fitness

    def checkpoint(final=False):
        recommendation = optimizer.provide_recommendation()
        fitness = evaluate_candidate(recommendation)
        summary = {}

        for fitness, program in best_programs:
            if fitness > MIN_FITNESS:
                summary[program.uid] = fitness
                if final:
                    program.persist = True

        with open(solutions_path / f'{config["OPTIMIZER"]}.json', 'w') as f:
            j.dump(summary, f)
    
    try:
        for idx in range(optimizer.budget):
            candidate = optimizer.ask()
            fitness = evaluate_candidate(candidate)
            optimizer.tell(candidate, - fitness)
            if idx % CHECKPOINT_INTERVAL == 0:
                checkpoint(final=False)
    finally:
        checkpoint(final=True)