import argparse
from cmath import log
import os
import csv
from csv import writer

from dataProcess import read1PDPTW
from exactModelsCplex import solve1PDPTW_MIP_CPLEX
from Agent import RLAgent, RLAgent_repair, ALNSAgent
from utils import float_to_str, dotdict
from multiprocessing import Pool
from datetime import datetime

from tqdm import tqdm
import time


import config as c
config = c.config()

def to_csv(output_dict, path):
    """
    output_dict (dict) : dictionary object that stores experiment outputs.
    path (str)         : path to save the csv file.

    """
    contents_names = ['instance', 'solution', 'cost', 'solve_time', 'status']
    output = []
    for (key, val) in output_dict.items():
        tmp = []
        try:
            for content in contents_names:
                tmp.append(val[content])
            output.append(tmp)
        except:
            print("broken json file", val)
            pass

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(contents_names)
        writer.writerows(output)


def task(items, timeLimit):
    print(items[1])
    instance = read1PDPTW(os.path.join(items[0], items[1]))
    soln, cost, solve_time, status = solve1PDPTW_MIP_CPLEX(instance, timeLimit=timeLimit)
    result = {'filename':items[1], 'soln':soln, 'cost':cost, 'solve_time':solve_time, 'status':status}

    return result

def runMIPall(*args):
    args = args[0]

    dataset_name = args.dataset_name
    batch_num = args.batch_num
    
    filenames = []
    solutions = []
    costs = []
    solve_times = []
    status_all = []
    dataset_path = os.path.join('/home/liucha90/scratch/1666project', 'data', dataset_name, 'INSTANCES', batch_num)

    result_dir = os.path.join('.', 'results', 'experiment', dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    result_filename = '{}_rp{}_{}_{}'.format(
                'mip',
                'none',
                dataset_name,
                batch_num
                )
    csv_path = os.path.join(result_dir, '{}.csv'.format(result_filename))
    csv_all_path = os.path.join(result_dir, '{}_all.csv'.format(result_filename))

    items = []
    for file in os.listdir(dataset_path):
        if not file.startswith('.'):
            items.append((dataset_path,file))


    # with Pool() as pool:
    #     for result in pool.map(task, items):
    for item in tqdm(items):
        start = time.time()
        result = task(item)    
        end = time.time()
        filenames.append(result['filename'])
        solutions.append(result['soln'])
        solve_times.append(end-start)
        status_all.append(result['status'])
        costs.append(result['cost'])
        # print(result)

        # print(list(result.values()))
        with open(csv_path, 'a', newline='') as f:
            writer_object = writer(f)
            csv_line = [datetime.fromtimestamp(end), result['filename'], [int(x) for x in result['soln']], result['cost'], end-start, result['status']]
            writer_object.writerow(csv_line)
            f.close()
           		    
    feasible_num = sum([1 for s in status_all if s in ['feasible', 'optimal']])
    feasible_rate = feasible_num / len(status_all)

    output_dict = {}
    for i, (file, soln, t, status, cost) in enumerate(zip(filenames, solutions, solve_times, status_all, costs)):
        data = {'instance'  : file, 
                'solution'  : [int(x) for x in soln],
                'cost'      : cost,
                'solve_time': t,
                'status'    : status,
                }
        output_dict[i] = data

   
    to_csv(output_dict, csv_all_path)

def runMIPall_nobatch(dataset_name, timeLimit):    
    filenames = []
    solutions = []
    costs = []
    solve_times = []
    status_all = []
    dataset_path = os.path.join('/Users/chang/PhD_workplace/MIE1666/Project', 'data', dataset_name, 'INSTANCES')

    result_dir = os.path.join('.', 'results', 'experiment', dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    result_filename = '{}_rp{}_{}_all_1min'.format(
                'mip',
                'none',
                dataset_name
                )
    csv_path = os.path.join(result_dir, '{}.csv'.format(result_filename))
    csv_all_path = os.path.join(result_dir, '{}_all.csv'.format(result_filename))

    items = []
    for file in os.listdir(dataset_path):
        if not file.startswith('.'):
            items.append((dataset_path,file))


    # with Pool() as pool:
    #     for result in pool.map(task, items):
    for item in tqdm(items):
        start = time.time()
        result = task(item, timeLimit)    
        end = time.time()
        filenames.append(result['filename'])
        solutions.append(result['soln'])
        solve_times.append(end-start)
        status_all.append(result['status'])
        costs.append(result['cost'])
        # print(result)

        # print(list(result.values()))
        with open(csv_path, 'a', newline='') as f:
            writer_object = writer(f)
            csv_line = [datetime.fromtimestamp(end), result['filename'], [int(x) for x in result['soln']], result['cost'], end-start, result['status']]
            writer_object.writerow(csv_line)
            f.close()
           		    
    feasible_num = sum([1 for s in status_all if s in ['feasible', 'optimal']])
    feasible_rate = feasible_num / len(status_all)

    output_dict = {}
    for i, (file, soln, t, status, cost) in enumerate(zip(filenames, solutions, solve_times, status_all, costs)):
        data = {'instance'  : file, 
                'solution'  : [int(x) for x in soln],
                'cost'      : cost,
                'solve_time': t,
                'status'    : status,
                }
        output_dict[i] = data

   
    to_csv(output_dict, csv_all_path)


runMIPall_nobatch('1PDPTW_generated_d31_i200_tmin300_tmax500_sd2022_test', 60)
runMIPall_nobatch('1PDPTW_generated_d51_i200_tmin300_tmax500_sd2022_test', 60)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     # Model
#     parser.add_argument("--dataset_name", type=str, default='1PDPTW_generated_d21_i1000_tmin300_tmax500_sd2022_test')
#     # parser.add_argument("--batch_num", type=str, default='batch2')

#     args, remaining = parser.parse_known_args()

#     runMIPall(args)