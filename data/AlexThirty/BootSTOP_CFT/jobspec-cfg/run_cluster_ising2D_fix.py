from ast import Param
import sys
from parameters_ising2D import ParametersIsing2D_SAC
from environment.blocks_ising2D import Ising2D_SAC
from environment.data_z_sample import ZData
import environment.utils as utils
from neural_net.sac import soft_actor_critic
from pro_get_params import get_grid_Ising2D_params
import argparse
import os
import ray
import itertools
import numpy as np
import time

sigma=True
guessings, guess_sizes, starts = get_grid_Ising2D_params(sigma=sigma)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--faff_max', type=int, default=10000, help='Maximum number of steps without improving')
    parser.add_argument('--pc_max', type=int, default=10, help='Maximum number of reinitializations before reducing window')
    parser.add_argument('--window_rate', type=float, default=0.7, help='Rate of search window reduction')
    parser.add_argument('--max_window_exp', type=int, default=25, help='Maximun number of window reductions')
    parser.add_argument('--same_spin_hierarchy', type=bool, default=True, help='Whether same spin deltas should be ordered')
    parser.add_argument('--dyn_shift', type=float, default=0., help='Minimum distance between same spin deltas')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate for actor network')
    parser.add_argument('--beta', type=float, default=0.0005, help='Learning rate for critic and value network')
    parser.add_argument('--reward_scale', type=float, default=0.001, help='The reward scale, also related to the entropy parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for cumulative reward')
    parser.add_argument('--tau', type=float, default=0.0005, help='Tau parameter for state-value function update')
    parser.add_argument('--layer1_size', type=int, default=256, help='Dense units for first layer')
    parser.add_argument('--layer2_size', type=int, default=256, help='Dense units for the second layer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--max_cpus', type=int, default=450, help='Maximum number of CPUs')
    parser.add_argument('--cpus_per_job', type=int, default=1, help='Maximum number of CPUs per job')
    parser.add_argument('--runs_per_args', type=int, default=100, help='Number of runs for each combination of parameters')
    
    args = parser.parse_args()
    
    
    res_path = f'/data/trenta/results_Ising_fix_new'
    res_path_steps = f'/data/trenta/results_Ising_fix_new_steps'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    if not os.path.exists(res_path_steps):
        os.makedirs(res_path_steps)
    
    ray.init(address='172.16.18.254:6379', _node_ip_address="172.16.18.254")
    print("Connected to Ray cluster.")
    print(f"Available nodes: {ray.nodes()}")

    assert ray.is_initialized(), "Error in initializing ray."

    @ray.remote(num_cpus=args.cpus_per_job)
    def run_exp(func,
                max_window_changes,
                window_decrease_rate,
                pc_max,
                file_name,
                file_name_steps,
                array_index,
                lower_bounds,
                search_window_sizes,
                guessing_run_list,
                environment_dim,
                search_space_dim,
                faff_max,
                starting_reward,
                x0,
                agent_config,
                verbose,
                best_teor):
        soft_actor_critic(func=func,
                      max_window_changes=max_window_changes,
                      window_decrease_rate=window_decrease_rate,
                      pc_max=pc_max,
                      file_name=file_name,
                      file_name_steps=file_name_steps,
                      array_index=array_index,
                      lower_bounds=lower_bounds,
                      search_window_sizes=search_window_sizes,
                      guessing_run_list=guessing_run_list,
                      environment_dim=environment_dim,
                      search_space_dim=search_space_dim,
                      faff_max=faff_max,
                      starting_reward=starting_reward,
                      x0=x0,
                      agent_config=agent_config,
                      verbose=verbose,
                      best_teor=best_teor)
    
    blocks = utils.generate_Ising2D_block_list(10, [], sigma=sigma)
    remaining_ids = []
    for i in range(guessings.shape[0]):
        run_config = {}
        run_config['faff_max'] = args.faff_max
        run_config['pc_max'] = args.pc_max
        run_config['window_rate'] = args.window_rate
        run_config['max_window_exp'] = args.max_window_exp
        run_config['same_spin_hierarchy'] = args.same_spin_hierarchy
        run_config['dyn_shift'] = args.dyn_shift
        run_config['reward_scale'] = args.reward_scale
    
        agent_config = {}
        agent_config['alpha'] = args.alpha
        agent_config['beta'] = args.beta
        agent_config['reward_scale'] = args.reward_scale
        agent_config['rew_scale_schedule'] = 0
        agent_config['gamma'] = args.gamma
        agent_config['tau'] = args.tau
        agent_config['layer1_size'] = args.layer1_size
        agent_config['layer2_size'] = args.layer2_size
        agent_config['batch_size'] = args.batch_size
        agent_config['output_steps'] = 1
        agent_config['mean_output_k'] = 100

        params = ParametersIsing2D_SAC(run_config, sigma=sigma)
        
        params.guessing_run_list_deltas = guessings[i]
        params.guess_sizes_deltas = guess_sizes[i]
        
        delta_start = starts[i]
        ope_start = params.shifts_opecoeffs
        
        params.global_best = np.concatenate((delta_start, ope_start))
        
        params.guessing_run_list = np.concatenate((params.guessing_run_list_deltas,
                                                   params.guessing_run_list_opes))
        params.guess_sizes = np.concatenate((params.guess_sizes_deltas, params.guess_sizes_opes))
        
        for j in range(args.runs_per_args):
            # ---Instantiating some relevant classes---
            zd = ZData()

            # ---Kill portion of the z-sample data if required---
            zd.kill_data(params.z_kill_list)

            # ---Load the pre-generated conformal blocks for long multiplets---
            #blocks = utils.generate_block_list(max(params.spin_list), params.z_kill_list)

            # ---Instantiate the crossing_eqn class---
            cft = Ising2D_SAC(blocks, params, zd)

            teor_reward = cft.best_theoretical()
            # array_index is the cluster array number passed to the console. Set it to zero if it doesn't exist.
            array_index = 100*i+j
            
            # form the file_name where the code output is saved to
            file_name = os.path.join(res_path, params.filename_stem + str(array_index) + '.csv')
            file_name_steps = os.path.join(res_path_steps, params.filename_stem + str(array_index) + '_steps.csv')
            output = str(array_index)
            utils.output_to_file(file_name=file_name, output=output)
            
            # determine initial starting point in the form needed for the soft_actor_critic function
            x0 = params.global_best - params.shifts
            print(f'Starting run {100*i+j}')
            

            remaining_ids.append(run_exp.remote(func=cft.crossing,
                        max_window_changes=params.max_window_exp,
                        window_decrease_rate=params.window_rate,
                        pc_max=params.pc_max,
                        file_name=file_name,
                        file_name_steps=file_name_steps,
                        array_index=array_index,
                        lower_bounds=params.shifts,
                        search_window_sizes=params.guess_sizes,
                        guessing_run_list=params.guessing_run_list,
                        environment_dim=zd.env_shape,
                        search_space_dim=params.action_space_N,
                        faff_max=params.faff_max,
                        starting_reward=params.global_reward_start,
                        x0=x0,
                        agent_config=agent_config,
                        verbose=params.verbose,
                        best_teor=teor_reward))
            
            time.sleep(1)
        
    n_jobs = len(remaining_ids)
    print(f"Total jobs: {n_jobs}")
    #print(remaining_ids)

    # wait for jobs
    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
        for result_id in done_ids:
            # There is only one return result by default.
            result = ray.get(result_id)
            n_jobs -= 1
            print(f'Job {result_id} terminated.\nJobs left: {n_jobs}')
