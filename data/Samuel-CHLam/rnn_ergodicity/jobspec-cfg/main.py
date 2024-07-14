import argparse
import pandas as pd
from simulation import simulate_all
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_neurons', type=int, default=100, help='number of neurons in the hidden layer')
parser.add_argument('--max_time', type=int, default=1000, help='number of simulation steps')
parser.add_argument('--n_paths', type=int, default=10, help='number of simulated paths')
parser.add_argument('--rolling_mean', type=bool, default=True, help='if True then return rolling mean, if False then return actual samples')
parser.add_argument('--no_rolling_mean', dest='rolling_mean', action='store_false', help='if flagged then return actual samples')
parser.add_argument('--return_summary', type=bool, default=False, help='if True then return only the summary statistic when rolling_mean=True or last step of memories when rolling_mean=False, otherwise return the history of the rolling mean or memories.')
parser.add_argument('--save_history', type=bool, default=True, help='applicable with --rolling_mean is False, toggle to save/discard history.')
parser.add_argument('--discard_history', dest='save_history', action='store_false', help='applicable with --rolling_mean is False, toggle to discard history.')
parser.add_argument('--individual_kde', type=bool, default=True, help='if True then return individual kernel density for each path, otherwise return one single kernel density for single path')
parser.add_argument('--memorize', type=bool, default=True, help='if True then include memory layer')
parser.add_argument('--store_xz', type=bool, default=False, help='if True then return the input sequence as well')
parser.add_argument('--seed', type=int, default=1234, help='seed for random number generation')
parser.add_argument('--device', type=str, default='cuda', help='device for running simulation CPU/Cuda/MPS')
parser.add_argument('--in_cluster', type=bool, default=True, help='True if running in a cluster')
parser.add_argument('--not_in_cluster', dest='in_cluster', action='store_false', help='flag only if not running in a cluster')

args = parser.parse_args()

if args.in_cluster == True:
    storage_path = '/data/math-hpc-machine-learning/univ5366'
else:
    storage_path = './'

if args.rolling_mean == True:

    if args.return_summary == True:
        output_arr = simulate_all(n_neurons=args.n_neurons, max_time=args.max_time,
                                  n_paths=args.n_paths, rolling_mean=True,
                                  memorize=args.memorize, return_summary=True,
                                  seed=args.seed, use_device=args.device)

        output_df = pd.DataFrame(output_arr, columns=["time_arr",
                                                      "overall_mean_mean_h",
                                                      "overall_mean_std_h",
                                                      "overall_var_mean_h",
                                                      "overall_var_std_h",
                                                      "lower_mean_h",
                                                      "upper_mean_h",
                                                      "union_lower_mean_h",
                                                      "union_upper_mean_h",
                                                      "lower_var_h",
                                                      "upper_var_h",
                                                      "union_lower_var_h",
                                                      "union_upper_var_h"])
        
        output_df.to_csv(storage_path + f"/output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_n_paths_{args.n_paths}_rolling_mean_summary")

    else:
        simulate_avg_h, simulate_avg_h2, time_arr, = simulate_all(n_neurons=args.n_neurons,
                                                            max_time=args.max_time,
                                                            n_paths=args.n_paths,
                                                            rolling_mean=True,
                                                            memorize=args.memorize,
                                                            store_xz=args.store_xz,
                                                            seed=args.seed,
                                                            use_device=args.device)

        output_df = pd.DataFrame()
        output_df["time_arr"] = time_arr

        output_df["overall_mean_mean_h"] = simulate_avg_h.mean(axis=(1,2)).to(device="cpu").numpy()
        output_df["overall_mean_std_h"] = simulate_avg_h.std(axis=(1,2)).to(device="cpu").numpy()
        output_df["overall_var_mean_h"] = simulate_avg_h2.mean(axis=(1,2)).to(device="cpu").numpy()
        output_df["overall_var_std_h"] = simulate_avg_h2.std(axis=(1,2)).to(device="cpu").numpy()

        output_df["lower_mean_h"] = simulate_avg_h.mean(axis=1).min(axis=1).values.to(device="cpu").numpy()
        output_df["upper_mean_h"] = simulate_avg_h.mean(axis=1).max(axis=1).values.to(device="cpu").numpy()
        output_df["union_lower_mean_h"] = (simulate_avg_h.mean(axis=1) - 2 * simulate_avg_h.std(axis=1)).min(axis=1).values.to(device="cpu").numpy()
        output_df["union_upper_mean_h"] = (simulate_avg_h.mean(axis=1) + 2 * simulate_avg_h.std(axis=1)).max(axis=1).values.to(device="cpu").numpy()

        output_df["lower_var_h"] = simulate_avg_h2.mean(axis=1).min(axis=1).values.to(device="cpu").numpy()
        output_df["upper_var_h"] = simulate_avg_h2.mean(axis=1).max(axis=1).values.to(device="cpu").numpy()
        output_df["union_lower_var_h"] = (simulate_avg_h2.mean(axis=1) - 2 * simulate_avg_h2.std(axis=1)).min(axis=1).values.to(device="cpu").numpy()
        output_df["union_upper_var_h"] = (simulate_avg_h2.mean(axis=1) + 2 * simulate_avg_h2.std(axis=1)).max(axis=1).values.to(device="cpu").numpy()
        
        output_df.to_csv(storage_path + f"/output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_n_paths_{args.n_paths}_rolling_mean")

else:
    if args.individual_kde:
        simulate_avg_h = simulate_all(n_neurons=args.n_neurons, max_time=args.max_time,
                                        n_paths=args.n_paths, rolling_mean=False,
                                        memorize=args.memorize, store_xz=args.store_xz,
                                        return_summary = False, save_history = False, 
                                        seed=args.seed, use_device=args.device).T

        kde_positions = np.linspace(0,1,500)
        individual_kde = np.zeros((args.n_paths, 500))

        for path in tqdm(range(args.n_paths)):
            kernel = gaussian_kde(simulate_avg_h[path])
            individual_kde[path] = kernel(kde_positions)

        individual_df = pd.DataFrame(individual_kde.T)
        individual_df.to_csv(storage_path + f"/output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_n_paths_{args.n_paths}_final_step_kde_individual")

        all_kernel = gaussian_kde(simulate_avg_h.reshape(-1))
        all_kde = all_kernel(kde_positions)
        all_df = pd.Series(all_kde)
        all_df.to_csv(storage_path + f"/output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_n_paths_{args.n_paths}_final_step_kde_all")
    
    else:
        simulate_avg_h = simulate_all(n_neurons=args.n_neurons, max_time=args.max_time,
                                        n_paths=args.n_paths, rolling_mean=False,
                                        memorize=args.memorize, store_xz=args.store_xz,
                                        return_summary = args.return_summary, save_history = False, 
                                        seed=args.seed, use_device=args.device)

        output_df = pd.Series(simulate_avg_h)
        output_df.to_csv(storage_path + f"/output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_n_paths_{args.n_paths}_final_step_kde")

