import pickle
import sys
import time

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from pyro.infer import MCMC, NUTS
from tqdm import tqdm

import dataset
import utils
from cvae_utils import parse_args as parse_args_cvae
from eval.models import get_cvae_loss, get_nll, reconstruct_cov, is_est
from eval_cvae import initialize as initialize_cvae
from eval_direct import initialize
from train_utils import parse_args


def check_sampling_convergence(val_dl, model, args):
    # args.batch = 20
    n_examples = 1
    unique, unique_indexes = torch.unique(
        val_dl.dataset.x_exist, return_inverse=True, dim=0
    )
    unique_idx = torch.all(unique, dim=1).nonzero()[0]
    indexes = torch.arange(
        val_dl.dataset.x.shape[0]
    )[unique_idx == unique_indexes]
    indexes = indexes[5:5 + n_examples]

    x, y = val_dl.dataset.x[indexes], val_dl.dataset.y[indexes]
    z = val_dl.dataset.z[indexes]
    nlls = []
    n_samples = [1 * i for i in range(1, 1000, 10)]
    with tqdm(n_samples) as pbar:
        pbar.set_description("Evaluating sampling convergence")
        for j in pbar:
            mean, l_tri_flat, prior_param, rec_param = model(
                x, y, z, n_samples=j, sample_from_prior=True
            )

            prior_param = (
                prior_param[0][:n_examples], prior_param[1][:n_examples]
            )
            rec_param = (rec_param[0][:n_examples], rec_param[1][:n_examples])

            nll = get_cvae_loss(
                x, mean, l_tri_flat, prior_param, rec_param, None, j, args.min_d
            )[:, 0]
            # print(nll)
            nlls.append(nll)

    fig, ax = plt.subplots()
    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    for i in range(n_examples):
        print()
        ax.plot(
            n_samples, torch.stack(nlls)[:, i].detach().numpy(),
            color=colors[i]
        )
    ax.set_xlabel('Number of MC samples')
    ax.set_ylabel('MC estimate of $-\log p(x|y)$')
    fig.tight_layout()
    plt.show()


def check_single_chain_ess(data):
    for i in range(data.shape[0]):
        arviz_data = arviz.convert_to_dataset(data[i:i + 1])
        ess_bulk = arviz.ess(arviz_data)['x'].to_numpy()
        ess_tail = arviz.ess(arviz_data, method='tail')['x'].to_numpy()

        # print(f'chain {i + 1} ess bulk:', ess_bulk)
        # print(f'chain {i + 1} ess tail:', ess_tail, '\n')
        if (ess_bulk < 50).any() or (ess_tail < 50).any():
            return False
    return True


def check_stat(summary, stat, val, geq=True):
    stat_vals = summary[stat].to_numpy()
    if geq:
        return (stat_vals >= val).all()
    else:
        return (stat_vals <= val).all()


def check_mixing(data, min_ess_coefficient=100):
    accepted = check_single_chain_ess(data)
    summary = arviz.summary(
        data, hdi_prob=0.95
    )
    print(summary.to_string())
    num_chains = data.shape[0]
    min_ess = min_ess_coefficient * num_chains
    accepted &= check_stat(
        summary, 'ess_bulk', min_ess
    )
    accepted &= check_stat(
        summary, 'ess_tail', min_ess
    )
    accepted &= check_stat(
        summary, 'r_hat', 1.01, geq=False
    )
    return accepted


def run_hmc(
        model, args, data, cvae, device, sample_idx, num_chains=4,
        num_samples=2000, warmup_steps=2000, marginalize=False, n_samples=50
):
    x, y, z, x_exist = data['x'], data['y'], data['z'], data['x_exist']
    drop_filters = data['drop_filters']
    # print(drop_filters)
    if marginalize:
        mask = []
        for i, (f, idx) in enumerate(dataset.filter_indexes.items()):
            if f not in drop_filters:
                if i in utils.filters_to_keep_hmc:
                    mask.append(True)
                else:
                    mask.append(False)
        mask = torch.logical_and(torch.tensor(mask, device=device), x_exist[0])
        # x = x[:, mask]
    else:
        mask = x_exist[0]
        # x = x[:, x_exist[0]]
    # print(mask)
    # mask[0] = False
    # print(mask)
    y_dim = y.shape[-1]
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(y_dim, device=device), torch.eye(y_dim, device=device)
    )

    def potential_fn(y_dict):
        y_sample = y_dict['y']
        prior_log_p = prior.log_prob(y_sample)
        if cvae:
            # nll = is_est(model, x, y, z, mask, n_samples, args.min_d)

            mean, l_tri_flat, prior_param, rec_param = model(
                x, y_sample, z, n_samples=n_samples, sample_from_prior=True
            )
            n_features = mean.shape[-1]
            mean = mean[:, :, mask]
            x_marg = x[:, mask]
            prior_param = (prior_param[0], prior_param[1])
            rec_param = (rec_param[0], rec_param[1])
            nll, skip_update = get_cvae_loss(
                x_marg, mean, l_tri_flat, prior_param, rec_param, None,
                n_samples, args.min_d, mask=mask, n_features=n_features
            )
            nll = nll[:, 0]
        else:
            mean, l_tri_flat = model(y_sample, z)

            cov = reconstruct_cov(
                l_tri_flat, mean.shape[1], 0.
            )[0]
            cov = cov[:, :, mask][:, mask, :]
            mean = mean[:, mask]
            x_marg = x[:, mask]

            nll = get_nll(
                x_marg, mean, l_tri_flat, args.min_d, False,
                message='direct', covariance_matrix=cov
            )[0]
        return nll - prior_log_p

    init_params = y  # torch.cat([y, y], dim=0)
    samples = []
    start_time = time.time()
    skip_sample = False

    def hook_fn(*arg):
        if time.time() - start_time > 60 * 60 * 3:
            raise ValueError('Time limit exceeded')

    for _ in range(num_chains):
        pyro.clear_param_store()
        nuts_kernel = NUTS(
            potential_fn=potential_fn, adapt_step_size=True,
            adapt_mass_matrix=True, full_mass=True, target_accept_prob=0.65,
            jit_compile=True
        )
        mcmc = MCMC(
            nuts_kernel,
            initial_params={'y': init_params},
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=1,
            disable_progbar=True,
            hook_fn=hook_fn
        )
        try:
            mcmc.run()
        except ValueError:
            print(
                30 * '#' + f' SAMPLE IDX: {sample_idx} ' + 30 * '#'
            )
            print('Time limit exceeded, skipping')
            print(80 * '#', '\n')
            skip_sample = True
            break
        samples.append(mcmc.get_samples()['y'].unsqueeze(0))

    if skip_sample:
        return None, None, None, skip_sample

    runtime = time.time() - start_time
    print(
        30 * '#' + f' SAMPLE IDX: {sample_idx} ' +
        30 * '#'
    )
    data = torch.cat(samples, dim=0).squeeze(-2).detach().cpu().numpy()
    accepted = check_mixing(data)
    mean_pred = data.reshape(-1, y.shape[-1]).mean(axis=0)
    print('runtime:', runtime)
    print('true y:', y)
    print('sample mean:', mean_pred)
    print(
        'sample mean MAE:', np.abs(mean_pred - y.detach().cpu().numpy()).mean()
    )
    print(80 * '#', '\n')

    return data, runtime, accepted, skip_sample


def main(stats_path, args, cvae, marginalize=False, check_convergence=False):
    model, val_dl, labels = initialize(
        args
    ) if not cvae else initialize_cvae(args)

    if cvae and check_convergence:
        check_sampling_convergence(val_dl, model, args)

    print('marginalize:', marginalize)
    device = utils.get_device()
    print('device:', device)
    model.to(device)
    # data_path = sys.argv[1]  # '../data/full/sampled_filters_val.pkl'
    # experiment_path = sys.argv[2]  # './pocs/hmc-stats'
    sample_idx_path = f'{stats_path}/sample_idx.txt'
    y_dim = val_dl.dataset.y.shape[1]
    samples = np.loadtxt(sample_idx_path, dtype=int)
    # samples = samples
    # all_samples = []
    results = []
    hmc_samples_n = [5000]
    for i, sample in enumerate(samples):
        print('starting sample:', sample, f'({i + 1}/{len(samples)})')
        sample_res = {'idx': sample, 'hmc_runs': [], 'runtimes': []}
        y = val_dl.dataset.y[sample]
        data = {
            'x': val_dl.dataset.x[sample].unsqueeze(0).to(device),
            'y': y.unsqueeze(0).to(device),
            'z': val_dl.dataset.z[sample].unsqueeze(0).to(device),
            'x_exist': val_dl.dataset.x_exist[sample].unsqueeze(0).to(device),
            'drop_filters': val_dl.dataset.drop_filters
        }
        for n in hmc_samples_n:
            posterior_samples, runtime, accepted, skip_sample = run_hmc(
                model, args, data, cvae, device, sample,
                num_chains=4,
                num_samples=n,
                warmup_steps=n,
                marginalize=marginalize
            )
            if skip_sample:
                break
            sample_res['hmc_runs'].append(posterior_samples)
            sample_res['runtimes'].append(runtime)
            if accepted:
                break
        results.append(sample_res)
        with open(f'{experiment_path}/results.pkl', "wb") as f_out:
            pickle.dump(results, f_out)


if __name__ == "__main__":
    data_path = sys.argv[1]
    experiment_path = sys.argv[2]
    model_path = sys.argv[3]
    cvae_ = True if sys.argv[4] == 'cvae' else False
    marg = True if sys.argv[5] == 'marginalize' else False
    args_path = f'@{model_path}/args.txt'
    if cvae_:
        main(
            experiment_path,
            parse_args_cvae([data_path, model_path, args_path]), cvae_,
            marg, check_convergence=False
        )
    else:
        main(
            experiment_path, parse_args([data_path, model_path, args_path]),
            cvae_, marg
        )
