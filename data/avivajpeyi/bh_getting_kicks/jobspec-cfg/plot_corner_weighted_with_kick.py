# -*- coding: utf-8 -*-
"""Reweight results based on kick distribution

We know kick v ~ 200km/s +/- 50km/s

We can re-weight our posterior based on this

Calculate the prob(kick) based on the above distribution

re-weight our posterior by np.random.choice(posterior, fraction=0.1, weights=prob(kick_for_sample))

Example usage:

"""
import argparse
import logging
import os
import sys

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from bilby.gw import conversion
from matplotlib import rcParams
from scipy.stats import norm

from bbh_simulator.corner_kwargs import CORNER_KWARGS, VIOLET_COLOR, BILBY_BLUE_COLOR

rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20

logging.getLogger().setLevel(logging.INFO)

KICK_WEIGHT = "kick_weight"

PARAMS = dict(
    #mass_1_source=r'$m_1^{source}$',
    #mass_2_source=r'$m_2^{source}$',
    #remnant_mass=r'$m_{rem}^{source}$',
    chi_p=r'$\chi_p$',
    #chi_eff=r'$\chi_{eff}$',
    mass_ratio=r'$q$',
    #tilt_1=r'$tilt_1$',
    #tilt_2=r'$tilt_2$',
    #luminosity_distance=r'$d_l$',
    remnant_kick_mag=r'$|\vec{v}_k|\ $km/s'
)


def parse_cli_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='Reweight posterior based on true kick')
    parser.add_argument('--samples-csv', type=str, required=True,
                        help='A CSV file of the posterior')
    parser.add_argument('--kick-mean', type=float, required=False,
                        help='The mean value of the true kick distribution (im km/s)')
    parser.add_argument('--kick-stddev', type=float, required=False,
                        help='The mean value of the true kick distribution (im km/s)')
    parser.add_argument('--true-file', type=str, required=False,
                        help='CSV File with the True Vals of the parameters')
    parser.add_argument('--true-idx', type=int, required=False,
                        help='Row idx of the true val')

    parsed_args = parser.parse_args(args)

    return parsed_args


class Samples:
    def __init__(self, samples_csv, kick_mean, kick_sigma, truths):
        self.truths = truths
        self.kick_mean = kick_mean
        self.kick_sigma = kick_sigma
        self.posterior = self.load_posterior(samples_csv)
        self.add_new_kick_distribution_weights(kick_mean, kick_sigma)
        print(self.posterior[["remnant_kick_mag", KICK_WEIGHT]].describe())

    @staticmethod
    def load_posterior(result_file):
        """Loads posterior from result file"""
        logging.info(f"Loading {result_file}")
        samples = pd.read_csv(result_file)
        samples = samples.dropna()
        samples = conversion.generate_component_spins(samples)
        try:
            assert len(samples.remnant_kick_mag) > 0
        except Exception as e:
            raise ValueError(
                f"Error: {e}. "
                f"There are no `remnant_kick_mag` posterior. "
                f"The posterior present are for {list(samples.columns.values)}")
        return samples

    def get_reweighted_posterior(self):
        return self.posterior.sample(
            frac=0.1,
            weights=KICK_WEIGHT,
            replace=False
        )

    def add_new_kick_distribution_weights(self, kick_mean, kick_sigma):
        logging.info(f"Adding kick weights based on N({kick_mean}, {kick_sigma})")
        kick_prior = norm(loc=kick_mean, scale=kick_sigma)
        self.posterior[KICK_WEIGHT] = np.abs(
            np.exp(kick_prior.logpdf(self.posterior.remnant_kick_mag)))

    def get_plotting_kwargs(self):
        corner_kwargs = CORNER_KWARGS.copy()
        corner_kwargs.update(dict(labels=list(PARAMS.values())))
        if len(self.truths) > 0:
            corner_kwargs.update(dict(truths=[self.truths[k] for k in PARAMS.keys()]))
        return corner_kwargs

    def plot_corner(self, f, title, weights=False):
        assert f[-4:] == ".png", f"{f} is not a png"
        logging.info(f"Plotting {f}")
        s = self.posterior[list(PARAMS.keys())]
        corner_kwargs = self.get_plotting_kwargs()
        corner_kwargs.update(dict(label_kwargs=dict(labelpad=20, fontsize=30)))
        if weights:
            corner_kwargs.update(dict(color=VIOLET_COLOR))
            corner.corner(s, weights=self.posterior.kick_weight, **corner_kwargs)
        else:
            corner_kwargs.update(dict(color=BILBY_BLUE_COLOR))
            corner.corner(s, **corner_kwargs)
        plt.suptitle(f"{title}", fontsize=30, fontdict=dict(color='darkblue'))
        plt.savefig(f)
        plt.close()

    def plot_overlaid_corner(self, f):
        s1 = self.posterior.copy()[list(PARAMS.keys())]
        self.reweighted_posterior = self.get_reweighted_posterior()
        s2 = self.get_reweighted_posterior()[list(PARAMS.keys())]
        range = get_range(s1)

        normalising_weights_s2 = np.ones(len(s2)) * len(s1) / len(s2)
        corner_kwargs = self.get_plotting_kwargs()
        corner_kwargs.update(dict(range=range))
        corner_kwargs.update(dict(label_kwargs=dict(labelpad=50, fontsize=30)))
        fig = corner.corner(s1, **corner_kwargs)
        corner_kwargs.update(dict(color=VIOLET_COLOR))
        corner.corner(s2, fig=fig, weights=normalising_weights_s2, **corner_kwargs)

        orig_line = mlines.Line2D([], [], color=BILBY_BLUE_COLOR,
                                  label="Original Posterior")
        weighted_line = mlines.Line2D([], [], color=VIOLET_COLOR,
                                      label=r"Reweighted Posterior".format(
                                          get_normal_string(int(self.kick_mean),
                                                            int(self.kick_sigma))
                                      ))
        true_line =  mlines.Line2D([], [], color='tab:orange', label='Injected Value')
        plt.legend(handles=[orig_line, weighted_line, true_line], fontsize=25, frameon=False,
                   bbox_to_anchor=(1, len(PARAMS)), loc="upper right")
        plt.savefig(f)
        plt.close()

    def get_quantiles_width(self, param):
        quantiles = self.get_plotting_kwargs()['quantiles']
        s = self.posterior[[param]]
        original_qtles = corner.quantile(x=s, q=quantiles)
        self.reweighted_posterior = self.get_reweighted_posterior()
        rw = self.reweighted_posterior[[param]]
        reweighted_qtles = corner.quantile(x=rw, q=quantiles)
        return dict(
            original=original_qtles[1] - original_qtles[0],
            reweighted=reweighted_qtles[1] - reweighted_qtles[0],
        )


def get_truth_values(truth_csv, truth_idx):
    return pd.read_csv(truth_csv).iloc[truth_idx].to_dict()


def validate_cli_args(parsed_args):
    assert os.path.isfile(
        parsed_args.samples_csv), \
        f"Result {parsed_args.samples_csv} is not a csv file that can be accessed"

    if parsed_args.true_file is not None:
        assert parsed_args.true_idx is not None, \
            "A idx for the true values must be passed"
        assert os.path.isfile(
            parsed_args.true_file), f"True file {parsed_args.true_file} cant be accessed"


def combine_images_horizontally(fnames, f):
    images = [Image.open(x) for x in fnames]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(f)


def process_sample(samples, kick_mean, kick_sigma, f):
    fname1 = f.replace(".dat", "_no_reweighting_corner.png")
    samples.plot_corner(f=fname1, title="No Reweigting")

    fname2 = f.replace(".dat",
                       f"_kick_mu{int(kick_mean)}_sigma{int(kick_sigma)}_corner.png")
    samples.plot_corner(f=fname2, weights=True,
                        title=r"Reweighted with {}".format(
                            get_normal_string(
                                int(kick_mean),
                                int(kick_sigma)
                            )
                        ))

    fname3 = f.replace(".dat", "_overlaid.png")
    samples.plot_overlaid_corner(fname3)

    fname4 = f.replace(".dat", "_corner.png")
    combine_images_horizontally([fname3, fname1, fname2], f=fname4)


def get_normal_string(mean, sigma):
    return r"$\mathcal{{N}}(\mu={},\sigma={})$".format(mean, sigma)


def get_range(samples):
    return [[samples[l].min(), samples[l].max()] for l in samples]


def main():
    args = parse_cli_args()
    validate_cli_args(args)
    if args.true_file:
        truths = get_truth_values(args.true_file, args.true_idx)
        args.kick_mean = truths["remnant_kick_mag"]
        args.kick_sigma = 50
    else:
        truths = {}
    samples = Samples(samples_csv=args.samples_csv, kick_mean=args.kick_mean,
                      kick_sigma=args.kick_sigma, truths=truths)
    process_sample(samples, kick_mean=args.kick_mean,
                   kick_sigma=args.kick_sigma, f=args.samples_csv)


def test():
    truths = pd.read_csv('datafiles/injections.csv').iloc[11].to_dict()
    samples = Samples(samples_csv='injection_11_0_posterior_samples_with_kicks.dat',
                      kick_mean=truths['remnant_kick_mag'],
                      kick_sigma=50, truths=truths)
    print(samples.get_quantiles_width(param='chi_p'))


if __name__ == "__main__":
    main()
