import json
import os
import sys
import platform
from datetime import datetime

import corner
import numpy as np
from matplotlib import pyplot as plt

from analysis import (
    get_ecw_params,
    get_pta,
    read_data,
)
from plotting import plot_upper_limit
from sampling import create_red_noise_empirical_distr, run_dynesty, run_ptmcmc, read_ptmcmc_chain

def main():
    settings_file = sys.argv[1]
    with open(settings_file, "r") as sf:
        settings = json.load(sf)

    if "--no-sampler" in sys.argv:
        settings["run_sampler"] = False

    data_dir = settings["data_dir"]
    par_file = settings["par_file"]
    tim_file = settings["tim_file"]
    noise_file = settings["noise_dict_file"]

    psr, noise_dict = read_data(
        data_dir,
        par_file,
        tim_file,
        noise_file,
    )

    ecw_params = get_ecw_params(psr, settings)

    # vary_red_noise = settings["vary_red_noise"]
    pta = get_pta(
        psr,
        noise_dict,
        ecw_params,
        noise_only=settings["noise_only"],
        wn_vary=settings["white_noise_vary"],
        rn_vary=settings["red_noise_vary"],
        rn_components=settings["red_noise_nharms"],
    )
    print("Free parameters :", pta.param_names)

    # Make sure that the PTA object works.
    # This also triggers JIT compilation of julia code and the caching of ENTERPRISE computations.
    print("Testing likelihood and prior...")
    x0 = np.array([p.sample() for p in pta.params])
    print("Log-prior at the test point is", pta.get_lnprior(x0))
    print("Log-likelihood at the test point is", pta.get_lnlikelihood(x0))

    output_prefix = settings["output_prefix"]
    jobid = "" if "SLURM_JOB_ID" not in os.environ else os.environ["SLURM_JOB_ID"]
    time_suffix = datetime.now().strftime("%Y-%m-%dT%Hh%Mm")
    outdir = f"{output_prefix}_{jobid}_{time_suffix}/"
    if not os.path.exists(outdir):
        print(f"Creating output dir {outdir}...")
        os.mkdir(outdir)

    summary = get_summary(pta, outdir, settings)
    with open(f"{outdir}/summary.json", "w") as summarypkl:
        print("Saving summary...")
        json.dump(summary, summarypkl, indent=4)

    if settings["run_sampler"]:

        if settings["sampler"] == "ptmcmc":
            rn_ed = create_red_noise_empirical_distr(
                psr, "data/red_noise_empdist_samples.dat"
            )
            red_noise_group = [
                pta.param_names.index(f"{psr.name}_red_noise_gamma"),
                pta.param_names.index(f"{psr.name}_red_noise_log10_A"),
            ]
            gwecc_freq_mass_group = [
                pta.param_names.index("gwecc_log10_F"),
                pta.param_names.index("gwecc_log10_M"),
            ]
            gwecc_proj_group = [
                pta.param_names.index("gwecc_log10_A"),
                pta.param_names.index("gwecc_sigma"),
                pta.param_names.index("gwecc_rho"),
            ]
            run_ptmcmc(
                pta,
                settings["ptmcmc_niter"],
                outdir,
                groups=[red_noise_group],
                empdist=[rn_ed],
            )
            burned_chain = read_ptmcmc_chain(outdir, settings["ptmcmc_burnin_fraction"])
        elif settings["sampler"] == "dynesty":
            burned_chain = run_dynesty(pta, outdir)

        print("")

        if settings["create_plots"]:
            print("Saving plots...")

            ndim = burned_chain.shape[1]

            for i in range(ndim):
                plt.subplot(ndim, 1, i + 1)
                param_name = pta.param_names[i]
                plt.plot(burned_chain[:, i])
                # plt.axhline(true_params[param_name], c="k")
                plt.ylabel(param_name)
            plt.savefig(f"{outdir}/chains.pdf")

            corner.corner(burned_chain, labels=pta.param_names)
            plt.savefig(f"{outdir}/corner.pdf")

            if not settings["noise_only"]:
                plot_param_names = ["gwecc_log10_F", "gwecc_log10_M", "gwecc_e0"]
                plot_param_pltlbl = [
                    "$\\log_{10} f_{gw}$ (Hz)",
                    "$\\log_{10} M$ (Msun)",
                    "$e_0$",
                ]
                plot_param_lims = [(-9, -7), (6, 9), (0.01, 0.8)]
                plot_param_nbins = [16, 8, 8]
                plot_savefiles = [
                    "upper_limit_freq.pdf",
                    "upper_limit_mass.pdf",
                    "upper_limit_ecc.pdf",
                ]

                for pname, pltlbl, plim, pnbin, psavefile in zip(
                    plot_param_names,
                    plot_param_pltlbl,
                    plot_param_lims,
                    plot_param_nbins,
                    plot_savefiles,
                ):
                    plt.clf()
                    plot_upper_limit(
                        pta.param_names,
                        burned_chain,
                        pname,
                        pltlbl,
                        plim,
                        xparam_bins=pnbin,
                        quantile=0.95,
                    )
                    plt.savefig(f"{outdir}/{psavefile}")

    print("Done.")


def get_pta_param_summary(pta):
    def get_prior_summary(param):
        return {"type": param.type, "kwargs": param.prior.func_kwargs}

    return {par.name: get_prior_summary(par) for par in pta.params}


def get_summary(pta, outdir, settings):
    return (
        {
            "user": os.environ["USER"],
            "os": platform.platform(),
            "machine": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        }
        | settings
    ) | {
        "output_dir": outdir,
        "pta_param_summary": get_pta_param_summary(pta),
    }

if __name__ == "__main__":
    main()
