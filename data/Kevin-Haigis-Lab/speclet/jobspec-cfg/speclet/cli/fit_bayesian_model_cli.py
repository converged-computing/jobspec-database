#!/usr/bin/env python3

"""CLI for standardized fitting Bayesian models."""

from inspect import getdoc
from pathlib import Path
from time import time

import arviz as az
import pandas as pd
from dotenv import load_dotenv
from typer import Typer

import speclet.modeling.posterior_checks as post_check
from speclet import io
from speclet import model_configuration as model_config
from speclet.bayesian_models import BayesianModelProtocol, get_bayesian_model
from speclet.cli import cli_helpers
from speclet.loggers import logger, set_console_handler_level
from speclet.managers.cache_manager import cache_posterior, get_posterior_cache_name
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.managers.data_managers import broad_only as broad_only_filter
from speclet.managers.data_managers import data_transformation
from speclet.model_configuration import ModelingSamplingArguments
from speclet.modeling.fitting_arguments import (
    PymcSampleArguments,
    PymcSamplingNumpyroArguments,
)
from speclet.modeling.model_fitting_api import fit_model
from speclet.pipelines.slurm_interactions import cancel_current_slurm_job
from speclet.project_configuration import project_config_broad_only
from speclet.project_enums import ModelFitMethod

# --- Setup ---


load_dotenv()
cli_helpers.configure_pretty()
app = Typer()


# --- Helpers ---


def _read_crispr_screen_data(
    file: io.DataFile | Path, broad_only: bool
) -> pd.DataFrame:
    """Read in CRISPR screen data."""
    trans: list[data_transformation] = []
    if broad_only:
        trans = [broad_only_filter]
    return CrisprScreenDataManager(data_file=file, transformations=trans).get_data(
        read_kwargs={"low_memory": False}
    )


def _augment_sampling_kwargs(
    sampling_kwargs: ModelingSamplingArguments | None,
    mcmc_chains: int,
    mcmc_cores: int,
) -> ModelingSamplingArguments | None:
    if sampling_kwargs is None:
        sampling_kwargs = ModelingSamplingArguments()

    if sampling_kwargs.pymc_mcmc is not None:
        sampling_kwargs.pymc_mcmc.chains = mcmc_chains
        sampling_kwargs.pymc_mcmc.cores = mcmc_cores
    else:
        sampling_kwargs.pymc_mcmc = PymcSampleArguments(
            chains=mcmc_chains, cores=mcmc_cores
        )

    if sampling_kwargs.pymc_numpyro is not None:
        sampling_kwargs.pymc_numpyro.chains = mcmc_chains
    else:
        sampling_kwargs.pymc_numpyro = PymcSamplingNumpyroArguments(chains=mcmc_chains)

    return sampling_kwargs


def _add_model_attributes(
    model: BayesianModelProtocol, trace: az.InferenceData
) -> None:
    if (posterior := getattr(trace, "posterior")) is None:
        return None
    if (posterior_attrs := getattr(posterior, "attrs")) is None:
        return None
    posterior_attrs["model_name"] = type(model).__name__
    posterior_attrs["model_version"] = model.version
    posterior_attrs["model_doc"] = getdoc(model)
    return None


# --- Automated posterior checks ---


def _automated_posterior_checks(
    trace: az.InferenceData,
    additional_checks: list[post_check.PosteriorCheck] | None = None,
) -> post_check.PosteriorCheckResults:
    checks = [
        post_check.CheckStepSize(min_ss=0.0005),
        post_check.CheckBFMI(min_bfmi=0.2, max_bfmi=2.0),
        post_check.CheckNoMissingDraws(),
    ]
    if additional_checks is not None:
        msg = f"Recieved {len(additional_checks)} posterior checks from the model."
        logger.debug(msg)
        checks += additional_checks
    return post_check.check_mcmc_sampling(trace, checks)


def _handle_posterior_check_results(
    res: post_check.PosteriorCheckResults, cancel_slurm_job: bool
) -> None:
    logger.info(res.message)
    if res.all_passed:
        logger.info("Sampling statistics checks passed.")
        return None

    logger.error("Sampling statistics checks failed.")

    if cancel_slurm_job:
        logger.error("Trying to cancel SLURM job.")
        cancel_current_slurm_job()
        logger.error("Unable to fail SLURM job.")

    raise post_check.FailedSamplingStatisticsChecksError()


# --- Main ---


@app.command()
def fit_bayesian_model(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    mcmc_chains: int = 4,
    mcmc_cores: int = 4,
    cache_name: str | None = None,
    seed: int | None = None,
    broad_only: bool | None = None,
    log_level: str | None = None,
    check_sampling_stats: bool = False,
    cancel_slurm_job: bool = True,
) -> None:
    """Sample a Bayesian model.

    The parameters for the MCMC cores and chains is because I often fit the chains in
    separate jobs to help with memory management.

    Args:
        name (str): Name of the model configuration.
        config_path (Path): Path to the configuration file.
        fit_method (ModelFitMethod): Model fitting method to use.
        cache_dir (Path): Directory in which to cache the results.
        mcmc_chains (int, optional): Number of MCMC chains. Defaults to 4.
        mcmc_cores (int, optional): Number of MCMC cores. Defaults to 4.
        cache_name (Optional[str], optional): A specific name to use for the posterior
        cache ID. Defaults to None which results in using the `name` for the cache name.
        seed (Optional[int], optional): Random seed for models. Defaults to `None`.
        broad_only (bool, optional): Only include Broad screen data. Defaults to
        `False` to include all data.
        log_level (str | int | None, optional): Set a log level. Defaults to `None`.
        check_sampling_stats (bool, optional): Whether to check the sampling statistics
        at the end of sampling. Note, that a failed result will cause the program to
        exit with an error.
        cancel_slurm_job (bool, optional): If can find the SLURM job ID (`SLURM_JOB_ID`
        environment variable), then cancel the job if the posterior checks fail.
        Defaults to `True`.
    """
    tic = time()
    if log_level is not None:
        set_console_handler_level(log_level)
    logger.info("Reading model configuration.")
    config = model_config.get_configuration_for_model(
        config_path=config_path, name=name
    )

    if broad_only is None:
        logger.info("Using project config option for `broad_only`.")
        broad_only = project_config_broad_only()

    assert config is not None
    logger.info("Loading data.")
    data = _read_crispr_screen_data(config.data_file, broad_only=broad_only)
    logger.info("Retrieving Bayesian model object.")
    model = get_bayesian_model(config.model)(**config.model_kwargs)

    logger.info("Augmenting sampling kwargs (MCMC chains and cores).")
    sampling_kwargs_adj = _augment_sampling_kwargs(
        config.sampling_kwargs,
        mcmc_chains=mcmc_chains,
        mcmc_cores=mcmc_cores,
    )
    logger.info("Sampling model.")
    trace = fit_model(
        model=model,
        data=data,
        fit_method=fit_method,
        sampling_kwargs=sampling_kwargs_adj,
        seed=seed,
    )
    logger.info("Sampling finished.")

    logger.info("Adding model attributes.")
    _add_model_attributes(model, trace)

    if (posterior := getattr(trace, "posterior", None)) is None:
        raise AttributeError("Missing posterior draws.")

    print(posterior.data_vars)

    if check_sampling_stats:
        logger.info("Checking sampling stats.")

        model_checks: list[post_check.PosteriorCheck] | None = getattr(
            model, "posterior_sample_checks", lambda: None
        )()
        res = _automated_posterior_checks(trace, model_checks)
        _handle_posterior_check_results(res, cancel_slurm_job=cancel_slurm_job)

    if cache_name is None:
        logger.warning("No cache name provided - one will be generated automatically.")
        cache_name = get_posterior_cache_name(model_name=name, fit_method=fit_method)

    logger.info(f"Caching posterior data: '{str(cache_name)}'")
    cache_posterior(trace, id=cache_name, cache_dir=cache_dir)

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    app()
