#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yangyangli
@contact: li002252@umn.edu
@version: 0.0.1
@license: MIT Licence
@file: simulator.py
@time: 04/07/2021 19:52
"""
from pathlib import Path
import pandas as pd
import yaml

import rpy2.robjects as robjects
import logging


def get_logger(logger_name, create_file=False):
    """
    :param logger_name: logger
    :param create_file: if create log file
    :return: logger
    """
    # create logger for prd_ci
    log = logging.getLogger(logger_name)

    log.setLevel(level=logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if create_file:
        # create file handler for logger.
        fh = logging.FileHandler(f"{logger_name}_log.txt")

        fh.setLevel(level=logging.DEBUG)

        fh.setFormatter(formatter)

    # reate console handler for logger.
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if create_file:
        log.addHandler(fh)

    log.addHandler(ch)

    return log


def simulator(
    simulatedir,
    event="ir",
    max_genes=10,
    seq_depth=2e06,
    readlen=50,
    chrom=1,
    ncores=8,
    seed=42,
):
    """
    :param simulatedir: directory of simulator
    :param event: type of splicing event
    :param max_genes: numbers of events
    :param seq_depth: sequence depth of reads
    :param readlen: length of reads
    :param chrom:  chromosome of simulated data
    :param ncores: numbers of threads
    :param seed: random seed
    :return: directory of simulated data
    """
    as_events = ("es", "ir", "a3", "a5", "mes", "mee", "ale", "afe")
    event_prob = list(map(lambda x: 1 if x == event else 0, as_events))

    parameters = f"""
    # Note: contains the chromosomes fasta file and the genome annotation (in gtf or gff3 format)
    input = "{simulatedir}/{chrom}" #path to the input folder. 
    output = "{simulatedir}/out" #path to the output folder
    max_genes =  {max_genes}
    seq_depth  =  {seq_depth}

    seed = {seed} #seed for a reproducible simulation
    ncores = {ncores} #number of cores for the parallel simulation
    multi_events_per_exon = F #T - allow multiple events per exon; F - allow only one event per exon
    probs_as_freq = F #T - use fixed frequencies (should be summed up to 1); F - use probabilities
    error_rate = 0.001 #sequencing error rate. 0.001 = 0.1%
    readlen = {readlen} #read length
    num_reps = c(1,1) #number of samples and groups. E.g., c(1,1) - two groups with 1 sample each

    #distribution of events. One event per gene; equal distribution
    as_events = c('es','ir','a3','a5','mes','mee','ale','afe')
    #event_probs = rep(1/(length(as_events) + 1), length(as_events))
    # event_probs = c(0,1,0,0,0,0,0,0)
    event_probs = c{tuple(event_prob)}
    names(event_probs) = as_events

    
    ##other examples---
    #one event per gene; custom distribution
    #event_probs = setNames(c(0.060, 0.078, 0.098, 0.049, 0.022, 0.004), c('a5', 'a3', 'es', 'ir', 'mes', 'mee'))
    
    #two events per gene; equal distribution
    #as_events = c('es','ir','a3','a5','mes','mee','ale','afe')
    #as_combs = combn(as_events, 2, FUN = function(...) paste(..., collapse = ','))
    #event_probs = rep(1/(length(as_combs) + 1), length(as_combs))
    #names(event_probs) = as_combs
    
    #custom combinations of events; custom distribution
    #event_probs = setNames(c(0.060, 0.078, 0.098), c('a5,a3', 'a3,es,ir', 'es,mes'))
    

    #name of the output folder
    outdir = sprintf(
        '%s/%s_maxGenes%d_SeqDepth%g_errRate%g_readlen%d_multiEventsPerExon%s_probsAsFreq%s',
        output,
            gsub('[ ]+?', '-', Sys.time()),
        ifelse(is.null(max_genes), 0, max_genes),
        seq_depth,
        error_rate,
        readlen,
        multi_events_per_exon,
        probs_as_freq
      )

    params = list(
      seed = seed,
      ncores = ncores,
      input_dir = input,
      event_probs = event_probs,
      outdir = outdir,
      seq_depth = seq_depth,
      max_genes = max_genes,
      error_rate = error_rate,
      readlen = readlen,
      multi_events_per_exon = multi_events_per_exon,
      probs_as_freq = probs_as_freq,
      num_reps = num_reps)

    library(ASimulatoR)
    do.call(simulate_alternative_splicing, params) #simulate RNA-Seq datasets

    ### copy this script to output folder for reproducibility ----
    # rscript = sub('--file=', '', commandArgs()[4], fixed = T)
    # bn_rscript = basename(rscript)
    # file.copy(rscript, file.path(outdir, bn_rscript)) 
    """
    # print(parameters)
    robjects.r(parameters)
    outdir = robjects.r.outdir
    return outdir[0]


def creat_samples(path, config_path):
    """
    :param path: path of simulated data
    :param config_path: path of config file
    :return: None
    """
    files = "sim_rep_info.txt"  # file path of "sim_rep_info.txt"
    path = Path(path)

    sample = pd.read_csv(path / files, sep="\t")
    sample["fq1"] = sample.rep_id.map(lambda x: f"{str(path)}/{x}_1.fastq")
    sample["fq2"] = sample.rep_id.map(lambda x: f"{str(path)}/{x}_2.fastq")
    df = sample.loc[:, ["rep_id", "fq1", "fq2", "group"]].rename(
        {"rep_id": "samples"}, axis=1
    )
    df.to_csv(config_path, index=False, sep="\t")


if __name__ == "__main__":
    config = yaml.safe_load(open("config/config.yaml"))
    simulator_config = config["simulator"]
    logger = get_logger("simulator")

    outdir = simulator(
        simulatedir=simulator_config["wkdir"],
        event=simulator_config["event"],
        max_genes=simulator_config["max_genes"],
        seq_depth=simulator_config["seq_depth"],
        readlen=simulator_config["readlen"],
        chrom=simulator_config["chrom"],
        seed=simulator_config["seed"],
    )

    logger.info(f"Finishing simulator\n {simulator_config}\n{outdir}")

    creat_samples(outdir, "config/samples.tsv")
    logger.info(f"Creating samples.tsv")
