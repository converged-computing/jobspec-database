#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:11:28 2020

@author: hugo
"""

#### package imports #####

import simuPOP as sim
import random
import numpy as np
import argparse


#### set some variables ####

parser = argparse.ArgumentParser()
parser.add_argument("--chrom_loci", help="Number of loci per chromosome separated by comma", 
                    default = "303,195,233,184,268")
parser.add_argument("--n_selected_loci", help="Number of selected loci", type = int,
                    default = 10)
parser.add_argument("--place_selected_loci", help="How to place selected loci: random, equal",
                    default = "equal")
parser.add_argument("--loc_selected_loci", help="Manually specify the location of selected loci separated by comma",
                    default = None)
parser.add_argument("--selected_effect", help="Effect of selected alleles",
                    default = 0.5)
parser.add_argument("--n_adv_alleles", help="Number of advantageous alleles",
                    type = int, default = 1)
parser.add_argument("--outdir", help="Output directory",
                    default = ".")
parser.add_argument("--suffix", help="Output file name suffix",
                    default = ".")
parser.add_argument("--seed", help="Seed number for random number generator. Negative values will pick a random seed.",
                    type = int, default = -1)
args = parser.parse_args()


# set the seeds
if args.seed >= 0:
    sim.setOptions(seed = args.seed)
    random.seed(a = args.seed)


#### Helper functions ####

def qtrait(geno):
    """
    Assign quantitative trait based on genotype

    Parameters
    ----------
    geno : list
        simuPop genotype.

    Returns
    -------
    trait : numeric
        Trait value.
    n_adv : numeric
        Number of advantageous alleles.

    """
    
    # set up
    adv_alleles = [x for x in range(args.n_adv_alleles)] # which alleles increase trait
    relative_effect = float(args.selected_effect)  # effect of each added allele
    
    # check number of advantageous alleles
    n_adv = sum([allele in adv_alleles for allele in geno])
        
    # sample from a normal distribution - simple additive effect
    trait = random.normalvariate(n_adv*relative_effect, 1)
    
    return (trait, n_adv)


def rank(x):
    """
    Rank values in a list. Ties get assigned consecutive ranks. 
    See https://stackoverflow.com/a/57021330/5023162

    Parameters
    ----------
    x : list
        List of numeric values to rank.

    Returns
    -------
    r : list
        List of ranks.

    """
    
    n = len(x)
    t = list(range(n))
    s = sorted( t, key=x.__getitem__ )

    r = s.copy()
    for i,k in enumerate(s):
        r[k] = t[i]

    return r


def assignFitness(pop):
    
    # list to store ranks
    r = []
    
    # assign ranks for each selection type (sub-population)
    for selection in pop.subPopNames():
        # get indexes regarding the respective sub-population
        subpop_idx = pop.subPopByName(selection)
        start_idx = pop.subPopBegin(subpop_idx)
        end_idx = pop.subPopEnd(subpop_idx)
        
        # fetch trait values for those individuals
        trait = list(pop.indInfo('trait')[start_idx:end_idx])
        
        # assign ranks accordingly
        if selection == "directional":
            # add ranked trait values
            r = r + rank(trait)
        elif selection == "random":
            # add ranks based on shuffled trait values
            random.shuffle(trait)
            r = r + rank(trait)
        elif selection == "stabilising":
            # rank by absolute difference to mean
            trait = trait - np.mean(trait) # center on mean
            trait = np.abs(trait) # absolute value
            trait = trait * -1 # negative values (to rank in descending order)
            r = r + rank(trait)
        else:
            raise Exception("We should have never gotten here... this is a bug")
    
    # assign fitness
    for i in range(pop.popSize()):
        if r[i] >= 160:
            pop.individual(i).fitness = 1
        else:
            pop.individual(i).fitness = 0
        
        pop.individual(i).rank = r[i]
    
    return True



#### Initialise population ####

# parse number of loci per chromosome
loci_per_chrom = [int(i) for i in args.chrom_loci.split(",")]

# chromosome names
chrom_names = ["Chr" + str(i+1) for i in range(len(loci_per_chrom))]

# make names for each locus (easier to export data)
loci_names = []
for i in zip(chrom_names, loci_per_chrom):
    for j in range(i[1]):
        loci_names.append(i[0] + "-" + str(j + 1))

# advantageous loci
if args.loc_selected_loci is not None:
    adv_loci = [int(i) for i in args.loc_selected_loci.split(",")]
elif args.place_selected_loci == "random":
    adv_loci = [random.randint(0, len(loci_names)) for x in range(args.n_selected_loci)]
elif args.place_selected_loci == "equal":
    adv_loci_per_chrom = [len(x) for x in np.array_split(range(args.n_selected_loci), 5)]
    
    adv_loci = []
    for i in range(len(loci_per_chrom)):
        if adv_loci_per_chrom[i] == 0:
            next
        else:
            if i != 0:
                offset = sum(loci_per_chrom[:i])
            else:
                offset = 0
            for j in range(adv_loci_per_chrom[i]):
                adv_loci.append(offset + int(loci_per_chrom[i]/(adv_loci_per_chrom[i]+1)*(j+1)))
    #adv_loci = [int(i) for i in np.round(np.linspace(250, len(loci_names)-250, args.n_selected_loci))]
else:
    raise Exception("Argument --place_selected_loci has to be 'random' or 'equal'")

adv_loci.sort()


# initialise population
pop = sim.Population(
    size = [200, 200, 200], # number of individuals
    subPopNames = ["random", "directional", "stabilising"],
    loci = loci_per_chrom,  # number of loci per chromosome
    chromNames = chrom_names,
    lociNames = loci_names,
    infoFields = ['ind_id', 'father_id', 'mother_id', 'trait', 'nalleles', 'rank', 'fitness'],  # all the fields we will use
    ancGen = -1,                                 # record information for all generations
    )


#### evolve population ####

# evolve population
pop.evolve(
    # initial population
    initOps = [
        # individuals are randomly assigned as male or female
        sim.InitSex(),
        # each locus has 19 alleles with equal frequencies
        sim.InitGenotype(freq = [1/19]*19)
        ],
    # before mating rank individuals by their trait
    #preOps = sim.PyOperator(func = pickParents, param = args.selection_type),
    preOps = sim.PyOperator(func = assignFitness),
    # Random mating with fixed number of offspring
    # https://bopeng.github.io/simuPOP/userGuide_ch6_sec1.html#determine-the-number-of-offspring-during-mating
    matingScheme = sim.RandomMating(
        ops = [
            # average recombination across all chromosomes
            sim.Recombinator((1*len(pop.numLoci()))/sum(pop.numLoci())), # avg of 2 recombinations == poisson with rate 2
            sim.PedigreeTagger(),
            sim.IdTagger()
            ],
        # each cross results in 10 offspring
        numOffspring = 10
        ),
    # after mating assign quantitative trait based on genotype
    postOps = [
        sim.PyQuanTrait(
            func = qtrait,
            loci = adv_loci,
            infoFields = ['trait', 'nalleles'])
        ],
    # 11 generations (the first generation are the founders)
    gen = 11
)


#### save results ####

def write_pedigree(pop, outfile):
    
    out = open(outfile, "w")
    
    # get info field names
    info_names = pop.infoFields()
    
    # write a header
    header = "selection,gen,{}\n".format(",".join(info_names))
    out.write(header)
    
    # iterate generations: they are from last to first, so using a decreasing range
    for i_gen in range(pop.ancestralGens()-1, -1, -1):
        # activate this generation
        pop.useAncestralGen(i_gen)
        
        # generation number
        gen = pop.ancestralGens()-i_gen-1
        
        for selection in pop.subPopNames():
            # get indexes regarding the respective sub-population
            subpop_idx = pop.subPopByName(selection)
            start_idx = pop.subPopBegin(subpop_idx)
            end_idx = pop.subPopEnd(subpop_idx)
            
            # write values for each inividual
            for ind_idx in range(start_idx, end_idx):
                ind = pop.individual(ind_idx)
                vals = ",".join([str(ind.info(x)) for x in info_names])
                out.write(selection + "," + str(gen) + "," + vals + "\n")


def expected_heterozygosity(freq, pool_alleles):
    if pool_alleles >= len(freq):
        return 0
    else:
        # sort frequencies
        freq.sort(reverse = True)
        
        # get alleles to pool together
        freq1 = sum(freq[:pool_alleles])
        freq2 = freq[pool_alleles:]
        
        # expected heterozygosity
        het = 1 - (freq1*freq1 + sum([x*x for x in freq2]))
        return het

def write_heterozygosity(pop, outfile):
    
    out = open(outfile, "w")
    out.write("selection,gen,locus,selected,het,het2,het3,het4,het5,het6,het7,het8\n")
    
    # iterate generations: they are from last to first, so using a decreasing range
    for i in range(pop.ancestralGens()-1, -1, -1):
        # activate this generation
        pop.useAncestralGen(i)
        
        # calculate allele frequencies
        # note: the vars option needs to be added so frequences are calculated for each sub-population
        # http://simupop.sourceforge.net/manual_svn/build/userGuide_ch5_sec11.html#defdicttype
        sim.stat(pop, alleleFreq = range(np.sum(pop.numLoci())), vars=['alleleFreq_sp'])
        #sim.stat(pop, alleleFreq = range(np.sum(pop.numLoci())))

        # loop through populations
        for selection in pop.subPopNames():
            # get sub-population index
            subpop_idx = pop.subPopByName(selection)
            
            # loop through each locus
            for locus in range(np.sum(pop.numLoci())):
                # fetch allele frequency from this sub-population and calculate heterozygosity
                # http://simupop.sourceforge.net/manual_svn/build/userGuide_ch5_sec11.html#defdicttype
                #het = expected_heterozygosity(pop.dvars(subpop_idx).alleleFreq[locus].values())
                
                # write all but heterozygosity
                out.write("{a},{b},{c},{d}".format(a = selection,
                                                   b = pop.ancestralGens()-i-1,
                                                   c = pop.locusName(locus),
                                                   d = locus in adv_loci))
                
                # write heterozygosity
                for pool_alleles in range(1, 9):
                    freqs = list(pop.dvars(subpop_idx).alleleFreq[locus].values())
                    het = expected_heterozygosity(freqs, pool_alleles)
                    out.write(",{}".format(het))
                
                # add newline
                out.write("\n")
                

def write_frequency(pop, outfile):
    
    out = open(outfile, "w")
    out.write("selection,generation,locus,freq,selected\n")
    
    # iterate generations: they are from last to first, so using a decreasing range
    for i in range(pop.ancestralGens()-1, -1, -1):
        # activate this generation
        pop.useAncestralGen(i)
        
        # calculate allele frequencies
        # note: the vars option needs to be added so frequences are calculated for each sub-population
        # http://simupop.sourceforge.net/manual_svn/build/userGuide_ch5_sec11.html#defdicttype
        sim.stat(pop, alleleFreq = range(np.sum(pop.numLoci())), vars=['alleleFreq_sp'])
        #sim.stat(pop, alleleFreq = range(np.sum(pop.numLoci())))

        # loop through populations
        for selection in pop.subPopNames():
            # get sub-population index
            subpop_idx = pop.subPopByName(selection)
            
            # loop through each locus
            for locus in range(np.sum(pop.numLoci())):
                # fetch allele frequency from this sub-population and calculate heterozygosity
                # http://simupop.sourceforge.net/manual_svn/build/userGuide_ch5_sec11.html#defdicttype
                for freq in pop.dvars(subpop_idx).alleleFreq[locus].values():
                    # write
                    out.write("{a},{b},{c},{d},{e}\n".format(a = selection,
                                                             b = pop.ancestralGens()-i-1,
                                                             c = pop.locusName(locus),
                                                             d = freq,
                                                             e = locus in adv_loci))


# file_suffix = "{a}-{b}-{c}-seed{d}".format(a = args.n_selected_loci,
#                                                b = args.selected_effect,
#                                                c = args.n_adv_alleles,
#                                                d = "random" if args.seed < 0 else args.seed)
# save pedigree
write_pedigree(pop, 
               "{outdir}/pedigree_{suffix}.csv".format(outdir = args.outdir,
                                                       suffix = args.suffix))

# save allele frequency
write_frequency(pop,
                "{outdir}/freq_{suffix}.csv".format(outdir = args.outdir,
                                                    suffix = args.suffix))

write_heterozygosity(pop,
                     "{outdir}/het_{suffix}.csv".format(outdir = args.outdir,
                                                        suffix = args.suffix))


# =============================================================================
# # save heterozygosity statistics
# write_heterozygosity(pop,
#                      "het_{a}-{b}-{c}.csv".format(a = args.n_selected_loci,
#                                                  b = args.selected_effect,
#                                                  c = args.n_adv_alleles))
# 
# =============================================================================



# =============================================================================
# # evolve population - no selection, no linkage
# pop.evolve(
#     # initial operations
#     initOps = [
#         # individuals are randomly assigned as male or female
#         sim.InitSex(),
#         # each locus has 19 alleles with equal frequencies
#         sim.InitGenotype(freq = [1/19]*19)
#         ],
#     # Random mating with fixed number of offspring
#     # https://bopeng.github.io/simuPOP/userGuide_ch6_sec1.html#determine-the-number-of-offspring-during-mating
#     matingScheme = sim.RandomMating(
#         ops = [
#             sim.MendelianGenoTransmitter(),
#             sim.PedigreeTagger(),
#             sim.IdTagger()
#             ],
#         numOffspring = 10
#         ),
#     # 11 generations (the first generation are the founders)
#     gen = 11
# )
# 
# =============================================================================

# =============================================================================
# # evolve population - no selection, linkage
# pop.evolve(
#     # initial operations
#     initOps = [
#         # individuals are randomly assigned as male or female
#         sim.InitSex(),
#         # each locus has 19 alleles with equal frequencies
#         sim.InitGenotype(freq = [1/19]*19)
#         ],
#     # Random mating with fixed number of offspring
#     # https://bopeng.github.io/simuPOP/userGuide_ch6_sec1.html#determine-the-number-of-offspring-during-mating
#     matingScheme = sim.RandomMating(
#         ops = [
#             sim.Recombinator(rates = 2/pop.numLoci()[0]), # avg of 2 recombinations == poisson with rate 2
#             sim.PedigreeTagger(),
#             sim.IdTagger()
#             ],
#         numOffspring = 10
#         ),
#     # 11 generations (the first generation are the founders)
#     gen = 11
# )
# 
# =============================================================================

















