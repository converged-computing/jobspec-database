#!/usr/bin/env julia

# This is a script to run snaq! on a CF dataset for hmax=0-4. Run in shell as:
# julia snaq_subset1.jl

# enable multicore
using Distributed
addprocs(9)
# Install and initialize required packages to load CSV
# import Pkg; Pkg.add("CSV")
# import Pkg; Pkg.add("DataFrames")
using CSV, DataFrames
# star PhyloNetworks on all cores
@everywhere using PhyloNetworks
# load the CF table and turn it into a dataframe for SNaQ that only includes taxa names 
# and CF values
dat = DataFrame!(CSV.File("analyses/phylonetworks/snaq/CFtable.csv"))
buckyCF = readTableCF(dat)
writeTableCF(buckyCF)
# load start tree
astraltree = readTopologyLevel1("analyses/phylonetworks/snaq/start_tree_subset1.tre")
# run a 0-hybridization network
net0 = snaq!(astraltree, buckyCF, hmax=0, filename="analyses/phylonetworks/snaq/net0", seed=1234)
# run networks with h=1-4, 10 runs each
net1 = snaq!(net0, buckyCF, hmax=1, filename="analyses/phylonetworks/snaq/net1", seed=2345)
net2 = snaq!(net1, buckyCF, hmax=2, filename="analyses/phylonetworks/snaq/net2", seed=3456)
net3 = snaq!(net2, buckyCF, hmax=3, filename="analyses/phylonetworks/snaq/net3", seed=4567)
net4 = snaq!(net3, buckyCF, hmax=4, filename="analyses/phylonetworks/snaq/net4", seed=5567)
# get scores from each network to select best h
scores = [net0.loglik, net1.loglik, net2.loglik, net3.loglik, net4.loglik]
# export scores to plot in R
scoreDF = DataFrame(h = 0:4, score = scores) # to make a data frame
CSV.write("analyses/phylonetworks/snaq/plog_scores.csv", scoreDF)