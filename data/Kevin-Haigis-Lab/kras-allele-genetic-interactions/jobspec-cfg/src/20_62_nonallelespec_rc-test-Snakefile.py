
# Snakefile for running the Row-Column test
# for co-mutation and mutual exclusivity

from pathlib import Path
import os

# permutation numbers
NUM_PERMS = 10000
PERMUTATIONS = list(range(1, NUM_PERMS + 1))


data_dir = 'data/rc-test-nonallelespec/'
perm_gr_dir = '/n/scratch3/users/j/jc604/rc-test-nonallelespecific_permuted_graphs/'


wildcard_constraints:
	cancer='COAD|LUAD|MM|PAAD'


# run the pipeline
rule all:
	input:
		data_dir + 'output/exclusivity/COAD_exclusivity_results.rds',
		data_dir + 'output/exclusivity/LUAD_exclusivity_results.rds',
		data_dir + 'output/exclusivity/MM_exclusivity_results.rds',
		data_dir + 'output/exclusivity/PAAD_exclusivity_results.rds'




# create the bipartite graph from the table of samples and their mutated genes
rule make_bipartite_graph:
	input:
		in_file = data_dir + 'input/{cancer}_mutations.tsv'
	output:
		save_name = data_dir + 'intermediate/{cancer}_bgr.rds'
	script:
		'20_05_make-bipartite-graph.R'




perm_time_alloc_dict = {
	'COAD': '06:30:00',
	'LUAD': '24:00:00',
	  'MM': '01:00:00',
	'PAAD': '01:45:00',
	'SKCM': '48:00:00',
}
perm_mem_alloc_dict = {
	'COAD': '270',
	'LUAD': '250',
	  'MM': '240',
	'PAAD': '240',
	'SKCM': '300',
}

# use edge swapping to make the permutation matrices
rule permute_bipartite_graph:
	input:
		bipartite_gr = data_dir + 'intermediate/{cancer}_bgr.rds'
	output:
		save_name = perm_gr_dir + '{cancer}/{cancer}_perm{perm_num}.rds'
	params:
		time_alloc = lambda w: perm_time_alloc_dict[w.cancer],
		mem = lambda w: perm_mem_alloc_dict[w.cancer],
		partition = lambda w: 'medium' if w.cancer == 'LUAD' else 'short',
		Q = 20
	script:
		'20_06_permute-bipartite-graph.R'




results_df_time_dict = {
	'COAD':'00:25:00',
	'LUAD':'00:25:00',
	  'MM':'00:08:00',
	'PAAD':'00:10:00',
	'SKCM':'00:10:00',
}
results_df_mem_dict = {
	'COAD': 27000,
	'LUAD': 39000,
	  'MM': 10000,
	'PAAD': 20000,
	'SKCM': 20000,
}
rule make_results_df:
	input:
		real_gr = data_dir + 'intermediate/{cancer}_bgr.rds'
	output:
		output_name = data_dir + 'intermediate/{cancer}_{which_test}_results_df.rds'
	params:
		which_test_short = lambda w: 'co' if w.which_test == 'comutation' else 'ex',
		time = lambda w: results_df_time_dict[w.cancer],
		mem = lambda w: results_df_mem_dict[w.cancer],
		min_times_mut = 3
	script:
		'20_61_nonallelespec_make-results-df.R'




process_permuted_graph_time_dict = {
	'COAD': '200:00:00',
	'LUAD': '200:00:00',
	  'MM': '119:00:00',
	'PAAD': '119:00:00',
	'SKCM': '119:00:00',
}
process_permuted_graph_mem_dict = {
	'COAD': 50000,
	'LUAD': 50000,
	  'MM': 35000,
	'PAAD': 35000,
	'SKCM': 35000,
}

process_permuted_graph_partition_dict = {
	'COAD': 'long',
	'LUAD': 'long',
	  'MM': 'medium',
	'PAAD': 'medium',
	'SKCM': 'medium',
}

rule process_permuted_graph:
	input:
		results_df = data_dir + 'intermediate/{cancer}_{which_test}_results_df.rds',
		perm_grs = expand(perm_gr_dir + '{{cancer}}/{{cancer}}_perm{perm_num}.rds',
		                  perm_num=PERMUTATIONS)
	output:
		save_name = data_dir + 'output/{which_test}/{cancer}_{which_test}_results.rds'
	params:
		n_cores = 20,
		which_test_short = lambda w: 'co' if w.which_test == 'comutation' else 'ex',
		time = lambda w: process_permuted_graph_time_dict[w.cancer],
		mem = lambda w: process_permuted_graph_mem_dict[w.cancer],
		partition = lambda w: process_permuted_graph_partition_dict[w.cancer]
	script:
		'20_08_process-permuted-graph.R'
