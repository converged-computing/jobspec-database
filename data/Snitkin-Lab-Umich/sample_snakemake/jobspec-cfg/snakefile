# snakefile to sample from distributions and plot all on graph

distributions = ['rnorm','rexp']

rule all:
  input:
    expand('results/{distn}.tsv',distn=distributions),
    'figures/histogram.pdf',

rule sample_from_distn:
  input:
    'scripts/sample_from_distn.R'
  output:
    'results/{distn}.tsv'
  script:
    'scripts/sample_from_distn.R'

rule combine_distn_dat:
  input:
    expand('results/{distn}.tsv',distn=distributions)
  output:
    'results/combined_dat.tsv'
  shell:
    'cat {input} > {output[0]}'

rule plot_histogram:
  input:
    'scripts/plot_histogram.R',
    'results/combined_dat.tsv'
  output:
    'figures/histogram.pdf'
  script:
    'scripts/plot_histogram.R'

rule clean:
  shell:
    'rm -r figures/ results/'
