<img align="left" src="images/logo-newcastle-university.png" width="300" /> <img align="right" src="images/logo-wellcome-centre-mitcondrial-research.png" width="300" />  

&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
# A framework for tracking heteroplasmic mitochondrial mutations through cell lineages
> This research project was carried out and written up as the thesis for my MSc Bioinformatics degree at Newcastle University, and continued as a temporary bioinformatic research assistant for the Wellcome Centre for Mitochondrial Research. It was supervised by Dr Conor Lawless and Dr Dasha Deen. My full thesis can be found [here](./A_framework_for_tracking_heteroplasmic_mitochondrial_mutations_through_cell_lineages.pdf).

This specialized variant calling pipeline was developed to identify (non-pathogenic) mitochondrial mutations undergoing clonal expansion in human cell lineages. Improved understanding of clonal expansion may be necessary for the development of treatments for mitochondrial diseases in the future; this pipeline is intended to provide allele frequency data for mathematical modelling of mtDNA population dynamics.

Currently tailored for re-analysis of bulk ATAC-seq data from [Ludwig _et al._ (2019)](https://doi.org/10.1016/j.cell.2019.01.022) to quantify the expansion of mtDNA mutations throughout indirectly related clonal human cell cultures.  
&nbsp;  
<img src="images/Lineage_tree_README.jpg" width="300">  
TODO caption and reference. can't directly observe longituninally as sequencing kills clone and its descendants.


# Detecting Clonal Expansion through Lineage Validation
Specific properties are expected in mutations which show clonal expansion: they are heteroplasmic, inherited, and should demonstrate an autocorrelated pattern of changes in allele frequency between generations. These properties also provide a unique opportunity to validate variant calls: very low-level mutations which may be indistinguishable from sequencing/PCR errors in an individual, can be called with confidence when inherited and observed in related clone. Termed as _Lineage validation_, this improves variant calling because some of the limitations when calling mutations in single clones with high confidence can be bypassed. Relatively relaxed filters are used for initial variant calls in order to maximise the number of true positive, low-level calls. Then the mutations are filtered through lineage validation. This simultaneously removes potentially false positive calls from errors, (only called in individual clones), and identifies inherited mutations which may show clonal expansion. The allele frequencies of the variant calls are visualised for each path through a lineage tree, and an autocorrelated pattern of changes in allele frquencies can be used to identify clonal expansion.

# Workflow
Stages of the pipeline are split into 6 bash scripts. This is to allow different stages to be evaluated and adjusted if necessary before proceeding. For example: the quality of the data should be checked before alignment.


If running on the Newcastle rocket HPC, run scripts by submitting batch jobs to SLURM partitions: `sbatch SCRIPT_NAME.sh`. Otherwise run as normal `bash SCRIPT_NAME.sh`


1. Prefetch .sra files from the sequence read archive, and convert to fastq format ([prefetch\_files.sh](prefetch\_files.sh))
2. Assess prealignment quality ([QC.sh](QC.sh))
3. Align reads to reference genome ([analyse.sh](analyse.sh))
4. Assess alignment quality ([post\_alignment\_QC.sh](post\_alignment\_QC.sh))
5. Variant call ([variant\_call.sh](variant\_call.sh))
6. Visualise and explore clonal expansion in heteroplasmic variants ([plot\_mutations.sh](plot\_mutations.sh))

**prefetch\_files.sh** is a BASH shell script. Calls **parse.py** to get metadata (data/SRR\_Acc\_List.txt) with all the SRR names: one name per sample, for all the samples Ludwig et al., 2019). Uses SRX names in SRR\_Acc\_List.txt to download SRR files from the sequence read archive. Checks to make sure each SRR has been downloaded and is complete and reattempts download if not. Converts .sra files to .fastq format.

**parse.py** is a custom python script called by analyse.sh which gets the connection between GSM IDs (e.g. from metadata [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115218) and SRX IDs, which are needed to download the actual raw data.

**QC.sh** is a BASH shell script. Takes the SRR\*.fastq files in the group/s defined in **categories.txt**, and produces a quality report for each file using the program fastQC (results in fastQC\_results/). Then produces a summary of the whole group's quality using the program multiQC (results in multiQC/). See **categories.txt** for details.

**analyse.sh** is a BASH shell script that builds indices from the reference genome, aligns reads to the whole human genome, using [bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml). In this way reads of NUMTs (nuclear mitochondrial DNA: transposed from the mitochondrial genome to the nuclear genome) should not be mapped to the mitochondrial genome where they can introduce false variant calls. The raw alignment results (alignment\_stdout.txt) and a summary (alignment\_summary.txt) are produced.

**post\_alignment\_QC.sh** is a BASH shell script that outputs files with detailed stats of the alignment, both for the raw data and after applying filters for base and mapping quality. For each clone: mean coverage, base quality, per genomic position coverage and base quality are calculated. Can be repeated for different thresholds.

**variant_call.sh** is a BASH shell script that filters the reads and outputs the pileups of all mutations.
 
**plot\_mutations\_script.R** is an R script which outputs tables and plots to summarise and visualise heteroplasmic point mutations, to explore how allele frequencies change through different cell lineages. It also produces plots for different stages of pipeline, including pre-alignment quality plots, post-alignment. Finally it provides comparisons with the 44 high confidence alleles detected by Ludwig et al., 2019. 
It takes annotated mutserve variant files, coverage and read depth files, sample metadata, and Ludwig et al., allele frequency data, and lineage path information specified in **lineages\_paths.txt**.

### Variant Calling
1. Read filtering: 
* Mapping quality >4 
* Base quality >30
2. Variant filtering:
* Allele frequency >0.001 and < 0.990 (heteroplasmic)
* Exclude alleles with < 1 reads per allele per strand
3. __Lineage Validation__  <- FILTERING FOR MASTER'S THESIS, EXCLUDED FOR AUTOCORRELATION ANALYSIS IN UPDATED PIPELINE
* Mutation must be _present in >=2 clone in a lineage_, and have an _allele frequency >0.01 in at least one clone_ in the lineage. This allows confidence in a mutation with an allele frequency above the standard minimum threshold for sequencing/PCR errors (0.01) to be extended to low-level and indirectly inherited mutations (which would otherwise be indistinguishable from PCR error) at the same genomic position of related clones, in the same lineage.

Below is an exploratory plot of unfiltered, heteroplasmic or low-level variants for the samples in one possible path through the G11 lineage (Fig. 2). Using this plot (and tables of heteroplasmic mutations' allele frequencies) candidate mutations which demonstrate a pattern of autocorrelated allele frequencies throughout a specific lineage, indicative of clonal expansion, can be visually identified. Then a mutation load profile can be plotted for each candidate position (see Fig. 3). Each graph represents the mitochondrial genome of an clone in a specific lineage (see the [clone lineage tree](Lineage_tree_README.jpg)), with the genomic position on the x axis, and allele frequency of _heteroplasmic_ mutations on the y axis.   

<img src="results/G11_longest_upper_HET_OR_LOWLVL_nofilt.png">**Fig 2. Allele frequencies of heteroplasmic or low-level mutations for samples in one possible path through the G11 lineage.** Labels indicate the precise genomic position of the mutation. Colour indicates presence of strand bias (red = strand bias, blue = no strand bias). One example of an candidate mutation has been highlighted.

Using exploratory plots like the example above (Fig. 2) and the tables of heteroplasmic variant frequencies, individual point mutations which may show stochastic changes in allele frequency can be identified, and their mutation load profiles for a lineage plotted. 
For each candidate mutation, a mutation load profile (Fig. 3 below) can be used to observe the change in allele frequency between generations.  
<img src="results/G11_longest_upper_pos_822.png">
**Fig. 3: TODO**


Using exploratory plots like the example above (Fig. 2) and the tables of heteroplasmic variant frequencies, individual point mutations which may show stochastic changes in allele frequency can be identified, and their mutation load profiles for a lineage plotted. 
For each candidate mutation, a mutation load profile (Fig. 3 below) can be used to observe the change in allele frequency between generations.  
# Additional software 
Most of the programs used in this pipeline are already installed as a SLURM module, or can be automatically downloaded and installed using [scripts/install\_software.sh](install\_software.sh). However, SRAtoolkit must be installed _interactively_ by pasting and executing [scripts/configure\_sratools.sh](configure\_sratools.sh) _line by line_. On the rocket HPC, do this from the login node terminal (ie. do not submit script to SLURM). When prompted set default configuration by inputting: "f","y","o","x","y","o" (see the [scripts/configure\_sratools.sh](script) for more detail).



## TODO Overall
(See Progress\_log.md for shorter term aims)
- [x] Restructure project into folders, refactor code accordingly
- [x] Easier download of data/selection of samples
- [x] Check script comments
- [ ] rename project?
- [x] Mark duplicates
     - [ ] Optical duplicates
- [x] Create consenus sequence
- [x] Re-align to consensus instead of ref
- [ ] Compare genome coverage, no. variants, overlap of variants, strand bias for:
     - [x] duplicates vs no dups
     - [x] aligned to reference vs aligned to consensus
     - [ ] STRAND BIAS
- [ ] Replace mutserve variant caller:
     - [ ] Research joint variant callers: just groups or lineage/relationship metatdata
     - [ ] raw pileup calls - only quality filtering, no sequencing error adjustments
     - [ ] **write variant caller which incorporates AF autocorrelation**
     - [ ] alternative variant annotation software
- [ ] Remove two samples with skewed GC distribution?
- [ ] Modify for use outside of Rocket - yzer?
- [ ] Limit distance between paired-end reads when aligning
- [ ] Cheatsheet of useful bash and slurm commands
- [x] make group\_SRRs file
- [x] extract SRX numbers for prefetch into group\_\_SRX.txt
- [ ] baseq 20 too low? Representes quality before alignment, not recalibrated, so increasing threshold dfsafdsa. Especially when av baseq is high ~31. See spread of baseqs per position and compare between baseq 20 and 30: 1 in 100 vs 1 in 1000. baseq 25.23 = 3 in 1000 (Q = -10log10(0.003)). proportion of wrong calls per genomic position = Pr(wrong base)\*coverage/16569?
- [ ] Add validation\_groups.txt creation
- [ ] lower threshold for Heteroplasmic when creating consensus, remove HET\_orLOWLVL filters when plotting with consensus aligned data.
- [x] Plot for multiple positions within lineage/same positions in different lineages
- [ ] Look at plots for suspicious looking 
- [ ] Find out why variant calls are missing between bulk replicates when they aren't in LUDWIG - mutserve vs raw read no.s
- [ ] List known NextSeq sequencing error locations on reference sequences and compare to calls - combined lineage mutation load plots could help visualise if generation associated with sequencing runs - see metadata.
- [ ] arguments across script instead of defining variables within script
- [x] arguments to control which parts of mutation\_plot\_script.R are ran - large plots take a relatively long time and disk space.
- [ ] Watch talks from Conor and Jordan
- [ ] Find potential alternative datasets: look lineage tracing, short distance between generations.
- [ ] Research calling deletions

## Questions (add answers to research and reasoning)
- Mark known low level seq error patterns, statistically compare autocorrelation between potential seq error and not
- for this dataset and the relationships between clones (time, colony size, model - effectively back one generation, forward two)
- How distinguishable will random changes in AF be from autocorrelated? How accurate can we really expect AF estimate to be? ...Given the amount of information lost when sequencing <- illustrate with graph. Model sequencing process (assumming no amplification bias) eg.:
  - proportion of clone mtDNAs retained/captured after fragmentation, adaptor ligation and size selection: estimated mtDNA copy no.\*efficiency of ligation\*proportion of ligated in correct size range (research). 
- at base quality 30 (1 in 1000) we could expect an average of 1 read containing a mutation at every position just by chance? Not including pcr error etc.
