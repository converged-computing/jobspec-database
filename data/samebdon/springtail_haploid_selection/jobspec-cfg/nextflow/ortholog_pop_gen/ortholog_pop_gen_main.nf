// Species 1 data files
params.species_1 = "allacma_fusca"
params.genome_fasta_1 = "$launchDir/data/results/genomes/allacma_fusca/GCA_947179485.1.simple_header.earlGrey_masked.fasta"
params.genome_dict_1 = "$launchDir/data/results/genomes/allacma_fusca/GCA_947179485.1.simple_header.earlGrey_masked.dict"
params.vcf_1 = "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.hard_filtered.sorted.vcf.gz"
params.vcf_index_1 = "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.hard_filtered.sorted.vcf.gz.csi"
params.callable_bed_1 = "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.all_callable.bed"
params.annot_1 = "$launchDir/data/results/braker3/allacma_fusca/braker.gtf"
params.prot_fasta_1 = "$launchDir/data/results/braker3/allacma_fusca/braker.aa"

// Species 2 data files
params.species_2 = "dicyrtomina_minuta"
params.genome_fasta_2 = "$launchDir/data/results/genomes/dicyrtomina_minuta/qeDicMinu4_1.earlGrey_masked.fa"
params.genome_dict_2 = "$launchDir/data/results/genomes/dicyrtomina_minuta/qeDicMinu4_1.earlGrey_masked.dict"
params.vcf_2 = "$launchDir/data/results/var_call/dicyrtomina_minuta/dicyrtomina_minuta.hard_filtered.sorted.vcf.gz"
params.vcf_index_2 = "$launchDir/data/results/var_call/dicyrtomina_minuta/dicyrtomina_minuta.hard_filtered.sorted.vcf.gz.csi"
params.callable_bed_2 = "$launchDir/data/results/var_call/dicyrtomina_minuta/dicyrtomina_minuta.callable.bed"
params.annot_2 = "$launchDir/data/results/braker2/dicyrtomina_minuta/braker.gtf"
params.prot_fasta_2 = "$launchDir/data/results/braker2/dicyrtomina_minuta/braker.aa"

// add genome dicts and vcf indices

// Output Directory
params.outdir = "$launchDir/data/results/ortholog_pop_gen/allacma_fusca.vs.dicyrtomina_minuta"

log.info """\
         O R T H O L O G  P O P  G E N   N F   P I P E L I N E    
         ===================================
         Species 1 : ${params.species_1}
         Species 2 : ${params.species_2}
         outdir : ${params.outdir}
         """
         .stripIndent()

include { gen_haps_flow as gen_haps_flow_1 } from './ortholog_pop_gen_flows.nf'
include { gen_haps_flow as gen_haps_flow_2 } from './ortholog_pop_gen_flows.nf'
include { infer_orthology_flow; orthodiver_flow; merge_orthodiver_gene_pop} from './ortholog_pop_gen_flows.nf'
include { gene_pop_flow_SFS as gene_pop_flow_SFS_1 } from './gene_pop_flows.nf'
include { gene_pop_flow_SFS as gene_pop_flow_SFS_2 } from './gene_pop_flows.nf'

workflow {
        gen_haps_flow_1(params.species_1, params.genome_fasta_1, params.vcf_1, params.callable_bed_1, params.annot_1, params.prot_fasta_1)
        gen_haps_flow_2(params.species_2, params.genome_fasta_2, params.vcf_2, params.callable_bed_2, params.annot_2, params.prot_fasta_2)
        infer_orthology_flow(gen_haps_flow_1.out[1].concat(gen_haps_flow_2.out[1]).collect(), params.annot_1, params.annot_2)
        orthodiver_flow(gen_haps_flow_1.out[0], gen_haps_flow_2.out[0], infer_orthology_flow.out[1])
        gene_pop_flow_SFS_1(params.genome_fasta_1, params.genome_dict_1, params.vcf_1 ,params.vcf_index_1, infer_orthology_flow.out[2], params.species_1)
        gene_pop_flow_SFS_2(params.genome_fasta_2, params.genome_dict_2, params.vcf_2 ,params.vcf_index_2, infer_orthology_flow.out[3], params.species_2)
        merge_orthodiver_gene_pop(orthodiver_flow.out, gene_pop_flow_SFS_1.out, gene_pop_flow_SFS_2.out, infer_orthology_flow.out[4])
}

// mamba activate ortholog_pop_gen
// mamba install -c conda-forge -c bioconda seqtk samtools bedtools bcftools=1.2 docopt orthofinder mafft translatorx tqdm agat parallel pandas numpy tqdm degenotate pybedtools scikit-allel