params.genome = "$launchDir/data/results/genomes/allacma_fusca/GCA_947179485.1.simple_header.fasta"
params.genome_index = "$launchDir/data/results/genomes/allacma_fusca/GCA_947179485.1.simple_header.fasta.fai"
params.genome_dict = "$launchDir/data/results/genomes/allacma_fusca/GCA_947179485.1.simple_header.dict"
params.vcf = "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.hard_filtered.sorted.vcf.gz"
params.vcf_index = "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.hard_filtered.sorted.vcf.gz.csi"
params.annotation = "$launchDir/data/results/braker3/allacma_fusca/braker.agat.gff3"
params.outdir = "$launchDir/data/results/gene_pop"
params.species = "allacma_fusca"

log.info """\
         V A R  C A L L   N F   P I P E L I N E    
         ===================================
         vcf : ${params.vcf}
	 annotation : ${params.annotation}
         outdir : ${params.outdir}
         species : ${params.species}
         """
         .stripIndent()

include { gene_pop_flow_GTF; gene_pop_flow_GFF; gene_pop_flow_SFS } from './gene_pop_flows.nf'

workflow {
        gene_pop_flow_SFS(params.genome, params.genome_dict, params.vcf,params.vcf_index, params.annotation, params.species)
}
