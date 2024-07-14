params.vcf =  "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.hard_filtered.sorted.vcf.gz"
params.vcf_index = "$launchDir/data/results/var_call/allacma_fusca/allacma_fusca.hard_filtered.sorted.vcf.gz.csi"
params.bed = "$launchDir/data/results/braker/allacma_fusca/allacma_fusca.intergenic.bed"
params.outdir = "$launchDir/data/results/herho/allacma_fusca/"
params.species = "allacma_fusca"

log.info """\
         H E R H O   N F   P I P E L I N E    
         ===================================
         vcf : ${params.vcf}
	 bed : ${params.bed}
         outdir : ${params.outdir}
         species : ${params.species}
         """
         .stripIndent()

include { herho_flow } from './herho_flows.nf'

workflow {
	herho_flow(params.vcf, params.vcf_index,  params.bed, params.species)
}
