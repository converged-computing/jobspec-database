// could prob just put index and dict in as processes
// need to figure out config files

params.genome = "$launchDir/data/results/genomes/dicyrtomina_minuta/qeDicMinu4_1.earlGrey_masked.fa"
params.genome_index = "$launchDir/data/results/genomes/dicyrtomina_minuta/qeDicMinu4_1.earlGrey_masked.fa.fai"
params.genome_dict = "$launchDir/data/results/genomes/dicyrtomina_minuta/qeDicMinu4_1.earlGrey_masked.dict"
params.reads = "$launchDir/data/raw_data/reseq/dicyrtomina_minuta/*/*{1,2}.fq.gz"
params.outdir = "$launchDir/data/results/var_call/dicyrtomina_minuta/"
params.species = "dicyrtomina_minuta"

log.info """\
         V A R  C A L L   N F   P I P E L I N E    
         ===================================
         genome : ${params.genome}
	 reads : ${params.reads}
         outdir : ${params.outdir}
         species : ${params.species}
         """
         .stripIndent()

include { var_call_flow } from './var_call_flows.nf'

workflow {
        read_pairs_ch = Channel.fromFilePairs( params.reads, checkIfExists:true )
        var_call_flow(params.genome, params.genome_index, params.genome_dict, read_pairs_ch, params.species)
}


