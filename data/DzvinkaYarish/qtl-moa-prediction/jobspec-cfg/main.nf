nextflow.enable.dsl=2

include { FindOverlaps; CombineOverlaps } from './modules/bedtools_intersect'
include { FindOverlaps as FindOverlaps1; CombineOverlaps as CombineOverlaps1 } from './modules/bedtools_intersect'

include { GetEnformerPreds } from './modules/enformer'
include { GetChrombpnetPreds } from './modules/chrombpnet'
include { GetPangolinPreds } from './modules/pangolin'
include { GetSpliceAIPreds } from './modules/spliceai'
include {GetFeatures } from './modules/get_features'


workflow {
    infile = "${params.in_file}"
    cell_type = "${params.cell_type}"
//     cell_types_enf = Channel.from( [5110, 5213, 5218, 4758] ) // CAGE tracks from Enformer
//        cell_types_enf = Channel.from( [14, 62, 156, 166, 41] ) // DNASE tracks from Enformer

    cell_types_chrom = Channel.from( ["ENCSR637XSC", "ENCSR452COS", "ENCSR159GFS", "ENCSR485TLP", "ENCSR000EPK"] )
    out1 = GetEnformerPreds(infile, cell_type)
    out2 = GetChrombpnetPreds(infile, cell_types_chrom)
    out3 = GetPangolinPreds(infile, cell_type)
    out4 = GetSpliceAIPreds(infile, cell_type)

    bedFiles = Channel.fromPath( "/gpfs/space/home/dzvenymy/Thesis/RBP/data/Nostrand2020/*.bed" )

    findOverlapsResults = FindOverlaps(bedFiles, infile)

    out5 = CombineOverlaps(findOverlapsResults.collect(), "rbp")

    bedFiles = Channel.fromPath( "/gpfs/space/home/dzvenymy/Thesis/chromatin_data/*.bed" )

    findOverlapsResults = FindOverlaps1(bedFiles, infile)

    out6 = CombineOverlaps1(findOverlapsResults.collect(), "vars_in_peaks")

    res = Channel.empty()
    res.concat( out1, out4, out3, out5, out6, out2 ).set {preds}
    GetFeatures(preds.collect())






}



