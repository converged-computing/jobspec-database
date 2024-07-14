nextflow.enable.dsl=2

include {
    barcode_correction;
} from './src/barcode/main.nf' params(params)
include {
    adapter_trimming;
} from './src/trimgalore/main.nf' params(params)
include {
    mapping;
} from './src/bwamaptools/main.nf' params(params)


workflow atac_preprocess {
    barcode_correction(file(params.data.metadata))
    adapter_trimming(barcode_correction.out)
    mapping(adapter_trimming.out)
}

