nextflow.enable.dsl=2

////////////////////
//DEFAULT PARAMETERS
////////////////////

//outdirectory for the results
params.outdir = "$baseDir/data/${params.accessionNumber}"

//directory with the scripts
params.scriptDir = "$baseDir/src"

//sample table for the experiment
params.sampleTable = "$baseDir/params/sampleTables/${params.accessionNumber}_sampleTable.txt"

//define the number of cpus use in the multi-cpu steps
params.cpuCores=6

//location of the fastqFiles if pre-downloaded
params.fastqFileDir = "$params.outdir/fastqFiles/*{R1,R2}*.fastq.gz"
downloadFiles = file(params.fastqFileDir).isEmpty()

//directory with the index files
params.referenceDir = "$baseDir/references"

//cell type database to use for the SingleR cell type annotation
if(params.species=="human"){

  params.cellTypeDatabase = "HumanPrimaryCellAtlasData"

  } else {
    params.cellTypeDatabase = "MouseRNAseqData"
  }

//species to use for downloading the index file and enrichment analysis
params.species="human"

//index for kallisto if pre-made
indexName = "${params.species}_Index.idx"
params.index = "$params.referenceDir/$indexName"

//transcript to gene map for kallisto if pre-made
t2gName = "${params.species}_t2g.txt"
params.t2g = "$params.referenceDir/$t2gName"

//the chemistry used in the single cell experiment - used in kallisto quantification
params.chemistry="10xv3"

//does the experiment contain replicated samples to perform diff exp analysis with milo?
params.replicates = false

//group names to use for liana cell communication analysis
params.cellGroups = ["SingleR.labels","seurat_clusters"]

//assume human for the default cellranger transcriptome
params.cellRangerTranscriptome = "$params.referenceDir/refdata-gex-GRCh38-2020-A"

////////////
//HELP
////////////

def helpMessage() {
  log.info"""
  =========================================
  Skeletalvis-SingleCell Pipeline
  =========================================
  Usage:
  The typical command for running the pipeline is as follows:
  nextflow run skeletalvis.nf -profile slurm -params-file params/<accessionNumber>.yaml -with-singularity library://jsoul/default/singlecell:latest

  Required arguments:
  -profile                      Configuration profile to use. <local, slurm>
  -params-file                  Yaml file containing parameters for the analysis
  --with-singularity            Recommended to use the provided singularity container

  """.stripIndent()
}

params.help = false
if (params.help){
  helpMessage()
  exit 0
}


//////////////
//LOAD MODULES
//////////////

include { ENA_DOWNLOAD } from './modules/ena_download'
include { SRA_DOWNLOAD } from './modules/sra_download'

include { BAM_DOWNLOAD } from './modules/bam_download'
include { BAMTOFASTQ } from './modules/bamtofastq'

include { FASTQTOSAMPLEID } from './modules/fastqFileToSampleID'
include { FASTQC } from './modules/fastQC'
include { MULTIQC } from './modules/multiQC'
include { KALLISTOBUSTOOLS_REF } from './modules/kallistobustools_ref'
include { KB_COUNT } from './modules/kbcount.nf'
include { CELLRANGER_COUNT } from './modules/cellrangercount.nf'
include { SEURAT_QC } from './modules/seurat_QC'
include { SEURAT_CELLCYCLENORMALISE } from './modules/seurat_cellCycleNormalise'
include { SEURAT_MARKERGENES } from './modules/seurat_markergenes'
include { CELL_ANNOTATE } from './modules/cell_annotate'
include { SEURAT_INTEGRATE } from './modules/seurat_integrate'
include { MAKEREACTOMEPATHWAYS } from './modules/makeReactomePathways'
include { GSEA } from './modules/gsea'
include { CELL_CROSSTALK } from './modules/cell_crosstalk'
include { MILO } from './modules/milo'

////////////
//WORKFLOWS
////////////

//subworkflow to download the fastq files from ENA/SRA or bam files if needed
workflow downloadRNASeqData {

  take: sampleTable

  main:
  if(params.downloadSite=="ENA") {
   fastqFiles = ENA_DOWNLOAD(sampleTable)
   } else if (params.downloadSite=="SRA") {
     fastqFiles = SRA_DOWNLOAD(sampleTable)
     } else {
      fastqFiles = BAM_DOWNLOAD(sampleTable) | flatten | BAMTOFASTQ

    }


    emit:
    fastqFiles


  }


//workflow to perform single cell-seq  from fastq files
//including alignment, qc, clustering and cell type annotation
workflow  {

  //read in the sample table for the experiment
  sampleTable = file( params.sampleTable )


  //download the fastq files if not present in the fastq dir
  if(downloadFiles){

    rawData = downloadRNASeqData(sampleTable).fastqFiles.flatten()

    } else {

      rawData = channel.fromPath( params.fastqFileDir).view()

    }


    //create the kb index if not pesent in the ref dir
    if(!file(params.index).exists() | !file(params.t2g).exists()){

      ref = KALLISTOBUSTOOLS_REF(params.species)
      index = ref.index
      t2g = ref.t2g

      } else {

        index = file(params.index)
        t2g = file(params.t2g)
      }

      //group the reads into pairs based on the file name
      groupedReads = rawData.map { file ->
        def key = file.name.toString().split("_R\\d*")[0]
        return tuple(key, file)
      }
      .groupTuple(sort:true).view()

      //map the fastq files to the sampleID and group by sampleID
      samples_grouped = FASTQTOSAMPLEID(groupedReads,sampleTable)
      .groupTuple()
      .map{id,fastq -> tuple(id,fastq.flatten().sort())}.view()

      //quality control of the fastq files
      FASTQC(samples_grouped)
      MULTIQC(FASTQC.out.stats.collect())

      //map the reads to the transcriptome to with kallisto
      //to create cell x gene matrix for each sample
      counts = KB_COUNT(samples_grouped,index,t2g,params.chemistry)

      //run seurat based analysis on individual samples

      //runQC
      SEURAT_QC(KB_COUNT.out.counts_unfiltered,t2g,sampleTable)
      
      //normalise the data
      SEURAT_CELLCYCLENORMALISE(SEURAT_QC.out.seurat)
      
      //automatically annotate the cells by celltype
      CELL_ANNOTATE(SEURAT_CELLCYCLENORMALISE.out.seurat)

      //identify clusters of cells and find marker genes for each cluster
      SEURAT_MARKERGENES(CELL_ANNOTATE.out.seurat)
      annotated = CELL_ANNOTATE.out.seurat.map(it->it[1]).collect()

      //predict ligand-receptor interactions between annotated cell types
      CELL_CROSSTALK(SEURAT_MARKERGENES.out.seurat,params.cellGroups)    

      //generate reactome pathway for the designated species
      pathways = MAKEREACTOMEPATHWAYS(params.species)

      //perform pathway enrichment analysis of the marker genes
      GSEA(SEURAT_MARKERGENES.out.markers,pathways)

      if(params.replicates){
      //integrate all the data together and perform diffexp analysis between conditions with milo
      SEURAT_INTEGRATE(annotated)
      MILO(SEURAT_INTEGRATE.out.seurat)
    }



  }

  //workflow to perform single cell-seq  from fastq files
//including alignment, qc, clustering and cell type annotation
workflow CellRanger {

  //read in the sample table for the experiment
  sampleTable = file( params.sampleTable )


  //download the fastq files if not present in the fastq dir
  if(downloadFiles){

    rawData = downloadRNASeqData(sampleTable).fastqFiles.flatten()

    } else {

      rawData = channel.fromPath( params.fastqFileDir).view()

    }


    //group the reads into pairs based on the file name
    groupedReads = rawData.map { file ->
      def key = file.name.toString().split("_R\\d*")[0]
      return tuple(key, file)
    }
    .groupTuple(sort:true).view()

    //map the fastq files to the sampleID and group by sampleID
    samples_grouped = FASTQTOSAMPLEID(groupedReads,sampleTable)
    .groupTuple()
    .map{id,fastq -> tuple(id,fastq.flatten().sort())}.view()

    //map the reads to the transcriptome to with cellranger
    //to create cell x gene matrix and qc report for each sample
    transcriptome = file(params.cellRangerTranscriptome)

    counts = CELLRANGER_COUNT(samples_grouped,transcriptome)


}
