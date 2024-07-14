#!/usr/bin/env nextflow
 
params.in = "./data/*.sdf"

sdfs = Channel.fromPath(params.in)

/*
 * process multiple sdf files
 */
process sdffiles {
    publishDir 'results_mae', mode: 'copy', overwrite: true
 
    input:
    file x from sdfs
 
    output:
    file("*.*")  
  

 """

  file="\$(echo ${x} | cut -d'.' -f 1)"
  ligprep  -isd ${x} -omae \${file}.mae -LOCAL -NOJOBID
 
"""
}
sdfs.close()
