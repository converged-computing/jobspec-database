#!/usr/bin/env nextflow

/*
#                          _              _
#                         | |            | |
#      ___ __ _  __ _ ___| |_ ___   ___ | |___
#    / __/ _` |/ _` / __| __/ _ \ / _ \| / __|
#   | (_| (_| | (_| \__ \ || (_) | (_) | \__ \
#   \___\__,_|\__,_|___/\__\___/ \___/|_|___/
#
# A Convergent Amino Acid Substitution identification 
# and analysis toolbox - Nextflow edition
#
# Github: https://github.com/nozerorma/caastools/blob/nextflow-rize
#
# Author:         Fabio Barteri (fabio.barteri@upf.edu)
# Contributors:   Alejandro Valenzuela (alejandro.valenzuela@upf.edu),
#                 Xavier Farré (xfarrer@igtp.cat),
#                 David de Juan (david.juan@upf.edu),
#                 Miguel Ramon (miguel.ramon@upf.edu) - Nextflow Protocol Elaboration
#
# File: main.nf
#
*/

/*
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * CAAStools: A cutting-edge bioinformatics pipeline tailored for the identification and 
 * analysis of Convergent Amino Acid Substitutions (CAAS) in evolutionary studies. 
 * Harnessing the power of Nextflow, CAAStools offers seamless integration across 
 * multiple platforms, ensuring reproducibility and scalability for large-scale genomic 
 * datasets. Dive into a world of evolutionary insights with CAAStools!
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

nextflow.enable.dsl = 2

version = "0.0.1"

// Display input parameters
log.info """

CAASTOOL - N F PIPELINE  ~  version ${version}
=============================================

 A Convergent Amino Acid Substitution identification 
 and analysis toolbox

 Author:         Fabio Barteri (fabio.barteri@upf.edu)
 Contributors:   Alejandro Valenzuela (alejandro.valenzuela@upf.edu),
                 Xavier Farré (xfarrer@igtp.cat),
                 David de Juan (david.juan@upf.edu),
                 Miguel Ramon (miguel.ramon@upf.edu) - Nextflow Protocol Elaboration

"""

/*
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  NAMED WORKFLOW FOR PIPELINE: This section includes the main CT workflow from the ct.nf file.
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

include {HELP} from './workflows/help.nf'
include { CT } from './workflows/ct.nf'

/*
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  RUN CAASTOOLS ANALYSIS: This section initiates the main CT workflow.
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

workflow {

    // Check if --help is provided
    if (params.help) {
        HELP ()
    } else {
        CT ()
    }
}


/*
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  THE END: End of the main.nf file.
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
