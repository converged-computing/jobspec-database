// Follows the tutorial at https://odelaneau.github.io/GLIMPSE/docs/tutorials/getting_started/#imputation-using-1000gp-reference-panel

nextflow.enable.dsl=2



samples_ch = Channel.fromPath(params.samples)
    .splitCsv(header: true, sep: "\t", strip: true)
    .map( row -> [row.sample, row.bam] )

chrs_ch = Channel.fromList(params.chromosomes.split(",") as List)

ref_pan_ch = Channel.fromPath(params.reference_panel)

ref_gen_ch = Channel.fromPath(params.reference_genome)

genetic_map_ch = Channel.fromPath(params.genetic_map, type: "dir")



process split_ref_pan {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        tuple val(chr), path(ref_pan), path(ref_pan_idx)

    output:
        tuple val(chr), path(ref_pan_chr), path(ref_pan_chr_idx)

    script:
        ref_pan_chr = "ref_pan.chr${chr}.vcf.gz"
        ref_pan_chr_idx = "${ref_pan_chr}.csi"

        """
        bcftools view -r ${chr} -i "INFO/SVTYPE=''" -m2 -M2 --threads $task.cpus ${ref_pan} -Oz -o ${ref_pan_chr}
        bcftools index -f --threads $task.cpus ${ref_pan_chr}
        """
}

process extract_ref_pan_sites {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        tuple val(chr), path(ref_pan_chr), path(ref_pan_chr_idx)

    output:
        tuple val(chr), path(ref_pan_chr), path(ref_pan_chr_idx), path(sites_vcf), path(sites_vcf_idx), path(sites_tsv), path(sites_tsv_idx), path(sites_no_indels_vcf), path(sites_no_indels_vcf_idx), path(sites_no_indels_tsv), path(sites_no_indels_tsv_idx)

    script:
        sites_vcf = "ref_pan.chr${chr}.sites.vcf.gz"
        sites_no_indels_vcf = "ref_pan.chr${chr}.sites.no_indels.vcf.gz"
        sites_tsv = "ref_pan.chr${chr}.sites.tsv.gz"
        sites_no_indels_tsv = "ref_pan.chr${chr}.sites.no_indels.tsv.gz"
        sites_vcf_idx = "${sites_vcf}.csi"
        sites_no_indels_vcf_idx = "${sites_no_indels_vcf}.csi"
        sites_tsv_idx = "${sites_tsv}.tbi"
        sites_no_indels_tsv_idx = "${sites_no_indels_tsv}.tbi"
        // Create 2 versions of sites: one has indels, other doesn't
        """
        bcftools view --drop-genotypes --threads $task.cpus ${ref_pan_chr} -Oz -o ${sites_vcf}
        bcftools view --drop-genotypes -v snps --threads $task.cpus ${ref_pan_chr} -Oz -o ${sites_no_indels_vcf}
        bcftools index --threads $task.cpus -f ${sites_vcf}
        bcftools index --threads $task.cpus -f ${sites_no_indels_vcf}
        bcftools query -f '%CHROM\\t%POS\\t%REF,%ALT\\n' ${sites_vcf} | bgzip --threads $task.cpus -c > ${sites_tsv}
        bcftools query -f '%CHROM\\t%POS\\t%REF,%ALT\\n' ${sites_no_indels_vcf} | bgzip --threads $task.cpus -c > ${sites_no_indels_tsv}
        tabix -s1 -b2 -e2 ${sites_tsv}
        tabix -s1 -b2 -e2 ${sites_no_indels_tsv}
        """
}

process chunk_chr {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        tuple val(chr), path(ref_pan_chr), path(ref_pan_chr_idx), path(sites_vcf), path(sites_vcf_idx), path(sites_tsv), path(sites_tsv_idx), path(sites_no_indels_vcf), path(sites_no_indels_vcf_idx), path(sites_no_indels_tsv), path(sites_no_indels_tsv_idx)

    output:
        tuple val(chr), path(ref_pan_chr), path(ref_pan_chr_idx), path(sites_vcf), path(sites_vcf_idx), path(sites_tsv), path(sites_tsv_idx), path(sites_no_indels_vcf), path(sites_no_indels_vcf_idx), path(sites_no_indels_tsv), path(sites_no_indels_tsv_idx), path(chunks)

    script:
        window_size = 2000000
        buffer_size = 200000
        chunks = "ref_pan.chr${chr}.chunks.txt"
        
        """
        GLIMPSE_chunk --input ${sites_vcf} --region ${chr} --window-size ${window_size} --buffer-size ${buffer_size} --thread $task.cpus --output ${chunks}
        """
}

// TODO: Run in parallel with the previous process.
// TODO: To impute X-chromosome, give --ploidy-file to bcftools call. Use val(chr) to determine ploidy. 
process sample_GLs {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        tuple val(sample), path(bam), path(bam_idx), path(ref_gen), path(ref_gen_idx), val(chr), path(ref_pan_chr), path(ref_pan_chr_idx), path(sites_vcf), path(sites_vcf_idx), path(sites_tsv), path(sites_tsv_idx), path(sites_no_indels_vcf), path(sites_no_indels_vcf_idx), path(sites_no_indels_tsv), path(sites_no_indels_tsv_idx), path(chunks)

    output:
        tuple val(sample), val(chr), path(ref_pan_chr), path(ref_pan_chr_idx), path(chunks), path(sample_GLs), path(sample_GLs_idx)

    script:
        sample_GLs = "${sample}.chr${chr}.GLs.vcf.gz"
        sample_GLs_idx = "${sample_GLs}.csi"
        sample_name = "sample_name.txt"
        sample_GLs_temp = "${sample}.chr${chr}.GLs.new_samplename.vcf.gz"
        
        """
        bcftools mpileup -f ${ref_gen} --redo-BAQ --annotate 'FORMAT/DP' -T ${sites_no_indels_vcf} --regions ${chr} ${bam} -Ou | bcftools call --threads $task.cpus -Aim -C alleles -T ${sites_no_indels_tsv} -Oz -o ${sample_GLs}
        echo $sample > $sample_name
        bcftools reheader --samples $sample_name --threads $task.cpus $sample_GLs -o $sample_GLs_temp
        mv $sample_GLs_temp $sample_GLs
        bcftools index -f $sample_GLs
        """
}

process impute {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        tuple val(sample), val(chr), path(ref_pan_chr), path(ref_pan_chr_idx), path(chunks), path(sample_GLs), path(sample_GLs_idx), path(map)

    output:
        tuple val(sample), val(chr), path(ligated), path(ligated_idx)

    script:
        imputed_filenames = "imputed_filenames.txt"
        ligated = "${sample}.chr${chr}.imputed.ligated.bcf"
        ligated_idx = "${ligated}.csi"

        """
        while ifs="" read -r line || [ -n "\$line" ]; do
            printf -v id "%02d" \$(echo \$line | cut -d" " -f1)
            irg=\$(echo \$line | cut -d" " -f3)
            org=\$(echo \$line | cut -d" " -f4)
            imputed=${sample}.chr${chr}.imputed.\${id}.bcf
            GLIMPSE_phase --impute-reference-only-variants --input ${sample_GLs} --reference ${ref_pan_chr} --map ${map} --input-region \$irg --output-region \$org --output \$imputed
            bcftools index --threads $task.cpus -f \$imputed
        done < ${chunks}

        ls ${sample}.chr${chr}.imputed.*.bcf > ${imputed_filenames}
        GLIMPSE_ligate --thread $task.cpus --input ${imputed_filenames} --output ${ligated}
        bcftools index --threads $task.cpus -f ${ligated}
        """
}

process sample {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"
    publishDir "${params.outdir}/glimpse/phased/${sample}/chr${chr}", mode: "copy", pattern: "${phased}*"
    
    input:
        tuple val(sample), val(chr), path(ligated), path(ligated_idx)
    
    output:
        tuple val(sample), path(ligated), path(ligated_idx), path(phased), path(phased_idx)
    
    script:
        phased = "${sample}.chr${chr}.phased.bcf"
        phased_idx = "${phased}.csi"

        """
        GLIMPSE_sample --thread $task.cpus --input ${ligated} --solve --output ${phased}
        bcftools index --threads $task.cpus -f ${phased}
        """    
}

// Everything from this point on is my original creation.
process concat_chrs {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        tuple val(sample), path(chromosomes_bcfs), path(chromosomes_idxs)

    output:
        tuple path(sorted), path(sorted_idx)

    script:
        concat_filenames = "concat_filenames.txt"
        sorted = "${sample}.all_chrs.sorted.bcf.gz"
        sorted_idx = "${sorted}.csi"

        """
        ls ${sample}.chr*.imputed.ligated.bcf > $concat_filenames
        bcftools concat --threads $task.cpus --file-list $concat_filenames -Ou | bcftools sort -Ob -o $sorted
        bcftools index --threads $task.cpus $sorted
        """
}

process merge_inds {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"

    input:
        path(samples_files)

    output:
        tuple path(merged), path(merged_idx)

    script:
        merge_filenames = "merge_filenames.txt"
        merged = "merged.imputed.ligated.no_tags.vcf.gz"
        merged_idx = "${merged}.csi"

        """
        ls *.all_chrs.sorted.bcf.gz > $merge_filenames
        bcftools merge -m none --file-list $merge_filenames --threads $task.cpus -Ou | bcftools +impute-info --threads $task.cpus -Oz -o $merged
        bcftools index --threads $task.cpus $merged
        """
}

process fill_tags {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"
    publishDir "${params.outdir}/glimpse/imputed.ligated", mode: "copy"

    input:
        tuple path(merged), path(merged_idx)
    
    output:
        tuple path(merged_fill_tags), path(merged_fill_tags_idx)

    script:
        merged_fill_tags = "merged.imputed.ligated.vcf.gz"
        merged_fill_tags_idx = "${merged_fill_tags}.csi"

        """
        bcftools +fill-tags ${merged} -Oz -o ${merged_fill_tags}
        bcftools index --threads $task.cpus ${merged_fill_tags}
        """
}

process filter_vcf {
    container = "quay.io/eqtlcatalogue/glimpse:1.1.1"
    publishDir "${params.outdir}/glimpse/imputed.ligated", mode: "copy"

    input:
        tuple path(merged), path(merged_idx)

    output:
        tuple path(filtered), path(filtered_idx)

    script:
        filtered = "merged.imputed.ligated.filtered.vcf.gz"
        filtered_idx = "${filtered}.csi"

        """
        bcftools filter -i 'MAF[0] > 0.01 && INFO/INFO > 0.4' ${merged} -Oz -o ${filtered}
        bcftools index --threads $task.cpus ${filtered}
        """
}



workflow {
    //println "chr, ref_pan"
    ref_pan_chr_ch = chrs_ch
        .combine(ref_pan_ch)
        .map( it -> it + ["${it[1]}.csi"])
    //ref_pan_chr_ch.view()

    //println "chr, ref_pan, sites, chunks"
    split_ref_pan_ch = split_ref_pan(ref_pan_chr_ch) | extract_ref_pan_sites | chunk_chr
    //split_ref_pan_ch.view()


    //println "sample, bam, ref_gen, chr, ref_pan, sites, chunks"
    job_params_ch = samples_ch
        .map( it -> it + ["${it[1]}.bai"] )
        .combine(ref_gen_ch)
        .map( it -> it + ["${it[3]}.fai"] )
        .combine(split_ref_pan_ch)
    //job_params_ch.view()

    //println "sample, chr, ref_pan, chunks, sample_GLs"
    job_params_ch = sample_GLs(job_params_ch)
    //job_params_ch.view()
        

    //println "sample, chr, ref_pan, chunks, sample_GLs, map"
    job_params_ch = job_params_ch
        .combine(genetic_map_ch)
        .map( it -> it[0..6] + ["${it[7]}/chr${it[1]}.gmap.gz"] )
    //job_params_ch.view()

    //println "sample, ligated, phased"
    imputed_chromosomes_ch = impute(job_params_ch) | sample 
    //imputed_chromosomes_ch.view()

    //println "sample, [ligated]"
    imputed_chromosomes_grouped_ch = imputed_chromosomes_ch
        .map( it -> it[0..2] )  // Phased files were published in the previous process. Continue only with ligated.
        .groupTuple() // TODO: use the expected values param to specify how many chromosomes were imputed to send to next process ASAP.
    //imputed_chromosomes_grouped_ch.view()

    //println "merged"
    concat_chrs(imputed_chromosomes_grouped_ch) | \
    collect | \
    merge_inds | \
    fill_tags | \
    filter_vcf //| view()
}
