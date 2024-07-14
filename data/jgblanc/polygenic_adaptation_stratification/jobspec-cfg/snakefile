CHR =[]
for i in range(0, 200):
  CHR.append(str(i))
CONFIG=["C1"]
REP = []
for i in range(1,101):
  REP.append("B"+str(i))
HERITABILITY = ["h2-0.3"]
#ENV = ["env_0.0","env_0.01", "env_0.02", "env_0.03", "env_0.04", "env_0.05", "env_0.06", "env_0.07", "env_0.08", "env_0.09", "env_0.1"]
#ENV = ["env_0.0","env_0.2", "env_0.5", "env_1.0"]
ENV = ["env_0.0","env_-0.2", "env_0.2"]
TS=["p-0.50", "p-0.53", "p-0.56", "p-0.59", "p-0.62"]
#TS=["p-0.50"]
#NUM_CAUSAL = ["c-200", "c-2000", "c-20000", "c-all"]
NUM_CAUSAL = ["c-200"]
PC=[1]
SIZE=2000
NUM_RESAMPLE=1000

wildcard_constraints:
    rep="[A-Z]\d+",
    config="C.",
    h2="h2-[0-1].[0-9]",
    env="env_-?[0-9].[0-9]*",
    ts="p-[0-1].[0-9][0-9]",
    dir="[a-z]*",
    pc="[0-9]"

def get_seed_msprime(rep):
  out = int(''.join(list(rep)[1::])) * 1000
  return out

def get_pc_list(x):
  A = [str(i) for i in x]
  out = "-".join(A)
  return out

def get_params(x):
  out = x.split("-")[1]
  return out

def get_env(x):
  out = x.split("_")[1]
  return out

def get_pc_num(x):
  end = str(int(x) + 4)
  start = str(5)
  out = start + "-" + end
  if int(x) == 1:
     out = str(5)
  #print(out)
  return out

rule all:
    input:
        expand("output/PGA_test/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/Qx_ss.txt", chr=CHR,rep=REP, config=CONFIG, h2=HERITABILITY, ts=TS, env=ENV,nc=NUM_CAUSAL, pc=PC)


# Summary Stat version

rule Calc_ss:
  input:
    betas="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.pheno_strat.glm.linear",
    genos_test="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam",
    Tvec="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tvec.txt",
    pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop",
    es="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.effects.txt"
  output:
    qx="output/PGA_test/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/Qx_ss.txt"
  params:
    num=NUM_RESAMPLE
  shell:
    """
    Rscript code/PGA_test/calcQ_4PopSplit_SS.R output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common {input.Tvec} {input.pops} {params.num} {output.qx} {input.es} {input.betas} output/Run_GWAS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/
    rm {input.betas}
    """




# Simluate Genotypes

rule simulate_genotypes_4popsplit:
    output:
        expand("output/Simulate_Genotypes/4PopSplit/{{rep}}/genos_{chr}.vcf", chr=CHR),
	      "output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop"
    params:
        chr_num = len(CHR),
        seed = lambda wildcards: get_seed_msprime(wildcards.rep)
    shell:
        """
        python code/Simulate_Genotypes/generate_genotypes_4PopSplit.py \
	      --outpre output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/genos \
	      --chr {params.chr_num} \
       	--Nanc 10000 \
	      --NA 10000 \
	      --NB 10000 \
	      --NC 10000 \
	      --ND 10000 \
  	    -a 40000 \
	      -b 40000 \
	      -c 40000 \
	      -d 40000 \
        -s1 5000 \
        -s2 2500 \
        -L 1000000 \
        --seed {params.seed}
        """

rule format_VCF:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos_{chr}.vcf"
    output:
        gz="output/Simulate_Genotypes/4PopSplit/{rep}/genos_{chr}.ids.vcf.gz"
        #csi="output/Simulate_Genotypes/4PopSplit/{rep}/genos_{chr}.ids.vcf.gz.csi"
    shell:
        """
	      head -n6 {input} > output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/header_{wildcards.chr}.txt
	      cat output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/header_{wildcards.chr}.txt <(cat output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/genos_{wildcards.chr}.vcf | awk -v OFS="\t" 'NR>6 {{$3=$1"_"$2"_A_T";$4="A"; $5="T"; print ;}}') | bgzip > {output.gz}
		    #bcftools index {output.gz}
			  rm output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/header_{wildcards.chr}.txt
			  rm {input}
			  """

rule concat_vcfs:
    input:
        expand("output/Simulate_Genotypes/4PopSplit/{{rep}}/genos_{chr}.ids.vcf.gz", chr=CHR)
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos.ids.vcf.gz"
    shell:
        """
        bcftools concat {input} -o output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/temp.vcf.gz -O z
        bcftools annotate --rename-chrs code/Simulate_Genotypes/convert_chr.txt output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/temp.vcf.gz -o {output} -O z
        rm output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/temp.vcf.gz
	rm {input}
        """

rule convert_vcf_to_plink:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos.ids.vcf.gz"
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos.psam",
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos.pgen",
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos.pvar"
    shell:
        """
        plink2 \
        --double-id \
        --make-pgen \
        --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/genos \
        --vcf {input}
        """

rule create_panels_4PopSplit:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop"
    output:
        C1_gwas = "output/Simulate_Genotypes/4PopSplit/{rep}/C1/ids.gwas",
        C1_test = "output/Simulate_Genotypes/4PopSplit/{rep}/C1/ids.test",
        C2_gwas = "output/Simulate_Genotypes/4PopSplit/{rep}/C2/ids.gwas",
        C2_test ="output/Simulate_Genotypes/4PopSplit/{rep}/C2/ids.test"
    params:
        size = SIZE,
        seed = lambda wildcards: get_seed_msprime(wildcards.rep)
    shell:
        "Rscript code/Simulate_Genotypes/split_gwas-test_4PopSplit.R {input} {params.size} {output.C1_gwas} {output.C1_test} {output.C2_gwas} {output.C2_test} {params.seed}"

rule split_into_test_gwas:
    input:
        gwas="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/ids.gwas",
        test="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/ids.test",
        psam="output/Simulate_Genotypes/4PopSplit/{rep}/genos.psam"
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.psam",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.pvar",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.pgen",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.psam",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.pvar",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.pgen"
    shell:
        """
	      plink2 \
	      --keep {input.gwas} \
		    --make-pgen \
			  --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas \
				--pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/genos \
				--rm-dup exclude-all

        plink2 \
        --keep {input.test} \
        --make-pgen \
        --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test \
        --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/genos \
	      --rm-dup exclude-all
	      """

rule get_variant_freq:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.psam"
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.afreq",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.afreq"
    shell:
        """
        plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test \
	      --freq \
		    --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test

			  plink2 \
        --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas \
        --freq \
        --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas
        """

rule get_common_snp_list:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.afreq",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.afreq"
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/common_snp_ids.txt"
    script:
        "code/Simulate_Genotypes/get_common_snp_list.R"

rule remake_panels_with_common_snps:
    input:
        common_id="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/common_snp_ids.txt",
        test="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.psam",
        gwas="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.psam",
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam"
    shell:
        """
        plink2 \
        --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test \
        --extract {input.common_id} \
        --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common \
	      --make-pgen

        plink2 \
        --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas \
        --extract {input.common_id} \
        --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
	      --make-pgen
        """

rule common_snp_freq:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam"
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.afreq",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.afreq"
    shell:
        """
        plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
	      --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
		    --freq

		    plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common \
	      --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common \
		    --freq
		    """

rule calculate_fst:
    input:
        gwas_genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
        test_genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam",
        gwas_ids="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/ids.gwas",
        test_ids="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/ids.test"
    output:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.fst.summary",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.fst.summary"
    shell:
        """
        plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
	      --pheno {input.gwas_ids} \
	      --fst POP \
	      --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common

		    plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common \
	      --pheno {input.test_ids} \
	      --fst POP \
	      --out output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common
		    """

#rule aggregate_genotypes:
#    input:
#        frq_common_gwas=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.afreq", rep=REP, config=CONFIG),
#        frq_common_test=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.afreq", rep=REP, config=CONFIG),
#        genos=expand("output/Simulate_Genotypes/4PopSplit/{rep}/genos_{chr}.vcf", chr=CHR, rep=REP, config=CONFIG),
#        gz_chr=expand("output/Simulate_Genotypes/4PopSplit/{rep}/genos_{chr}.ids.vcf.gz", chr=CHR, rep=REP),
#        frq_test=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.afreq", rep=REP, config=CONFIG),
#        frq_gwas=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.afreq", rep=REP, config=CONFIG),
#        gz=expand("output/Simulate_Genotypes/4PopSplit/{rep}/genos.ids.vcf.gz", rep=REP),
#        gwas_pgen=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.pgen", rep=REP, config=CONFIG),
#        gwas_pvar=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.pvar", rep=REP, config=CONFIG),
#        gwas_psam=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas.psam", rep=REP, config=CONFIG),
#        test_pgen=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.pgen", rep=REP, config=CONFIG),
#        test_pvar=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.pvar", rep=REP, config=CONFIG),
#        test_psam=expand("output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test.psam", rep=REP, config=CONFIG),
#        pgen=expand("output/Simulate_Genotypes/4PopSplit/{rep}/genos.pgen", rep=REP, config=CONFIG),
#        pvar=expand("output/Simulate_Genotypes/4PopSplit/{rep}/genos.pvar", rep=REP, config=CONFIG),
#        psam=expand("output/Simulate_Genotypes/4PopSplit/{rep}/genos.psam", rep=REP, config=CONFIG)
#    output:
#        expand("output/Simulate_Genotypes/4PopSplit/{rep}/ff.txt", rep=REP)
#    shell:
#        """
#	      touch {output}
#	      echo {input.frq_common_gwas}
#	      echo {input.frq_common_test}
#	      rm {input.genos}
#	      rm {input.frq_test}
#	      rm {input.frq_gwas}
#	      rm {input.gz}
#	      rm {input.gz_chr}
#	      rm {input.gwas_pgen}
#	      rm {input.gwas_pvar}
#	      rm {input.gwas_psam}
#	      rm {input.test_pgen}
#	      rm {input.test_pvar}
#	      rm {input.test_psam}
#	      rm {input.pgen}
#	      rm {input.pvar}
#	      rm {input.psam}
#	      """

# Simluate Phenotypes

rule draw_effect_sizes:
    input:
        freq="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.afreq",
        pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop"
    output:
        "output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.effects.txt"
    params:
        her = lambda wildcards: get_params(wildcards.h2),
        prob = lambda wildcards: get_params(wildcards.ts),
        nc = lambda wildcards: get_params(wildcards.nc)
    shell:
        "Rscript code/Simulate_Phenotypes/draw_effect_sizes_num_causal_4PopSplit.R {input.freq} {output} {params.her} 0.4 output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common {input.pops} {params.prob} {params.nc}"

rule generate_genetic_values:
    input:
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.pvar",
        "output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.pgen",
        "output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.effects.txt"
    output:
        "output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.gvalue.sscore"
    shell:
        """
        plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
	      --out output/Simulate_Phenotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common.gvalue \
		    --score output/Simulate_Phenotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common.effects.txt cols=dosagesum,scoresums
		    """

rule simulate_phenotype_4PopSplit:
    input:
        gvalues="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.gvalue.sscore",
        pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop"
    output:
        "output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.phenos.txt"
    params:
        her = lambda wildcards: get_params(wildcards.h2),
        en = lambda wildcards: get_env(wildcards.env)
    shell:
        "Rscript code/Simulate_Phenotypes/simulate_phenotypes_4PopSplit.R {input.gvalues} {input.pops} {output} {params.her} {params.en}"

# Project Test vector

rule make_test_vector:
    input:
        pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop",
        fam="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam"
    output:
        "output/Calculate_FGr/4PopSplit/{rep}/{config}/Tvec.txt"
    shell:
        "Rscript code/Calculate_FGr/4PopSplit_make_tvec.R {input.pops} {input.fam} {output}"

rule compute_FGr:
    input:
        test="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam",
        gwas="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.pgen",
        tvec="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tvec.txt"
    output:
        "output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm.txt"
    shell:
        """
        Rscript code/Calculate_FGr/calc_FGr.R output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common {input.tvec} output/Calculate_FGr/4PopSplit/{wildcards.rep}/{wildcards.config}/
 		    """

# GWAS PCA

rule GWAS_PCA:
    input:
        psam="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
        pgen="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.pgen",
        pvar="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.pvar"
    output:
        "output/Calculate_FGr/4PopSplit/{rep}/{config}/genos-gwas.eigenvec",
        "output/Calculate_FGr/4PopSplit/{rep}/{config}/genos-gwas.eigenval"
    shell:
        """
        plink2 \
	      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
	      --out output/Calculate_FGr/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas \
		    --pca 10 approx
		    """

# Run GWAS

rule format_covars:
    input:
      pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop",
      fam="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
      Tm="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm.txt",
      PC="output/Calculate_FGr/4PopSplit/{rep}/{config}/genos-gwas.eigenvec"
    output:
      "output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm-ID_covars.txt",
    shell:
      "Rscript code/Calculate_FGr/format_ID_covars.R {input.pops} {input.Tm} {input.fam} {output} {input.PC}"

rule gwas_no_correction:
  input:
      genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
      pheno="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.phenos.txt"
  output:
      "output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.pheno_strat.glm.linear"
  shell:
      """
      plink2 \
      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
      --glm allow-no-covars\
      --pheno {input.pheno} \
      --pheno-name pheno_strat \
      --out output/Run_GWAS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common
      """

rule gwas_Tm:
    input:
      genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
      pheno="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.phenos.txt",
      Tm="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm-ID_covars.txt"
    output:
      "output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.pheno_strat.glm.linear"
    shell:
      """
      plink2 \
      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
      --glm hide-covar \
      --covar {input.Tm} \
      --covar-col-nums 3 \
      --pheno {input.pheno} \
      --pheno-name pheno_strat \
      --out output/Run_GWAS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common-Tm
      """

rule gwas_PopID:
    input:
      genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
      pheno="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.phenos.txt",
      Tm="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm-ID_covars.txt"
    output:
      "output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.pheno_strat.glm.linear"
    shell:
      """
      plink2 \
      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
      --glm hide-covar \
      --covar {input.Tm} \
      --covar-col-nums 4 \
      --pheno {input.pheno} \
      --pheno-name pheno_strat \
      --out output/Run_GWAS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common-ID
      """

rule gwas_PC:
    input:
      genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-gwas_common.psam",
      pheno="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.phenos.txt",
      Tm="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm-ID_covars.txt"
    output:
      "output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-{pc}.pheno_strat.glm.linear"
    params:
       pc_num = lambda wildcards: get_pc_num(wildcards.pc)
    shell:
      """
      plink2 \
      --pfile output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common \
      --glm hide-covar \
      --covar {input.Tm} \
      --covar-col-nums {params.pc_num} \
      --pheno {input.pheno} \
      --pheno-name pheno_strat \
      --out output/Run_GWAS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common-{wildcards.pc}
      """

# Ascertain SNPs

rule pick_SNPS:
    input:
      causal_effect="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.effects.txt",
      gwas_u="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.pheno_strat.glm.linear",
      gwas_Tm="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.pheno_strat.glm.linear",
      gwas_ID="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.pheno_strat.glm.linear",
      gwas_PC=expand("output/Run_GWAS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.pheno_strat.glm.linear",pc=PC)
    output:
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.c.betas",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.c.betas",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.c.betas",
      expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.c.betas", pc=PC),
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.nc.betas",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.nc.betas",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.nc.betas",
      expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.nc.betas", pc=PC)
    params:
      pc_list = get_pc_list(PC)
    shell:
      """
      Rscript code/PRS/clump.R {input.causal_effect} output/Run_GWAS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common output/PRS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common {params.pc_list}
      rm {input.gwas_u} {input.gwas_Tm} {input.gwas_ID} {input.gwas_PC}
      """


# Compute joint effect sizes

rule joint_effects:
    input:
      uc="output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.c.betas",
      Tmc="output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.c.betas",
      IDc="output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.c.betas",
      PCc =expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.c.betas", pc=PC),
      unc="output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.nc.betas",
      Tmnc="output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.nc.betas",
      IDnc="output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.nc.betas",
      PCnc=expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.nc.betas", pc=PC),
      pheno="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.phenos.txt",
      Tm="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tm-ID_covars.txt"
    output:
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.c.betas.joint",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.c.betas.joint",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.c.betas.joint",
      expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.c.betas.joint", pc=PC),
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.nc.betas.joint",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.nc.betas.joint",
      "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.nc.betas.joint",
      expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.nc.betas.joint", pc=PC)
    params:
      pc_list = get_pc_list(PC)
    shell:
      """
      Rscript code/PRS/compute_joint_effect_sizes.R output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-gwas_common output/PRS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common {input.pheno} {input.Tm} {params.pc_list}
      rm {input.uc} {input.Tmc} {input.IDc} {input.unc} {input.Tmnc} {input.IDnc} {input.PCc}  {input.PCnc}
      """

# Do PGA test

rule Calc_Qx:
  input:
    "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.c.betas.joint",
    "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.c.betas.joint",
    "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.c.betas.joint",
    expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.c.betas.joint", pc=PC),
    "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.nc.betas.joint",
    "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-Tm.nc.betas.joint",
    "output/PRS/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common-ID.nc.betas.joint",
    expand("output/PRS/4PopSplit/{{rep}}/{{config}}/{{h2}}/{{ts}}/{{nc}}/{{env}}/genos-gwas_common-{pc}.nc.betas.joint", pc=PC),
    genos="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam",
    Tvec="output/Calculate_FGr/4PopSplit/{rep}/{config}/Tvec.txt",
    pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop",
    es="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.effects.txt"
  output:
    qx="output/PGA_test/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/Qx.txt"
  params:
    num=NUM_RESAMPLE,
    pc_list = get_pc_list(PC)
  shell:
    """
    Rscript code/PGA_test/calcQ_4PopSplit.R output/PRS/4PopSplit/{wildcards.rep}/{wildcards.config}/{wildcards.h2}/{wildcards.ts}/{wildcards.nc}/{wildcards.env}/genos-gwas_common output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common {input.Tvec} {input.pops} {params.num} {output.qx} {input.es} {params.pc_list}
    """

# Calculate true signal magnitude

rule calc_ts_magnitude:
  input:
    psam="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam",
    pops="output/Simulate_Genotypes/4PopSplit/{rep}/genos.pop",
    true_es="output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/genos-gwas_common.effects.txt"
  output:
    "output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{nc}/{env}/ts_magnitude.txt"
  shell:
    "Rscript code/Simulate_Phenotypes/calculate_true_signal_magnitude.R {input.pops} {input.true_es} output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common {output}"


# Bin and average effect size estimates

rule bin_effect_size:
    input:
      gwas="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common.pheno_strat.glm.linear",
      gwas_Tm="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-Tm.pheno_strat.glm.linear",
      gwas_ID="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-ID.pheno_strat.glm.linear",
      r="output/Calculate_Tm/4PopSplit/{rep}/{config}/r.txt"
    output:
      es="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common.mean",
      es_Tm="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-Tm.mean",
      es_ID="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-ID.mean"
    shell:
      """
      Rscript code/Run_GWAS/bin_avg_effect_sizes.R {input.r} {input.gwas} {input.gwas_Tm} {input.gwas_ID} {output.es} {output.es_Tm} {output.es_ID}
      """


# Compute r and bin

rule compute_r:
    input:
      Tvec="output/Calculate_Tm/4PopSplit/{rep}/{config}/Tvec.txt",
      tp="output/Simulate_Genotypes/4PopSplit/{rep}/{config}/genos-test_common.psam"
    output:
      r="output/Calculate_Tm/4PopSplit/{rep}/{config}/r.txt",
      r_bins="output/Calculate_Tm/4PopSplit/{rep}/{config}/r_bins.txt"
    shell:
      """
      Rscript code/Calculate_Tm/compute_and_bin_r.R {input.Tvec} output/Simulate_Genotypes/4PopSplit/{wildcards.rep}/{wildcards.config}/genos-test_common output/Calculate_Tm/4PopSplit/{wildcards.rep}/{wildcards.config}/
      """

rule bin_causal_effect_size:
    input:
      gwas="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common.pheno_strat.glm.linear",
      gwas_Tm="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-Tm.pheno_strat.glm.linear",
      gwas_ID="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-ID.pheno_strat.glm.linear",
      te = "output/Simulate_Phenotypes/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common.effects.txt",
      r="output/Calculate_Tm/4PopSplit/{rep}/{config}/r.txt"
    output:
      es="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common.causal_mean",
      es_Tm="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-Tm.causal_mean",
      es_ID="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/genos-gwas_common-ID.causal_mean",
      es_true="output/Run_GWAS/4PopSplit/{rep}/{config}/{h2}/{ts}/{env}/true_effects.causal_mean"
    shell:
      """
      Rscript code/Run_GWAS/bin_avg_effect_sizes_causal.R {input.r} {input.gwas} {input.gwas_Tm} {input.gwas_ID} {input.te} {output.es} {output.es_Tm} {output.es_ID} {output.es_true}
      """


