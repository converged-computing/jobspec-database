# This snakefile creates input fastq files for the Mock Communities Salmonella Detection project.

PAIR = [1,2]
reps = config['reps']
bgnum = config['libsize'] - config['salmnum']

print("Creating metagenome files with " + str(config['salmnum']) + f" salmonella reads and {bgnum} additional reads.")

rule all:
  input:
    expand("Library/Mockcomm" + str(config['salmnum']) + "/Mock_salm" + str(config['salmnum']) + "rep{r}_R{pair}.fq.gz", r=range(1,reps), pair=PAIR)


for r in range(1,reps):
  rule:
    input: 
      file_1 = config['pathtosalm'] + config['salmfile1'],
      file_2 = config['pathtosalm'] + config['salmfile2']
    output:
      out1 = temp("Library/Mockcomm" + str(config['salmnum']) + f"/salm{r}_R1.fq.gz"),
      out2 = temp("Library/Mockcomm" + str(config['salmnum']) + f"/salm{r}_R2.fq.gz")
    params:
      num = config['salmnum']
    shell:
      "reformat.sh in1={input.file_1} in2={input.file_2} \
      out1={output.out1} out2={output.out2} \
      samplereadstarget={params.num} zl=6 int=f"

for r in range(1,reps):
  rule:
    input:
      file_1 = config['pathtobg'] + config['bgfile1'],
      file_2 = config['pathtobg'] + config['bgfile2']
    output:
      out1 = temp("Library/Mockcomm" + str(config['salmnum']) + f"/comm{r}_R1.fq.gz"),
      out2 = temp("Library/Mockcomm" + str(config['salmnum']) + f"/comm{r}_R2.fq.gz")
    params:
      num = bgnum
    shell:
      "reformat.sh in1={input.file_1} in2={input.file_2} \
      out1={output.out1} out2={output.out2} \
      samplereadstarget={params.num} zl=6 int=f"

rule concat_metagenome:
  input:
      salm = "Library/Mockcomm" + str(config['salmnum']) + "/salm{r}_R{pair}.fq.gz",
      bg = "Library/Mockcomm" + str(config['salmnum']) + "/comm{r}_R{pair}.fq.gz"
  output:
      "Library/Mockcomm" + str(config['salmnum']) + "/Mock_salm" + str(config['salmnum']) + "rep{r}_R{pair}.fq.gz"
  shell:
      "cat {input.salm} {input.bg}  > {output}"
