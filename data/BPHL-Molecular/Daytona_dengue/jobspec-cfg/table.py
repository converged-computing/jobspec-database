
report = "./output/sum_report.txt"
seroreport = "./output/Serotypes.txt"
finalreport = "./output/final_report.txt"
with open(report, 'r') as report:

  head = report.readline()
  datalines = report.readlines()
  
  headcells = head.split("\t")
  headcells.insert(1,'Seotype')
  headcells.insert(2,'Kraken2_Viral_Broad_Percentage')
  
  f = open(finalreport, 'a')
  f.write('\t'.join(headcells))
  
  report2=open(seroreport,'r')
  serolines=report2.readlines()[1:]
  
  for aline in datalines:
    linecells = aline.split("\t")
    #subcells = linecells[0].split("_")
    #linecells[0] = subcells[1]
    for seroline in serolines:
      seroline = seroline.strip()
      if linecells[0] in seroline:
        serocells = seroline.split(",")
        linecells.insert(1,serocells[1])
        linecells.insert(2,serocells[2])
        f.write('\t'.join(linecells))
        break
  f.close()
  report2.close()
