
# get list of SRA sequence names from GSEst of SRA sequence names from GSE

import GEOparse
import csv

def write_all(gse_name, meta):
  with open(gse_name+".txt","w") as txtfile:
    writer = csv.writer(txtfile, dialect = "excel-tab")
    for m in meta:
      writer.writerow((m["sra"],m["gsm"],m["title"]))

def write_sra(gse_name, meta):
  with open(gse_name+"_sra.txt","w") as txtfile:
    for m in meta:
      txtfile.write(m["sra"]+"\n")

# use GEOpar.get_GEO to 
def make_meta(gse_name):
  gse = GEOparse.get_GEO(geo=gse_name)
  gsms = list(gse.gsms.keys())
  gsms.sort()

  meta = [{"gsm": gsm, "sra": gse.gsms[gsm].relations["SRA"][0].split("=")[1], "title":gse.gsms[gsm].metadata["title"][0]} for gsm in gsms] 
  return(meta)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = "Get GSE ID")
  parser.add_argument("gse", metavar="gse", type = str, help = "NCBI GSE collection ID")
  args = parser.parse_args()
  gse_name = args.gse
  #gse_name = "GSE115218"
  print(gse_name)
  meta = make_meta(gse_name)
  write_sra(gse_name,meta)
  write_all(gse_name,meta)

