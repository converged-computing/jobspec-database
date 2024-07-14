from trankit import Pipeline
from trankit import trankit2conllu
import sys


t = Pipeline('finnish')



for line in sys.stdin:
  line=line.strip()
  print("###C: NEWDOC")
  try:
    all_tu = t(line)
  except:
    print("FAILED ITEM", line)
  tu_conllu = trankit2conllu(all_tu)
  print(tu_conllu)

