import pandas as pd
import pickle as pkl
data=pd.read_stata("data/BloombergVOTELEVEL_Touse.dta",chunksize=100,convert_categoricals=False)
data_sliced=pd.DataFrame()
print("started slicing")
for chunk in data:
    data_sliced = data_sliced.append(chunk[['Circuit', 'caseid', 'date']], ignore_index=True)

print("started parsing date")
data_sliced['date']=data_sliced['date'].dt.year

print("started creating dictionary of circuit year level")
groups = dict(list(data_sliced.groupby(['Circuit','date'])['caseid']))

print('dumping into circuit_year_level')
pkl.dump(groups, open("circuit_year_level", "wb"))
