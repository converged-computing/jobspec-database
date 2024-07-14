import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import dmatrix
import pickle
import os

parser = argparse.ArgumentParser(description='Run Stan negative binomial on a single microbe.')
parser.add_argument('--biom', action="store", type=str, help='Biom table')
parser.add_argument('--metadata', action="store", type=str, help='Sample metadata table')
parser.add_argument('--batch', action="store", type=str, help='Batch parameters')
parser.add_argument('--model', action="store", type=str, help='Stan model path')
parser.add_argument('--microbe', action="store", type=int, help='Microbe number')
parser.add_argument('--output-file', action="store", type=str, help='Output pickle file')
args = parser.parse_args()


#fname = '../data/File_1_TaoDing_microbiome.biom'
table = load_table(args.biom)

# fname = 'File_3_FerretMicrobiomeMetadata.xlxs.xlsx'
# fname = os.path.join('../data/', fname)
metadata = pd.read_excel(args.metadata)
pckl = pickle.load(open(args.batch, 'rb'))
batch_metadata = metadata.set_index('sampleID')
batch_metadata = batch_metadata.reset_index()

# filter and match
batch_filter = lambda v, i, m: i in set(batch_metadata.SeqID)
read_filter = lambda v, i, m: v.sum()>100
obs_filter = lambda v, i, m: (v>0).sum()>5

table.filter(batch_filter, axis='sample')
table.filter(read_filter, axis='sample')
table.filter(obs_filter, axis='observation')

# Drop samples that cannot be converted to days
# batch_metadata['infected'] = batch_metadata['Days'] != 'uninfected'
# batch_metadata.loc[batch_metadata['Days']=='unknown', 'infected'] = True
# batch_metadata.loc[~batch_metadata['infected'], 'Days'] = 0
# batch_metadata.loc[batch_metadata['Days']=='unknown', 'Days'] = 0
idx = batch_metadata['SampleType'] == 'NW'
batch_metadata = batch_metadata.loc[idx]
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

idx = batch_metadata['Days'].apply(is_float)
batch_metadata = batch_metadata.loc[idx]
batch_metadata['Days'] = batch_metadata['Days'].astype(np.float)

tab = table.to_dataframe().T.to_dense()
tab = tab.loc[list(batch_metadata.SeqID.values)].dropna()
batch_metadata = batch_metadata.set_index('SeqID')
batch_metadata = batch_metadata.loc[tab.index]
# batch_metadata = batch_metadata.reset_index()
tab = tab.loc[batch_metadata.index]
tab = tab.loc[:, pckl['biom'].columns]

formula = 'Days'
X = dmatrix(formula, batch_metadata, return_type='dataframe')

# prepare ids for stan
subj_ids = batch_metadata.AnimalID.unique()
id_lookup = pd.Series(np.arange(len(subj_ids)), index=subj_ids)
per_sample_subj_ids = batch_metadata.AnimalID.apply(lambda x: id_lookup[x]).values + 1
run_ids = batch_metadata.runID.unique()
id_lookup = pd.Series(np.arange(len(run_ids)), index=run_ids)
per_sample_batch_ids = batch_metadata.runID.apply(lambda x: id_lookup[x]).values + 1

# Convert gamma to clr coordinates
gamma = pckl['res']['bdiff'].mean(axis=0)
r, d = gamma.shape
gamma = np.hstack((np.zeros((r, 1)), gamma))
gamma = gamma - gamma.mean(axis=1).reshape(-1, 1)


# Actual stan modeling
import pystan

code = open(args.model, 'r').read()
sm = pystan.StanModel(model_code=code)

dat = {
    'N' : X.shape[0],
    'R' : len(set(per_sample_batch_ids)),
    'p' : X.shape[1],
    'J' : len(set(per_sample_subj_ids)),
    'depth' : np.log(tab.sum(axis=1).values),
    'x' : X.values,
    'y' : tab.iloc[:, args.microbe].values.ravel().astype(np.int64),
    'subj_ids' : list(per_sample_subj_ids),
    'batch_ids' : list(per_sample_batch_ids),
    'gamma' : gamma[:, args.microbe].ravel()
}

fit = sm.sampling(data=dat, iter=1000, chains=4)
res =  fit.extract(permuted=True)

# Save  model
pickle.dump({'dat': dat, 'res': res,
             'microbes' : list(tab.columns),
             'samples' : list(tab.index)},
             open(args.output_file, 'wb'))
