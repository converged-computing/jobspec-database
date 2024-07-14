from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
#from tqdm.notebook import tqdm
from tqdm import tqdm
import bm
os.chdir(Path(bm.__file__).parent.parent)
from bm import train

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

print("Complete import...")

output_dir = train.main.dora.dir
print(output_dir)
eval_dir = output_dir / "eval" / "signatures"
sigs_to_eval = [p.name for p in (output_dir / "grids" / "nmi.main_table").iterdir()]
print(sigs_to_eval)
assert output_dir.exists()
assert eval_dir.exists()

# Common function
def load_data_from_sig(sig, level="segment"):
    """
    Load data from solver signature
    - probs (torch tensor): probability on vocab [N, V]
    - vocab (torch tensor): vocab of word hashes [V]
    - words (torch tensor): the word hash for each sample [N]
    - metadata (panda dataframe) of len [N] which contains for each sample:
           'word_hashes', 'word_indices', 'seq_indices',
           'word_strings', 'subject_id', 'recording_id'
    """
    assert level in ["word", "segment"], "level should be 'word' or 'segment'"
    probs = torch.load(eval_dir / sig / f"probs_{level}.pth") 
    vocab = torch.load(eval_dir / sig / f"vocab_{level}.pth") # vocab (hashes)
    metadata = pd.read_csv(eval_dir / sig / "metadata.csv", index_col=0, low_memory=False) 
    
    words = torch.tensor(metadata[f"{level}_hashes"].tolist()).long() # for each sample, the word (hashes)
    assert probs.shape == (len(words), len(vocab)), (probs.shape, len(words), len(vocab))
    assert len(words) == len(metadata)
    metadata["idx"] = range(len(words))
    metadata["word_strings"] = metadata["word_strings"].str.lower()

    return probs, vocab, words, metadata


def get_accuracy_from_probs(probs, row_labels, col_labels, topk=10):
    """
    probs: for each row, the probability distribution over a vocab
    returns the topk accuracy that the topk best predicted labels
    match the row_labels
    Inputs:
        probs: of shape [B, V] probability over vocab, each row sums to 1
        row_labels: of shape [B] true word for each row
        col_labels: [V] word that correspond to each column
        topk: int
    Returns: float scalar, topk accuracy
    """
    assert len(row_labels) == len(probs)
    assert len(col_labels) == probs.shape[1]

    # Extract topk indices
    idx = probs.topk(topk, dim=1).indices

    # Get the corresponding topk labels
    whs = col_labels[idx.view(-1)].reshape(idx.shape)

    # 1 if the labels matches with the targets
    correct = ((whs == row_labels[:, None]).any(1)).float()
    assert len(correct) == len(row_labels)

    # Average across samples
    acc = correct.mean()

    return acc.item()


def eval_acc_one_sig(sig, topks=(1, 5, 10), level="word", add_baselines=True):
    """
    Return accuracy dataframe from one solver signature
    level: whether to return `word` or `segment` level accuracy
    """
    # Load data
    probs, vocab, words, _ = load_data_from_sig(sig, level=level)
#     if level == "segment":
#         print(probs.shape)
        
    # Compute acc
    acc_df = []
    for topk in topks:
        
        # --- Acc ---
        acc = get_accuracy_from_probs(probs, words, vocab, topk=topk)
        
        out = {
            f"acc":acc,
            "topk":topk,
        }
        
        if add_baselines:
        
            # --- Baseline on vocab ---
            # equivalent to : shuffle targets vocab (inf times)
            # equivalent to : output uniform prob on vocab
            # equivalent to : 1/vocab_len
            rand_probs_vocab = torch.ones_like(probs) / len(vocab)
            out["baseline_vocab"] = get_accuracy_from_probs(rand_probs_vocab, words, vocab, topk=topk)

            # --- Baseline on words ---
            # equivalent to : shuffle word targets before aggregating on vocab (inf times)
            # equivalent to : output uniform prob on samples
            # equivalent to : each_word_freq
            check_vocab, counts = torch.unique(words, return_counts=True)
            import pdb
#             assert (check_vocab == vocab).all()
            rand_probs_words = torch.stack([counts/len(words)]*len(probs))
            out["baseline"] = get_accuracy_from_probs(rand_probs_words, words, vocab, topk=topk)

            # Update
            acc_df.append(out)
    acc_df = pd.DataFrame(acc_df)
    return acc_df

def eval_acc(sigs, level="word", add_baselines=True):
    """
    Return accuracy dataframe for multiple sigs 
    level: whether to return word or segment level accuracy
    """
    futures = []
    acc = []
    with ProcessPoolExecutor(20) as pool:
        for sig in sigs:
            future = pool.submit(eval_acc_one_sig, sig, level=level, add_baselines=add_baselines)
            futures.append((sig, future))
        for sig, future in tqdm(futures):
            try:
                acc_sig = future.result()
            except Exception:
                print("ERROR WITH", sig)
                raise
                continue
            acc_sig["sig"] = sig
            acc.append(acc_sig)
    acc = pd.concat(acc)
    return acc

# Load meta dataframe
# Select signatures
valid_sigs = [sig for sig in sigs_to_eval if (eval_dir / sig / "vocab_segment.pth").is_file()]
configs = [OmegaConf.load(eval_dir / sig / "solver_config.yaml") for sig in valid_sigs]
for c, s in zip(configs, valid_sigs):
    if not hasattr(c.dset, 'features'):
        c.dset.features = c.dset.forcings
        c.dset.features_params = c.dset.forcings_params

print("sigs_to_eval: ", set(sigs_to_eval))
print("valid_sigs: ", set(valid_sigs)) 
print(set(sigs_to_eval) - set(valid_sigs))
run_df = pd.DataFrame({
    "sig":valid_sigs,
})
run_df["dataset"] = ["-".join(conf.dset.selections) for conf in configs]
run_df["seed"] = [conf.seed for conf in configs]
run_df["forcings"] = ["-".join(conf.dset.features) for conf in configs]
run_df["loss"] = [conf.optim.loss for conf in configs]
run_df["is_random"] = [conf.test.wer_random for conf in configs]
run_df["max_scale"] = [conf.norm.max_scale for conf in configs]
run_df["n_mels"] = [conf.dset.features_params.MelSpectrum.n_mels for conf in configs]
run_df["deepmel"] = [getattr(conf, 'feature_model_name', None) == 'deep_mel' for conf in configs]
run_df["ft"] = [conf.optim.epochs == 1 and not conf.test.wer_random for conf in configs]
run_df["random"] = [conf.test.wer_random for conf in configs]

run_df["batch_size"] = [conf.optim.batch_size for conf in configs]
run_df["lr"] = [conf.optim.lr for conf in configs]
run_df["autorej"] = [conf.dset.autoreject for conf in configs]
run_df["n_rec"] = [conf.dset.n_recordings for conf in configs]
# run_df["ft"] = [conf.optim.lr == 0 for conf in configs]
run_df["dropout"] = [conf.simpleconv.merger_dropout > 0 for conf in configs]
run_df["gelu"] = [bool(conf.simpleconv.gelu) for conf in configs]
run_df["skip"] = [bool(conf.simpleconv.skip) for conf in configs]
run_df["initial"] = [bool(conf.simpleconv.initial_linear) for conf in configs]
run_df["complex"] = [bool(conf.simpleconv.complex_out) for conf in configs]
run_df["subject_lay"] = [bool(conf.simpleconv.subject_layers) for conf in configs]
run_df["subject_emb"] = [bool(conf.simpleconv.subject_dim) for conf in configs]
run_df["attention"] = [bool(conf.simpleconv.merger) for conf in configs]
run_df["glu"] = [bool(conf.simpleconv.glu) for conf in configs]
run_df["depth"] = [conf.simpleconv.depth for conf in configs]
run_df["offset_meg"] = [conf.task.offset_meg_ms for conf in configs]
run_df = run_df[run_df.loss == "clip"]
print("run_df: ", run_df)
# Uncomment for running on specific dataset
#run_df = run_df[run_df.dataset == "broderick2019"]
print("run_df: ", run_df)

def get_name(row):
    if row.ft:
        return "Trained with MSE"
    if row.random:
        return "Random model"
    if row.deepmel:
        return "Deep Mel"
    if row.forcings == 'MelSpectrum':
        return 'MelSpectrum CLIP'
    if not row.dropout:
        return r"\wo spatial attention dropout"
    if not row.gelu:
        return r"\wo GELU, \w ReLU"
    if not row.skip:
        return r"\wo skip connections"
    if not row.initial:
        return r"\wo initial 1x1 conv."
    if not row.complex:
        return r"\wo final convs"
    if not row.attention:
        return r"\wo spatial attention"
    if not row.glu:
        return r"\wo non-residual GLU conv."
    if not row.subject_lay:
        if row.subject_emb:
            return r"\w subj. embedding*"
        else:
            return r"\wo subject-specific layer"
    if row.depth == 5:
        return r"less deep"
    if row.autorej:
        return "autoreject"
    if row.max_scale != 20:
        if row.max_scale == 100:
            return "\w clamp=100"
        else:
            return "\wo clamping brain signal"
    return "Our model"
    
        
run_df['name'] = run_df.apply(get_name, axis=1)

print("name: ", run_df['name'])
print("sig: ", run_df['sig'])

# Accuracy table
#%%time
acc_df = eval_acc(run_df["sig"].values, level="segment")
# acc_df = pd.merge(acc_df, on=["sig", "topk"], how="outer")
acc_df = pd.merge(acc_df, run_df, on="sig", how="left")
def dset_order(name):
    return name.map({
       'broderick2019': 0,
       #'audio_mous': 0, 
       #'gwilliams2022': 1,
       #'broderick2019': 2,
       #'brennan2019': 3,
    })
acc_df = acc_df.sort_values(["dataset"], key=dset_order)

print(acc_df)

for name in run_df.name.unique():
    print(name, ((run_df.name == name) & (run_df.dataset =='broderick2019')).sum())

# Keys to set depends on the Table we want to generate
keys = ["dataset", "name"]
# keys = ["dataset", "name", "lr", "batch_size", "offset_meg", "n_rec"]
# keys = ["dataset", "name", "lr", "batch_size"]
# keys = ["dataset", "name", "n_mels"]
# keys = ["dataset", "offset_meg"]
acc_table = acc_df
# acc_table = acc_df.query('batch_size == 256 & lr==3e-4 & n_rec != 16')
# acc_table = acc_df.query('batch_size == 256 & lr==3e-4 & n_rec == 16')
acc_table = acc_table.query("topk==10").sort_values(keys).groupby(keys)["acc"].agg(["mean", "std"])
key = "acc"
acc_table["str_acc"] = (100 * acc_table["mean"]).round(1).astype(str) + r" PM " + (100 * acc_table["std"]).round(2).astype(str)
print(acc_table)

def convert(x):
    if isinstance(x, float):
        print(x)
    return float(x.split(" ")[0])
    
toplot = acc_table.reset_index()
index = list(keys)
index.remove('dataset')
# index.remove('n_mels')
# index.remove('batch_size')
toplot =  pd.pivot_table(toplot, values=["str_acc"], columns=["dataset"], index=index, aggfunc="first")

toplot[('str_acc', "mean_dataset")] = toplot.applymap(convert).mean(axis=1).round(1)
# toplot.sort_values(index, ascending=False)
# dsets = ['audio_mous', 'gwilliams2022', 'broderick2019', 'brennan2019'][:-1]
#dsets = ['audio_mous', 'gwilliams2022', 'broderick2019', 'brennan2019']
dsets = ["broderick2019"]
# dsets = []
extra = []

toplot = toplot[[('str_acc', dset) 
                 for dset in dsets + ['mean_dataset'] +  extra]]
# toplot = acc_table.reset_index()
# index.remove('n_mels')
# toplot =  pd.pivot_table(toplot, values=["str_acc"], 
#                          columns=["n_mels"], index=index, aggfunc="first")

print(toplot.to_latex(index=True).replace('PM', r'$\pm$'))
