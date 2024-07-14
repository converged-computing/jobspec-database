import sys
import pandas as pd
import numpy as np
import itertools as it
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

bw_results_dir = sys.argv[1]

intra_df = pd.read_csv(
    f"{bw_results_dir}/bw_intra_send.data", header=None
).rename(
    columns={0: "data_size", 1: "time"}
)

intra_latency = intra_df.loc[np.min(intra_df["data_size"])]
intra_df = intra_df.drop([0])
intra_df["bandwidth"] = intra_df["data_size"]/(intra_df["time"] - intra_latency)

intra_df.to_csv(f"{bw_results_dir}/bw_intra.data", header=False, index=False)

# inter_df = pd.read_csv(
#     f"{bw_results_dir}/bw_inter_send.data", header=None
# ).rename(
#     columns={0: "data_size", 1: "time"}
# )

# inter_latency = inter_df.loc[np.min(inter_df["data_size"])]
# inter_df = inter_df.drop([0])
# inter_df["bandwidth"] = inter_df["data_size"]/(inter_df["time"] - inter_latency)

# inter_df.to_csv(f"{bw_results_dir}/bw_inter.data", header=False, index=False)