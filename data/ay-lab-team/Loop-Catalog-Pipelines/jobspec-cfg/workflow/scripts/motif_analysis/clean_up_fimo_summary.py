import os
import sys
import pandas as pd
os.chdir('/mnt/bioadhoc-temp/Groups/vd-ay/kfetter/hichip-db-loop-calling')

sample = sys.argv[1]
base = "biorep_merged/results/motif_analysis/meme/fimo/" + sample + "/summarize_results/"
f = base + "loops_overlap_motifs.sorted.uniq.txt"

df = pd.read_csv(f, sep = "\t", header=None)
columns = ["chr1", "start1", "end1", "chr2", "start2", "end2", "motif_chr", "motif_start", "motif_end", "motif_ID", "motif_name"]
df.columns = columns

df_1 = df.copy()
df_1['loop'] = df_1.apply(lambda x: str(x["chr1"]) + "/" + str(x["start1"]) + "/" +  str(x["end1"]) + "/" + str(x["chr2"]) + "/" + str(x["start2"]) + "/" + str(x["end2"]), axis=1)
df_1['anchor'] = df_1.apply(lambda x: "1" if x["motif_start"] >= x["start1"] and x["motif_start"] <= x["end1"] else "2", axis=1)
df_1['anchor_1'] = df_1.apply(lambda x: str(x["chr1"]) + "/" + str(x["start1"]) + "/" + str(x["end1"]), axis=1)
df_1['anchor_2'] = df_1.apply(lambda x: str(x["chr2"]) + "/" + str(x["start2"]) + "/" + str(x["end2"]), axis=1)

#df_1['anchor'] = df_1.apply(lambda x: "1" if x["motif_start"] >= x["start1"] and x["motif_start"] <= x["end1"] else "2", axis=1)
df_1['motif_ID_1'] = df_1.apply(lambda x: x["motif_ID"] if x["anchor"] == "1" else "", axis=1)
df_1['motif_name_1'] = df_1.apply(lambda x: x["motif_name"] if x["anchor"] == "1" else "", axis=1)
df_1['motif_ID_2'] = df_1.apply(lambda x: x["motif_ID"] if x["anchor"] == "2" else "", axis=1)
df_1['motif_name_2'] = df_1.apply(lambda x: x["motif_name"] if x["anchor"] == "2" else "", axis=1)

df_1 = df_1.drop(columns=["motif_chr", "motif_start", "motif_end", "motif_ID", "motif_name", "anchor", "anchor_1", "anchor_2"])
df_1 = df_1.groupby(["loop"]).agg({"chr1" : "first", "start1" : "first", "end1" : "first", "chr2" : "first", "start2" : "first", "end2" : "first", "motif_ID_1" : pd.Series.unique, "motif_name_1" : pd.Series.unique, "motif_ID_2" : pd.Series.unique, "motif_name_2" : pd.Series.unique})
df_1 = df_1.sort_values(by=["chr1", "start1", "start2"])

df_1['motif_ID_1'] = df_1.apply(lambda x: ",".join(x["motif_ID_1"]), axis=1)
df_1['motif_ID_1'] = df_1.apply(lambda x: x["motif_ID_1"].strip(","), axis=1)
df_1['motif_ID_1'] = df_1.apply(lambda x: "None" if len(x["motif_ID_1"]) == 0 else x["motif_ID_1"], axis=1)

df_1['motif_name_1'] = df_1.apply(lambda x: ",".join(x["motif_name_1"]), axis=1)
df_1['motif_name_1'] = df_1.apply(lambda x: x["motif_name_1"].strip(","), axis=1)
df_1['motif_name_1'] = df_1.apply(lambda x: "None" if len(x["motif_name_1"]) == 0 else x["motif_name_1"], axis=1)

df_1['motif_ID_2'] = df_1.apply(lambda x: ",".join(x["motif_ID_2"]), axis=1)
df_1['motif_ID_2'] = df_1.apply(lambda x: x["motif_ID_2"].strip(","), axis=1)
df_1['motif_ID_2'] = df_1.apply(lambda x: "None" if len(x["motif_ID_2"]) == 0 else x["motif_ID_2"], axis=1)

df_1['motif_name_2'] = df_1.apply(lambda x: ",".join(x["motif_name_2"]), axis=1)
df_1['motif_name_2'] = df_1.apply(lambda x: x["motif_name_2"].strip(","), axis=1)
df_1['motif_name_2'] = df_1.apply(lambda x: "None" if len(x["motif_name_2"]) == 0 else x["motif_name_2"], axis=1)

df_1 = df_1.reset_index().drop(columns=["loop"])
df_1.to_csv(base + "summary.txt", sep="\t", header=True, index=False)