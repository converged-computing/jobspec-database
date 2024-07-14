"""
Train single transformer model for downstream analysis
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ecmointerp.modeling.timeseriesCV import TstWrapper, load_to_mem
import pandas as pd
from ecmointerp.dataProcessing.fileBasedDataset import FileBasedDataset
from ecmointerp import config
import os
import datetime


if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tst = TstWrapper(max_epochs=100)
    all_sids = pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
    train_sids, test_sids = train_test_split(all_sids, test_size=0.1, random_state=42)

    train_ds = FileBasedDataset(processed_mimic_path="./mimicts", stay_ids=train_sids)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=tst.batch_size,
        num_workers=config.cores_available,
        pin_memory=True,
    )

    test_ds = FileBasedDataset(processed_mimic_path="./mimicts", stay_ids=test_sids)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=tst.batch_size,
        num_workers=config.cores_available,
        pin_memory=True,
    )

    print("[+] Data loaded, training...")
    tst.train(train_dl=train_dl, use_es=True, valid_dl=test_dl)

    save_path = f"cache/models/singleTst_{start_time_str}"
    print(f"[+] Training complete, saving to {save_path}")
    os.mkdir(save_path)
    torch.save(tst.model.state_dict(), f"{save_path}/model.pt")

    preds = list()
    actuals = list()
    X_all = list()
    pm_all = list()
    for X, y, pm in test_dl:
        tst.model.to("cuda")
        preds.append(tst.decision_function(X.to("cuda"), pm.to("cuda")))
        actuals.append(y)
        X_all.append(X.to("cpu"))
        pm_all.append(pm.to("cpu"))

    preds = torch.cat(preds, dim=0)
    actuals = torch.cat(actuals, dim=0)
    X_all = torch.cat(X_all, dim=0)
    pm_all = torch.cat(pm_all, dim=0)

    score = roc_auc_score(actuals, preds)
    print(f"Validation score: {score}")

    pd.Series(name="stay_id", data=train_sids).to_csv(f"{save_path}/train_sids.csv")
    pd.Series(name="stay_id", data=test_sids).to_csv(f"{save_path}/test_sids.csv")

    torch.save(preds, f"{save_path}/preds.pt")
    torch.save(actuals, f"{save_path}/y.pt")
    torch.save(X_all, f"{save_path}/X.pt")
    torch.save(pm_all, f"{save_path}/pad_masks.pt")

    with open(f"{save_path}/roc_auc_score.txt", "w") as f:
        f.write(str(score))
        f.write("\n")
