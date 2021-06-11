import os
import numpy as np
import pandas as pd
from sklearn import model_selection

import config

if __name__ == "__main__":
    # read the data
    train = pd.read_csv(os.path.join(config.DATA_DIR, "train_labels.csv"))
    test = pd.read_csv(os.path.join(config.DATA_DIR, "sample_submission.csv"))

    # add file path for each image id
    train["file_path"] = train["id"].apply(
        lambda x: os.path.join(config.DATA_DIR, f"train/{x[0]}/{x}.npy")
    )
    test["file_path"] = test["id"].apply(
        lambda x: os.path.join(config.DATA_DIR, f"test/{x[0]}/{x}.npy")
    )

    # stratified k-fold
    train["fold"] = -1
    kf = model_selection.StratifiedKFold(
        n_splits=4, shuffle=True, random_state=23
    )
    for f_, (t_, v_) in enumerate(kf.split(X=train, y=train.target)):
        train.loc[v_, "fold"] = f_
    train["fold"] = train["fold"].astype(int)
    train.to_csv(
        os.path.join(config.OUTPUT_DIR, "train_folds.csv"), index=False
    )

    print(train.head())
    print(train.groupby(["fold", "target"])["id"].count())
    print(test.head())
