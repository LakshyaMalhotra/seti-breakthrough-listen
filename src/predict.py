import os

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import config, dataset, seti_model

if __name__ == "__main__":
    # load the test data and read the dataframe
    test_df = pd.read_csv(
        os.path.join(config.DATA_DIR, "sample_submission.csv")
    )
    test_df["file_path"] = test_df["id"].apply(
        lambda x: os.path.join(config.DATA_DIR, f"test/{x[0]}/{x}.npy")
    )

    # define the transforms and create test dataset
    transforms = dataset.get_valid_transforms()
    test_ds = dataset.SETIDataset(test_df, transform=transforms)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE * 8,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # checkpoints for k-folds for best ROC and loss
    roc_ckpts = [
        os.path.join(
            config.MODEL_DIR, config.MODEL_NAME + f"_fold{i}_best_roc.pth"
        )
        for i in range(config.FOLDS)
    ]
    loss_ckpts = [
        os.path.join(
            config.MODEL_DIR, config.MODEL_NAME + f"_fold{i}_best_loss.pth"
        )
        for i in range(config.FOLDS)
    ]

    # instantiate the model and load the checkpoint
    model = seti_model.SETIModel(model_name=config.MODEL_NAME, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    for i in range(len(roc_ckpts)):
        print(f"Loading model for fold: {i}")
        model.load_state_dict(
            torch.load(roc_ckpts[i], map_location=device)["model"]
        )
        model = model.to(device)

        # perform inference
        model.eval()
        preds = []
        for images, _ in tqdm(test_loader):
            images = images.to(device)

            with torch.no_grad():
                y_preds = model(images)
            preds.append(torch.sigmoid(y_preds).cpu().numpy())
        predictions = np.concatenate(preds)
        test_df[f"target_fold{i}"] = predictions
    test_df["target"] = np.mean(
        test_df[[f"target_fold{i}" for i in range(len(roc_ckpts))]], axis=1
    )
    test_df[["id", "target"]].to_csv(
        os.path.join(config.OUTPUT_DIR, f"avg_roc_{config.MODEL_NAME}.csv"),
        index=False,
    )
    print(test_df.head())
