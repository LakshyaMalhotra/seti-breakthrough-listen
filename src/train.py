import os
import time

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn import metrics

import dataset, seti_model, utils, run, config, test_model


LOGGER = utils.init_logger()


def get_lr_scheduler(optimizer, scheduler_name):
    """Get learning rate scheduler."""
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
            eps=1e-6,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0, verbose=False
        )
    elif scheduler_name == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=6e-4,
            mode="triangular",
            cycle_momentum=False,
            verbose=False,
        )
    elif scheduler_name == "CyclicLR2":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=6e-4,
            mode="triangular2",
            cycle_momentum=False,
            verbose=False,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.5, verbose=True
        )

    return scheduler


def train_loop(df: pd.DataFrame, fold: int, desc: bool = False):
    LOGGER.info("-" * 30)
    LOGGER.info(f"       Training fold: {fold}       ")
    LOGGER.info("-" * 30)

    # turn off model details for subsequent folds
    if fold >= 1:
        desc = False

    # divide the data into training and validation dataframes based on folds
    train_idx = df[df["fold"] != fold].index
    valid_idx = df[df["fold"] == fold].index

    train_folds = df.loc[train_idx].reset_index(drop=True)
    valid_folds = df.loc[valid_idx].reset_index(drop=True)

    valid_labels = valid_folds["target"].values

    # get the image augmentations
    train_transforms = dataset.get_train_transforms()
    valid_transforms = dataset.get_valid_transforms()

    # create training and validation datasets
    train_data = dataset.SETIDataset(train_folds, transform=train_transforms)
    valid_data = dataset.SETIDataset(valid_folds, transform=valid_transforms)

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = seti_model.SETIModel(model_name=config.MODEL_NAME, pretrained=True)
    # model = test_model.CNNModel()
    LOGGER.info(f"Training with model: {config.MODEL_NAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # print model details
    if desc:
        x = torch.rand(config.BATCH_SIZE, 1, 256, 256)
        x = x.to(device)
        seti_model.model_details(model, x)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        amsgrad=False,
    )

    # learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, scheduler_name=config.SCHEDULER)

    # define the loss function
    loss_func = nn.BCEWithLogitsLoss()

    # define variables for ROC and loss
    best_score = 0
    best_loss = np.inf

    # instantiate training object
    engine = run.Run(
        model,
        device=device,
        criterion=loss_func,
        optimizer=optimizer,
        use_mixup=config.use_mixup,
    )

    # iterate through all the epochs
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        if config.SCHEDULER in ["CosineAnnealingLR", "CyclicLR", "CyclicLR2"]:
            # train
            avg_loss = engine.train(
                train_loader, epoch, scheduler=scheduler, print_every=100
            )

            # evaluate
            avg_val_loss, preds = engine.evaluate(valid_loader, print_every=50)
            scheduler = get_lr_scheduler(
                optimizer, scheduler_name=config.SCHEDULER
            )
        else:
            # train
            avg_loss = engine.train(train_loader, epoch, print_every=100)

            # evaluate
            avg_val_loss, preds = engine.evaluate(valid_loader, print_every=50)

            # step the scheduler
            if config.SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # scoring
        roc_score = metrics.roc_auc_score(valid_labels, preds)
        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}"
        )
        LOGGER.info(
            f"Epoch: {epoch +1}, \tROC-AUC score: {roc_score:.4f}, \ttime elapsed: {elapsed}"
        )

        if roc_score > best_score:
            best_score = roc_score
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Score: {best_score:.4f} Model"
            )
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                os.path.join(
                    config.MODEL_DIR,
                    f"{config.MODEL_NAME}_fold{fold}_best_roc.pth",
                ),
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Loss: {best_loss:.4f} Model"
            )
            # torch.save(
            #     {"model": model.state_dict(), "preds": preds},
            #     os.path.join(
            #         config.MODEL_DIR,
            #         f"{config.MODEL_NAME}_fold{fold}_best_loss.pth",
            #     ),
            # )
    # save the predictions in the valid dataframe
    valid_folds["preds"] = torch.load(
        os.path.join(
            config.MODEL_DIR, f"{config.MODEL_NAME}_fold{fold}_best_roc.pth"
        ),
        map_location=torch.device("cpu"),
    )["preds"]

    return valid_folds


def main(df):
    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df["target"].values
        score = metrics.roc_auc_score(labels, preds)
        LOGGER.info(f"Score: {score:<.4f}")

    oof_df = pd.DataFrame()
    for fold in range(config.FOLDS):
        _oof_df = train_loop(df, fold, desc=True)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info("-" * 30)
        LOGGER.info(f"       Fold: {fold} result        ")
        LOGGER.info("-" * 30)
        get_result(_oof_df)
        break
    # CV result
    LOGGER.info("-" * 30)
    LOGGER.info(f"       CV        ")
    LOGGER.info("-" * 30)
    get_result(oof_df)
    # save result
    oof_df.to_csv(os.path.join(config.OUTPUT_DIR, "oof_df.csv"), index=False)


if __name__ == "__main__":
    csv_path = os.path.join(config.OUTPUT_DIR, "train_folds.csv")
    df = pd.read_csv(csv_path)
    main(df)
