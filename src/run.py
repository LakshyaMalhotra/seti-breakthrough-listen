import time
from typing import Tuple

import torch
import numpy as np
import torch.nn as nn

import utils


class Run:
    def __init__(
        self,
        model,
        device: torch.device,
        criterion: nn.BCEWithLogitsLoss,
        optimizer: torch.optim.Adam,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        print_every: int = 100,
    ) -> float:
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()

        # put the model to train mode
        self.model.train()

        start = end = time.time()

        for batch_idx, (images, labels) in enumerate(data_loader):
            # measure the data loading time
            data_time.update(time.time() - end)

            # zero out the accumulated gradients
            self.optimizer.zero_grad()

            # send the data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # forward pass
            y_preds = self.model(images)
            loss = self.criterion(y_preds, labels.unsqueeze(1).type_as(y_preds))

            # record loss
            losses.update(loss.item(), batch_size)

            # backpropagate
            loss.backward()

            # step the optimizer
            self.optimizer.step()

            # measure the elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch: [{epoch+1}][{batch_idx+1}/{len(data_loader)}] "
                    # f"Data loading time: {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Elapsed {utils.time_since(start, float(batch_idx+1)/len(data_loader))} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                )
        return losses.avg

    def evaluate(
        self, data_loader: torch.utils.data.DataLoader, print_every: int = 50
    ) -> Tuple[utils.AverageMeter, np.ndarray]:
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()

        # switch the model to evaluation mode
        self.model.eval()
        preds = []
        start = end = time.time()
        for batch_idx, (images, labels) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # send the data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # compute loss with no backprop
            with torch.no_grad():
                y_preds = self.model(images)

            loss = self.criterion(y_preds, labels.unsqueeze(1).type_as(y_preds))
            losses.update(loss.item(), batch_size)

            # record accuracy
            preds.append(torch.sigmoid(y_preds).cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Evaluating: [{batch_idx+1}/{len(data_loader)}] "
                    # f"Data loading time: {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Elapsed {utils.time_since(start, float(batch_idx+1)/len(data_loader))} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                )
        predictions = np.concatenate(preds)
        return losses.avg, predictions
