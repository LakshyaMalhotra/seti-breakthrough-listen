import os
import time
import math
import logging

# some utility functions borrowed from:
# https://www.kaggle.com/yasufuminakama/seti-nfnet-l0-starter-training?select=train.log
OUTPUT_DIR = "outputs"


def as_minutes_seconds(s: int) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    m, s = int(m), int(s)
    return f"{m:2d}m {s:2d}s"


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f"{as_minutes_seconds(s)} (remain {as_minutes_seconds(rs)})"


def init_logger(
    log_file: str = os.path.join(OUTPUT_DIR, "train.log")
) -> logging.Logger:
    """Initiate the logger to log the progress into a file.

    Args:
    -----
        log_file (str, optional): Path to the log file. Defaults to os.path.join(OUTPUT_DIR, "train.log").

    Returns:
    --------
        logging.Logger: Logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(message)s"))
    handler2 = logging.FileHandler(filename=log_file)
    handler2.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    """Calculates and stores the average value of the metrics/loss"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all the parameters to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the value of the metrics and calculate their
        average value over the whole dataset.

        Args:
        -----
            val (float): Computed metric (per batch)
            n (int, optional): Batch size. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
