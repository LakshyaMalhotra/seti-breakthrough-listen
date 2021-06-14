from typing import Tuple
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

# dataset class
class SETIDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        """Dataset class to create the pytorch dataset. It expects the user
        to provide the dataframe containing the metadata.

        Args:
        -----
            df (pd.DataFrame): Image metadata.
            transform ([type], optional): Image augmentations. Defaults to None.
        """
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        label = self.df.loc[idx, "target"]
        image_path = self.df.loc[idx, "file_path"]
        image = np.load(image_path)[[0, 2, 4]]
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = image[None, :, :]
            image = image.from_numpy(image).float()

        return image, label


# define the transforms
def get_train_transforms():
    return A.Compose(
        [
            A.Resize(config.SIZE, config.SIZE),
            A.OneOf([A.GaussNoise(), A.MultiplicativeNoise(p=0.1)], p=0.25),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                ],
                p=0.2,
            ),
            A.ElasticTransform(p=0.3),
            ToTensorV2(),
        ]
    )


def get_train_transforms_v2():
    return A.Compose(
        [
            A.Resize(config.SIZE, config.SIZE),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=1.15),
                    A.MultiplicativeNoise(multiplier=1.1),
                ],
                p=0.2,
            ),
            A.RandomBrightnessContrast(
                contrast_limit=0.12, brightness_limit=0.12, p=0.2
            ),
            A.OpticalDistortion(distort_limit=0.07, shift_limit=0.07, p=0.25),
            A.GaussianBlur(p=0.15),
            A.RandomGridShuffle(grid=(4, 4), p=0.2),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    return A.Compose([A.Resize(config.SIZE, config.SIZE), ToTensorV2()])


if __name__ == "__main__":
    import os

    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "train_folds.csv"))
    transforms = get_train_transforms()
    data = SETIDataset(df, transforms)
    image, label = data.__getitem__(0)
    print(image.size())
    print(label)
