import timm
import torch.nn as nn
from torchinfo import summary

import config


class SETIModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super(SETIModel, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name=self.model_name, pretrained=pretrained, in_chans=1
        )
        if config.MODEL_NAME.startswith("rexnet"):
            self.input_fc = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(
                in_features=self.input_fc, out_features=1, bias=True
            )
        elif config.MODEL_NAME.startswith("efficientnet"):
            self.input_fc = self.model.classifier.in_features
            self.model.classifier = nn.Linear(
                in_features=self.input_fc, out_features=1, bias=True
            )
        elif config.MODEL_NAME.startswith("resnet"):
            self.input_fc = self.model.fc.in_features
            self.model.fc = nn.Linear(
                in_features=self.input_fc, out_features=1, bias=True
            )
        else:
            raise NotImplementedError("Model head has not been modified.")

    def forward(self, x):
        return self.model(x)


def model_details(model, x):
    print("Model summary:")
    summary(
        model,
        input_size=(config.BATCH_SIZE, 1, config.SIZE, config.SIZE),
        verbose=1,
    )
    print(f"Output size: {model(x).shape}")


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SETIModel(model_name=config.MODEL_NAME)
    model = model.to(device)
    x = torch.rand(16, 1, config.SIZE, config.SIZE)
    x = x.to(device)
    model_details(model, x)
