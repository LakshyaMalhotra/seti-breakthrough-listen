import timm
import torch.nn as nn
from torchinfo import summary


class SETIModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super(SETIModel, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name=self.model_name, pretrained=pretrained, in_chans=1
        )
        self.input_fc = self.model.classifier.in_features
        self.model.classifier = nn.Linear(
            in_features=self.input_fc, out_features=1, bias=True
        )

    def forward(self, x):
        return self.model(x)


def model_details(model, x):
    print("Model summary:")
    summary(model, input_size=(16, 1, 256, 256), verbose=1)
    print(f"Output size: {model(x).shape}")


if __name__ == "__main__":
    import torch
    import config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SETIModel(model_name=config.MODEL_NAME)
    model = model.to(device)
    x = torch.rand(1, 1, 256, 256)
    x = x.to(device)
    model_details(model, x)
