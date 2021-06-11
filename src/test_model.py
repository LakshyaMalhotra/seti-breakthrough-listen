import torch
import torch.nn as nn
from torchinfo import summary


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = self.relu(self.conv4(x))
        x = self.maxpool(x)

        x = self.relu(self.conv5(x))
        x = self.maxpool(x)

        x = x.view(1, -1)

        x = self.relu(self.dropout(self.fc1(x)))

        x = self.fc2(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel()
    x = torch.rand(1, 1, 256, 256)
    model, x = model.to(device), x.to(device)
    summary(model, input_size=(1, 1, 256, 256))
    output = model(x)
    print(output.shape)
