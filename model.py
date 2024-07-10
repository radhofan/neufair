import torch.nn as nn
import torch


class AdultCensusModel(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.layer1 = nn.Linear(in_features=34, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=64)
        self.layer4 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = self.layer4(x)
        return x


class BankModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32, out_features=32)
        self.layer2 = nn.Linear(in_features=32, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.layer3(x)
        return x


class DefaultModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=30, out_features=16)
        self.layer2 = nn.Linear(in_features=16, out_features=16)
        self.layer3 = nn.Linear(in_features=16, out_features=16)
        self.layer4 = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = self.layer4(x)
        return x


class CompasModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=12, out_features=32)
        self.layer2 = nn.Linear(in_features=32, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.layer3(x)
        return x


class MEPS16Model(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.layer1 = nn.Linear(in_features=138, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = self.layer4(x)
        return x
