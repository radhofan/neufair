import sys

sys.path.append("../")

from utils import preprocess_meps16
import torch
import torch.nn as nn
from model import MEPS16Model
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import copy
import os

seeds = [
    36821596,
    42,
    398198020,
    123,
    2579600519,
    42,
    3569605705,
    123,
    608130088,
    289553502,
    525558053,
    2123741781,
]

for seed in seeds:

    np.random.seed(seed)
    torch.manual_seed(seed)

    scaler = "mms"
    p = 0.1
    X_train, X_test, X_val, y_train, y_test, y_val, sens_idx = preprocess_meps16(
        seed, scaler=scaler
    )

    model = MEPS16Model(p=p)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.BCEWithLogitsLoss()

    epochs = 1000

    batch_size = 128

    train_loss, test_loss = 0, 0

    # Weighted random sampler
    class_weights = [1 / (y_train == i).sum() for i in range(2)]
    sample_weights = [class_weights[int(i)] for i in y_train]
    train_dataset = TensorDataset(X_train, y_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    best_model = None
    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        for x_i, y_i in train_loader:
            optimizer.zero_grad()

            y_logits = model(x_i).squeeze(1)
            loss = criterion(y_logits, y_i)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze()
            val_loss = criterion(val_logits, y_val)
            val_pred = torch.round(torch.sigmoid(val_logits))
            val_f1 = f1_score(
                y_val.detach().numpy(), val_pred.detach().numpy(), zero_division="warn"
            )

            train_logits = model(X_train).squeeze()
            train_pred = torch.round(torch.sigmoid(train_logits))
            train_f1 = f1_score(
                y_train.detach().numpy(),
                train_pred.detach().numpy(),
                zero_division="warn",
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            train_loss = round(loss.item(), 4)
            val_loss_val = round(val_loss.item(), 4)
            print(f"Epoch: {epoch + 1} | Loss: {train_loss} | Val Loss: {val_loss_val}")

    if not os.path.exists("../saved_models/meps16"):
        os.makedirs("../saved_models/meps16")

    torch.save(
        best_model,
        "../saved_models/meps16/meps16-{}-{}-{}-{}.pt".format(
            round(best_val_f1, 3), seed, scaler, p
        ),
    )
